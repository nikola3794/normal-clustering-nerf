import torch
import torch.nn.functional as F
from .custom_functions import \
    RayAABBIntersector, RayMarcher, VolumeRenderer
from einops import rearrange
import vren


@torch.cuda.amp.autocast()
def render(model, rays_o, rays_d, **kwargs):
    """
    Render rays by
    1. Compute the intersection of the rays with the scene bounding box
    2. Follow the process in @render_func (different for train/test)

    Inputs:
        model: NGP
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions

    Outputs:
        result: dictionary containing final rgb and depth
    """
    near_distance = kwargs['near_distance']
    rays_o = rays_o.contiguous(); rays_d = rays_d.contiguous()
    _, hits_t, _ = \
        RayAABBIntersector.apply(rays_o, rays_d, model.center, model.half_size, 1)
    hits_t[(hits_t[:, 0, 0]>=0)&(hits_t[:, 0, 0]<near_distance), 0, 0] = near_distance

    if kwargs.get('test_time', False):
        render_func = __render_rays_test
    else:
        render_func = __render_rays_train

    results = render_func(model, rays_o, rays_d, hits_t, **kwargs)
    for k, v in results.items():
        if kwargs.get('to_cpu', False):
            v = v.cpu()
            if kwargs.get('to_numpy', False):
                v = v.numpy()
        results[k] = v
    return results


@torch.no_grad()
def __render_rays_test(model, rays_o, rays_d, hits_t, **kwargs):
    """
    Render rays by

    while (a ray hasn't converged)
        1. Move each ray to its next occupied @N_samples (initially 1) samples 
           and evaluate the properties (sigmas, rgbs) there
        2. Composite the result to output; if a ray has transmittance lower
           than a threshold, mark this ray as converged and stop marching it.
           When more rays are dead, we can increase the number of samples
           of each marching (the variable @N_samples)
    """
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    max_samples = kwargs['max_samples']
    results = {}

    # output tensors to be filled in
    N_rays = len(rays_o)
    device = rays_o.device
    opacity = torch.zeros(N_rays, device=device)
    depth = torch.zeros(N_rays, device=device)
    rend_l = 3
    rend_l = rend_l + 3 if model.pred_norm else rend_l
    rend_l = rend_l + kwargs['n_sem_cls'] if model.pred_sem else rend_l
    rend = torch.zeros(N_rays, rend_l, device=device)

    samples = total_samples = 0
    alive_indices = torch.arange(N_rays, device=device)
    # if it's synthetic data, bg is majority so min_samples=1 effectively covers the bg
    # otherwise, 4 is more efficient empirically
    min_samples = 1 if exp_step_factor==0 else 4

    while samples < max_samples:
        N_alive = len(alive_indices)
        if N_alive==0: break

        # the number of samples to add on each ray
        N_samples = max(min(N_rays//N_alive, 64), min_samples)
        samples += N_samples

        xyzs, dirs, deltas, ts, N_eff_samples = \
            vren.raymarching_test(rays_o, rays_d, hits_t[:, 0], alive_indices,
                                  model.density_bitfield, model.cascades,
                                  model.scale, exp_step_factor,
                                  model.grid_size, max_samples, N_samples)
        total_samples += N_eff_samples.sum()
        xyzs = rearrange(xyzs, 'n1 n2 c -> (n1 n2) c')
        dirs = rearrange(dirs, 'n1 n2 c -> (n1 n2) c')
        valid_mask = ~torch.all(dirs==0, dim=1)
        if valid_mask.sum()==0: break

        output = model(xyzs[valid_mask], dirs[valid_mask], **kwargs)
        sigmas = torch.zeros(len(xyzs), device=device)
        _sigmas = output['sigmas']
        sigmas[valid_mask] = _sigmas.float()
        sigmas = rearrange(sigmas, '(n1 n2) -> n1 n2', n2=N_samples)
        rgbs = torch.zeros(len(xyzs), 3, device=device)
        _rgbs = output['rgbs']
        rgbs[valid_mask] = _rgbs.float()
        rgbs = rearrange(rgbs, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        raws = rgbs
        if model.pred_norm:
            norms = torch.zeros(len(xyzs), 3, device=device)
            _norms = output['norms']
            norms[valid_mask] = _norms.float()
            norms = rearrange(norms, '(n1 n2) c -> n1 n2 c', n2=N_samples)
            # if kwargs['pred_norm_nn_norm']:
            #     norms = F.normalize(norms, p=2.0, dim=-1)
            raws = torch.cat((raws, norms), dim=-1)
        if model.pred_sem:
            sems = torch.zeros(len(xyzs), output['sems'].shape[1], device=device)
            _sems = output['sems']
            sems[valid_mask] = _sems.float()
            sems = rearrange(sems, '(n1 n2) c -> n1 n2 c', n2=N_samples)
            raws = torch.cat((raws, sems), dim=-1)

        vren.composite_test_multi_fw(
            sigmas, raws, deltas, ts,
            hits_t[:, 0], alive_indices, kwargs.get('T_threshold', 1e-4),
            N_eff_samples, opacity, depth, rend)
        alive_indices = alive_indices[alive_indices>=0] # remove converged rays

    results['opacity'] = opacity
    results['depth'] = depth
    i = 3
    results['rgb'] = rend[..., :i]
    if model.pred_norm:
        results['norm_nn'] = rend[..., i:i+3]
        if kwargs['pred_norm_nn_norm']:
            results['norm_nn'] = F.normalize(results['norm_nn'], p=2.0, dim=-1)
        i += 3
    if model.pred_sem:
        n_cls = kwargs['n_sem_cls']
        results['sem'] = rend[..., i:i+n_cls]
        i += n_cls
    results['total_samples'] = total_samples # total samples for all rays

    if exp_step_factor==0: # synthetic
        rgb_bg = torch.ones(3, device=device)
    else: # real
        rgb_bg = torch.zeros(3, device=device)
    results['rgb'] += rgb_bg*rearrange(1-opacity, 'n -> n 1')

    return results


#@torch.cuda.amp.autocast()
def __render_rays_train(model, rays_o, rays_d, hits_t, **kwargs):
    """
    Render rays by
    1. March the rays along their directions, querying @density_bitfield
       to skip empty space, and get the effective sample points (where
       there is object)
    2. Infer the NN at these positions and view directions to get properties
       (currently sigmas and rgbs)
    3. Use volume rendering to combine the result (front to back compositing
       and early stop the ray if its transmittance is below a threshold)
    """
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    max_samples = kwargs['max_samples']
    results = {}

    # Optional training ray annealing warmup
    anneal_step = kwargs.get('anneal_steps', 0)
    global_step = kwargs.get('global_step', anneal_step)
    if anneal_step > global_step:
        anneal_strategy = kwargs.get('anneal_strategy', 'none')
        if anneal_strategy == 'avoid_near':
            # Taken from https://arxiv.org/abs/2112.00724
            ps = 0.5
            ray_mid = (hits_t[:, 0, 0] + hits_t[:, 0, 1]) / 2.0
            n_i = min(max(global_step/anneal_step, ps), 1.0)
            hits_t[:, 0, 0] = ray_mid + n_i * (hits_t[:, 0, 0] - ray_mid)
        elif anneal_strategy == 'depth':
            depth = kwargs['depth']
            ps = 0.05
            # Clipping at 1 here is not sufficient,
            # since depth is not the moiddle of the ray
            n_i = min(max(global_step/anneal_step, ps), 100.0)
            hits_t[:, 0, 0] = torch.max(depth + n_i * (hits_t[:, 0, 0] - depth), hits_t[:, 0, 0])
            hits_t[:, 0, 1] = torch.min(depth + n_i * (hits_t[:, 0, 1] - depth), hits_t[:, 0, 1])
        else:
            assert anneal_strategy == 'none'

    rays_a, xyzs, dirs, results['deltas'], results['ts'], results['rm_samples'] = \
        RayMarcher.apply(
            rays_o, rays_d, hits_t[:, 0], model.density_bitfield,
            model.cascades, model.scale,
            exp_step_factor, model.grid_size, max_samples)
    if (rays_a[:,2] == 0.0).any():
        a = 1

    for k, v in kwargs.items(): # supply additional inputs, repeated per ray
        if isinstance(v, torch.Tensor):
            kwargs[k] = torch.repeat_interleave(v[rays_a[:, 0]], rays_a[:, 2], 0)
    output = model(xyzs, dirs, **kwargs)
    sigmas = output['sigmas']
    raws = output['rgbs']
    if model.pred_norm:
        # if kwargs['pred_norm_nn_norm']:
        #     output['norms'] = F.normalize(output['norms'], p=2.0, dim=-1)
        raws = torch.cat((raws, output['norms']), dim=-1)
    if model.pred_sem:
        raws = torch.cat((raws, output['sems']), dim=-1)
    (results['vr_samples'], # volume rendering effective samples
    results['opacity'], results['depth'], rend, results['ws']) = \
        VolumeRenderer.apply(sigmas, raws.contiguous(), results['deltas'], results['ts'],
                             rays_a, kwargs.get('T_threshold', 1e-4))
    i = 3
    results['rgb'] = rend[..., :i]
    if model.pred_norm:
        results['norm_nn'] = rend[..., i:i+3]
        if kwargs['pred_norm_nn_norm']:
            results['norm_nn'] = F.normalize(results['norm_nn'], p=2.0, dim=-1)
        i += 3
    if model.pred_sem:
        n_cls = kwargs['n_sem_cls']
        results['sem'] = rend[..., i:i+n_cls]
        i += n_cls
    
    results['rays_d'] = rays_d
    results['rays_o'] = rays_d

    results['rays_a'] = rays_a
    results['depth_std'] = torch.ones_like(results['depth'], requires_grad=False)

    if exp_step_factor==0: # synthetic
        rgb_bg = torch.ones(3, device=rays_o.device)
    else: # real
        if kwargs.get('random_bg', False):
            rgb_bg = torch.rand(3, device=rays_o.device)
        else:
            rgb_bg = torch.zeros(3, device=rays_o.device)
    results['rgb'] = results['rgb'] + \
                     rgb_bg*rearrange(1-results['opacity'], 'n -> n 1')

    return results