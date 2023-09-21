# import os, sys
# sys.path.append(os.getcwd())

import einops
import torch
from torch import nn
import torch.nn.functional as F
import faiss
import faiss.contrib.torch_utils

import vren

from datasets.hypersim_src.utils import _extract_normals_from_ray_batch


class DistortionLoss(torch.autograd.Function):
    """
    Distortion loss proposed in Mip-NeRF 360 (https://arxiv.org/pdf/2111.12077.pdf)
    Implementation is based on DVGO-v2 (https://arxiv.org/pdf/2206.05085.pdf)

    Inputs:
        ws: (N) sample point weights
        deltas: (N) considered as intervals
        ts: (N) considered as midpoints
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]

    Outputs:
        loss: (N_rays)
    """
    @staticmethod
    def forward(ctx, ws, deltas, ts, rays_a):
        loss, ws_inclusive_scan, wts_inclusive_scan = \
            vren.distortion_loss_fw(ws, deltas, ts, rays_a)
        ctx.save_for_backward(ws_inclusive_scan, wts_inclusive_scan, ws, deltas, ts, rays_a)
        return loss

    @staticmethod
    def backward(ctx, dL_dloss):
        ws_inclusive_scan, wts_inclusive_scan, ws, deltas, ts, rays_a = ctx.saved_tensors
        dL_dws = vren.distortion_loss_bw(dL_dloss, ws_inclusive_scan, wts_inclusive_scan,
                                         ws, deltas, ts, rays_a)
        return dL_dws, None, None, None
    
    
def _cluster_indices(sim_all, centrs_i, c_i, old_assignments, t_merge, merge):
    if merge:
        c_i_sim = sim_all[c_i]
        c_indices =  centrs_i[c_i_sim > t_merge]
    else:
        c_indices = torch.tensor([c_i], dtype=torch.int64, device=sim_all.device)
    new_assignments = torch.eq(old_assignments.unsqueeze(1), c_indices.unsqueeze(0)).any(dim=1)
    return c_indices, new_assignments


def _find_opposite(sim_all, centrs_i, c_i, old_assignments, t_opposite, t_merge, find_opposite, merge_clusters):
    if not find_opposite:
        return False, None
    c_o_candidates = sim_all[c_i]
    c_o_argmin = c_o_candidates.argmin().item()
    c_0_min = c_o_candidates[c_o_argmin]
    if (-1.0 * c_0_min) > t_opposite:
        # Merge simialr clusters
        c_o_indices, assign_mask = _cluster_indices(sim_all=sim_all, 
                                    centrs_i=centrs_i, c_i=c_o_argmin,    
                                    old_assignments=old_assignments,
                                    t_merge=t_merge, merge=merge_clusters)
        # Assign new cluster
        return True, assign_mask
    else:
        return False, None


def _normals_clustering(
    normals_np, 
    device,
    K=10,
    niter=10,
    t_similar=0.99, 
    merge_clusters=True, 
    find_opposite=True):

    # Clustering
    # -----------------------------------------------
    kmeans = faiss.Kmeans(3, k=K, niter=niter, gpu=False, spherical=True, verbose=False)
    kmeans.train(normals_np)
    # Extract distance to and cluster membership of each vectotr
    dist_to_centr_np, clust_ass_np = kmeans.index.search(normals_np, 1)
    clust_ass = torch.from_numpy(clust_ass_np).to(device).squeeze(1)
    # Cluster centers
    centrs = torch.from_numpy(kmeans.centroids).to(device)
    centrs_i = torch.arange(start=0, end=centrs.shape[0], step=1, device=centrs.device)

    # Extracting most orthogonal clusters and discarding rest
    # -----------------------------------------------
    clust_ass_new = torch.zeros_like(clust_ass)
    # Calculate the similarity of every clsuter with respect to all other clusters
    # The cluster centroid represents the average normal vector of the cluster.
    # Therefore the dot product measures similarity.
    sim_all = (centrs @ centrs.T)
    sim_all_abs = torch.abs(sim_all)
    # Select C1 as the biggest cluster
    cluster_sizes = torch.bincount(clust_ass)
    cluster_sizes_sort, cluster_sizes_i_sort = \
        torch.topk(cluster_sizes, cluster_sizes.shape[0], largest=True, sorted=True)
    c1_i_old = cluster_sizes_i_sort[0].item()
    # C1: Merge simialr clusters
    c1_indices, c1_assign_mask = _cluster_indices(sim_all=sim_all, 
                                centrs_i=centrs_i, c_i=c1_i_old,    
                                old_assignments=clust_ass, 
                                t_merge=t_similar, merge=merge_clusters)
    # C1: Assign new cluster
    clust_ass_new[c1_assign_mask] = 1

    # Finding the most orthogonal set (C1, C2, C3)
    criteria = (sim_all_abs[:, c1_i_old].unsqueeze(1) +  sim_all_abs[c1_i_old, :].unsqueeze(0) + sim_all_abs)
    mins, min_idxs = torch.min(criteria, dim=0)
    c2_i_old = torch.argmin(mins).item()
    c3_i_old = min_idxs[c2_i_old].item()
    # C2: Merge simialr clusters
    c2_indices, c2_assign_mask = _cluster_indices(sim_all=sim_all, 
                                    centrs_i=centrs_i, c_i=c2_i_old,    
                                    old_assignments=clust_ass, 
                                    t_merge=t_similar, merge=merge_clusters)
    # C2: Assign new cluster
    clust_ass_new[c2_assign_mask] = 2

    # C3: Merge simialr clusters
    c3_indices, c3_assign_mask = _cluster_indices(sim_all=sim_all, 
                                    centrs_i=centrs_i, c_i=c3_i_old,    
                                    old_assignments=clust_ass, 
                                    t_merge=t_similar, merge=merge_clusters)
    # C3: Assign new cluster
    clust_ass_new[c3_assign_mask] = 3
    centrs_new = centrs[[c1_i_old, c2_i_old, c3_i_old]]

    # C1: Find opposite clusters
    assign_opposite, assign_mas = _find_opposite(sim_all=sim_all, 
                                    centrs_i=centrs_i, c_i=c1_i_old, 
                                    old_assignments=clust_ass ,t_opposite=t_similar, 
                                    t_merge=t_similar, find_opposite=find_opposite, 
                                    merge_clusters=merge_clusters)
    if assign_opposite:
        clust_ass_new[assign_mas] = -1

    # C2: Finde opposite clusters
    assign_opposite, assign_mas = _find_opposite(sim_all=sim_all, 
                                    centrs_i=centrs_i, c_i=c2_i_old, 
                                    old_assignments=clust_ass ,t_opposite=t_similar, 
                                    t_merge=t_similar, find_opposite=find_opposite, 
                                    merge_clusters=merge_clusters)
    if assign_opposite:
        clust_ass_new[assign_mas] = -2

    # C3: Finde opposite clusters
    assign_opposite, assign_mas = _find_opposite(sim_all=sim_all, 
                                    centrs_i=centrs_i, c_i=c3_i_old, 
                                    old_assignments=clust_ass ,t_opposite=t_similar, 
                                    t_merge=t_similar, find_opposite=find_opposite, 
                                    merge_clusters=merge_clusters)
    if assign_opposite:
        clust_ass_new[assign_mas] = -3


    return clust_ass_new, clust_ass, centrs_new


class NeRFMTLoss(nn.Module):
    def __init__(self, hparams_dict):
        super().__init__()
        # Regularization
        self.opacity_w = hparams_dict.get('loss_opacity_w', 0)
        self.distortion_w = hparams_dict.get('loss_distortion_w', 0)
        # Depth
        self.depth_w = hparams_dict.get('loss_depth_w', 0)
        # Semantics
        self.sem_w = hparams_dict.get('loss_sem_w', 0)
        self.manhattan_nerf_w = hparams_dict.get('loss_manhattan_nerf_w', 0)
        # Normals from depth
        self.norm_DEpth_L1_w = hparams_dict.get('loss_norm_depth_L1_w', 0)
        self.norm_DEpth_dot_w = hparams_dict.get('loss_norm_depth_dot_w', 0)
        self.norm_CAN_tres = hparams_dict.get('loss_norm_can_tres', 0)
        self.norm_D_C_ort_dot_w = hparams_dict.get('loss_norm_D_C_ort_dot_w', 0)
        self.norm_D_C_centr_dot_w = hparams_dict.get('loss_norm_D_C_centr_dot_w', 0)
        self.norm_D_C_centr_L1_w = hparams_dict.get('loss_norm_D_C_centr_L1_w', 0)
        self.norm_D_C_can_dot_w = hparams_dict.get('loss_norm_D_C_can_dot_w', 0)
        self.norm_D_C_can_L1_w = hparams_dict.get('loss_norm_D_C_can_L1_w', 0)
        self.reg_depth_w = hparams_dict.get('loss_reg_depth_w', 0)
        self.ray_sampling_strategy = hparams_dict.get('ray_sampling_strategy', None)
        self.random_tr_poses = hparams_dict.get('random_tr_poses', False)

        if (self.norm_DEpth_L1_w) > 0 or (self.norm_DEpth_dot_w > 0):
            if hparams_dict.get('loss_norm_GT_depth', False):
                self.norm_GT = 'normals_depth'
                assert hparams_dict['load_norm_depth_gt']
            else:
                self.norm_GT = 'normals'
                assert hparams_dict['load_norm_gt']

        self.pred_norm_nn = hparams_dict['pred_norm_nn']
        self.pred_norm_depth = hparams_dict['pred_norm_depth']

        self.L1_norm = lambda x, y: ((torch.abs(x-y)).sum(-1)).mean()
        #self.L2_norm = lambda x, y: (((x-y)**2).sum(-1)).mean()
        dot_prod = lambda x, y: (x*y).sum(-1)
        # self.dot_prod = lambda x, y: (1.0 - dot_prod(x, y)).mean()
        self.dot_prod = lambda x, y: (1.0 - torch.nn.CosineSimilarity(dim=-1)(x, y)).mean()
        # self.abs_dot_prod = lambda x, y: (1.0 - torch.abs(dot_prod(x, y))).mean()
        self.abs_dot_prod = None

        # Canonical normals
        start = hparams_dict.get('loss_norm_can_start', 0)
        self.can_sched_start = start
        self.can_sched_end = hparams_dict.get('loss_norm_can_end', -1)
        grow = hparams_dict.get('loss_norm_can_grow', 1)
        self.w_sched = lambda w, step: max(0, min(w, (step-start)*(w/grow) ))

        if self.pred_norm_depth:
            assert self.ray_sampling_strategy in ['all_images_triang', 'all_images_triang_val', 'same_image_triang',
                                                  'all_images_triang_patch', 'same_image_triang_patch']

        if self.norm_DEpth_dot_w or self.norm_DEpth_L1_w:
            assert self.pred_norm_depth

        if self.sem_w > 0 or self.manhattan_nerf_w > 0:
            if self.manhattan_nerf_w > 0:
                assert self.pred_norm_depth
                assert not hparams_dict['load_sem_gt']
                assert hparams_dict['load_sem_WF_gt']
                assert self.manhattan_nerf_w == 0.0

            if self.sem_w > 0:
                if self.manhattan_nerf_w == 0:
                    assert hparams_dict['load_sem_gt']
                    assert not hparams_dict['load_sem_WF_gt']


            assert hparams_dict['pred_sem']
            CE = nn.CrossEntropyLoss(ignore_index=-1)
            # Void class is ID==0, label-1 to shift void class to -1 
            self.CE_L = lambda logit, target: CE(logit, target-1)  

    def forward(self, pred_raw, target_raw, **kwargs):

        def _loss_validity_filter(loss, pred, name):
            if loss.nelement() != 1:
                print(f'{name} loss component is EMPTY and is skipped...')
                if pred.nelement() == 0:
                    print(f'({name} pred is empty...)')
                loss = torch.tensor(0.0, device=pred.device)
            elif torch.isnan(loss):
                print(f'{name} loss component is NaN and is skipped...')
                if torch.isnan(pred).any():
                    print(f'({name} pred has nan elements...)')
                loss = torch.tensor(0.0, device=pred.device)
            elif torch.isinf(loss):
                print(f'{name} loss component is INF and is skipped...')
                if torch.isinf(pred).any():
                    print(f'({name} pred has INF elements...)')
                loss = torch.tensor(0.0, device=pred.device) 
            return loss
        
        pred_w_gt = {}; target_gt = {}
        pred_unsup = {}

        gt_l = target_raw['rgb'].shape[0]
        target_gt['rgb'] = target_raw['rgb']
        if 'depth' in target_raw: target_gt['depth'] = target_raw['depth']
        if 'normals' in target_raw: target_gt['normals'] = target_raw['normals']
        if 'normals_depth' in target_raw: target_gt['normals_depth'] = target_raw['normals_depth']
        if 'semantics' in target_raw: target_gt['semantics'] = target_raw['semantics']
        if 'semantics_WF' in target_raw: target_gt['semantics_WF'] = target_raw['semantics_WF']
        assert len(set([target_gt[k].shape[0] for k in target_gt.keys()])) == 1 

        pred_w_gt['rgb'] = pred_raw['rgb'][:gt_l]
        pred_w_gt['depth'] = pred_raw['depth'][:gt_l]
        if 'sem' in pred_raw: pred_w_gt['sem'] = pred_raw['sem'][:gt_l]
        if 'norm_nn' in pred_raw: pred_w_gt['norm_nn'] = pred_raw['norm_nn'][:gt_l]
        #if 'norm_depth' in pred_raw: pred_w_gt['norm_depth'] = pred_raw['norm_depth'][:gt_l]
        pred_w_gt['rays_o'] = pred_raw['rays_o'][:gt_l]
        pred_w_gt['rays_d'] = pred_raw['rays_d'][:gt_l]
        
        # When randomly sampling poses, only apply geometric regularization 
        # on the randomly sampled poses (packed in pred_unsup).
        unsup_start = gt_l if self.random_tr_poses else 0
        # But take all samples for the "classical" ray regularization values
        pred_unsup['deltas'] = pred_raw['deltas']
        pred_unsup['ts'] = pred_raw['ts']
        pred_unsup['ws'] = pred_raw['ts']
        pred_unsup['rays_a'] = pred_raw['rays_a']
        pred_unsup['opacity'] = pred_raw['opacity']
        pred_unsup['depth'] = pred_raw['depth'][unsup_start:]
        if 'norm_nn' in pred_raw: pred_unsup['norm_nn'] = pred_raw['norm_nn'][unsup_start:]
        #if 'norm_depth' in pred_raw: pred_raw['norm_depth'] = pred_raw['norm_depth'][unsup_start:]
        pred_unsup['rays_o'] = pred_raw['rays_o'][unsup_start:]
        pred_unsup['rays_d'] = pred_raw['rays_d'][unsup_start:]

        def get_triang_idx(seq_len, dev):
            pix_idx = torch.arange(0, seq_len, device=dev)
            assert pix_idx.shape[0] % 3 == 0
            pix_idx = einops.rearrange(pix_idx, '(n s) -> n s', s=3)
            return {'x1': pix_idx[:, 0],
                    'x2': pix_idx[:, 1],
                    'x3': pix_idx[:, 2],}
        
        def get_patch_triang_idx(seq_len, patch_s, offset_local, dev):
            pix_idx = torch.arange(0, seq_len, device=dev)
            assert pix_idx.shape[0] % patch_s == 0
            pix_idx = einops.rearrange(pix_idx, '(n s) -> n s', s=patch_s)
            return {'x1': einops.rearrange(pix_idx[:, offset_local['x1']], 'n s -> (n s)'),
                    'x2': einops.rearrange(pix_idx[:, offset_local['x2']], 'n s -> (n s)'),
                    'x3': einops.rearrange(pix_idx[:, offset_local['x3']], 'n s -> (n s)'),}
  
        dev = pred_raw['rgb'].device
        n_unsup = pred_unsup['depth'].shape[0]
        n_w_gt = pred_w_gt['rgb'].shape[0]
        # Indices which help extract x1, x2 and x3 of each triangle
        if self.ray_sampling_strategy in ['all_images_triang', 'same_image_triang']: #'all_images_triang_val'
            pred_w_gt['x123_idx'] = get_triang_idx(seq_len=n_w_gt, dev=dev)
            pred_unsup['x123_idx'] = get_triang_idx(seq_len=n_unsup, dev=dev)
        # Indices which help extract x1, x2 and x3 of each triangle
        # (for patches the same point will belong to multuple tirangles)
        elif self.ray_sampling_strategy in ['all_images_triang_patch', 'same_image_triang_patch']:
            offset_local = {'x1':target_raw['x1_offsets_local'],
                            'x2':target_raw['x2_offsets_local'],
                            'x3':target_raw['x3_offsets_local'],}
            patch_s = target_raw['patch_area']
            pred_w_gt['x123_idx'] = get_patch_triang_idx(seq_len=n_w_gt, 
                                    patch_s=patch_s, offset_local=offset_local, dev=dev)
            pred_unsup['x123_idx'] = get_patch_triang_idx(seq_len=n_unsup,
                                    patch_s=patch_s, offset_local=offset_local, dev=dev)
        # Extract normals from rendered depth
        if self.pred_norm_depth:
            pred_w_gt['norm_depth']  = _extract_normals_from_ray_batch(
                                        rays_o=pred_w_gt['rays_o'], 
                                        rays_d=pred_w_gt['rays_d'], 
                                        depth=pred_w_gt['depth'],
                                        x123_idx=pred_w_gt['x123_idx'])
            pred_unsup['norm_depth'] = _extract_normals_from_ray_batch(
                                        rays_o=pred_unsup['rays_o'], 
                                        rays_d=pred_unsup['rays_d'], 
                                        depth=pred_unsup['depth'],
                                        x123_idx=pred_unsup['x123_idx'])

        loss_d = {}

        # RGB photometric loss
        rgb_pred = pred_w_gt['rgb']
        rgb_target = target_gt['rgb']
        if self.ray_sampling_strategy == 'all_images_triang_val' and rgb_target == None:
            loss_d['rgb'] = 0.0
        else:
            rgb_loss = ((rgb_pred-rgb_target)**2).mean()
            loss_d['rgb'] = _loss_validity_filter(rgb_loss, rgb_pred, 'rgb')
        
        # Opacity regularization loss
        if self.opacity_w > 0:
            o = pred_unsup['opacity']+1e-10
            # encourage opacity to be either 0 or 1 to avoid floater
            opacity_loss = self.opacity_w*(-o*torch.log(o)).mean()
            loss_d['opacity'] = _loss_validity_filter(opacity_loss, o, 'opacity')

        # Distortion regularization loss
        if self.distortion_w > 0:
            distortion_loss = self.distortion_w * \
                DistortionLoss.apply(pred_unsup['ws'], pred_unsup['deltas'],
                                     pred_unsup['ts'], pred_unsup['rays_a']).mean()
            loss_d['distortion'] = _loss_validity_filter(distortion_loss, pred_unsup['ws'], 'distortion')

        # Depth loss
        if self.depth_w > 0:
            d_pred = pred_w_gt['depth']
            d_target = target_gt['depth']
            # Discard elements where label is 0
            valid_idx = (d_target > 0).detach()
            if (valid_idx.numel() > 0) and (True in valid_idx):
                d_pred = d_pred[valid_idx]
                d_target = d_target[valid_idx]
                # Regular L2 loss
                d_loss = self.depth_w * ((d_pred-d_target)**2).mean()
                loss_d['depth'] = _loss_validity_filter(d_loss, d_pred, 'depth')                
            else:
                print('Depth target has no valid_idx in current batch...')
                loss_d['depth'] = torch.tensor(0.0, device=d_pred.device)

        # Normals Depth loss
        if (self.norm_DEpth_L1_w > 0) or (self.norm_DEpth_dot_w > 0):
            norm_depth = pred_w_gt['norm_depth']
            # Only select normal GT corresponding to x1 of the triangle
            nom_tar = target_gt[self.norm_GT][pred_w_gt['x123_idx']['x1']]
            # valid_idx = (target['depth'][:norm_target.shape[0]] > 0).detach()
            valid_idx = (nom_tar.abs().sum(-1) > 0).detach()
            if (valid_idx.numel() > 0) and (True in valid_idx):
                norm_depth = norm_depth[valid_idx]
                nom_tar = nom_tar[valid_idx]
                # Regular L1 loss (squared vetor norm)
                if self.norm_DEpth_L1_w > 0:
                    norm_depth_L1 = self.L1_norm(norm_depth, nom_tar)
                    norm_depth_L1 *= self.norm_DEpth_L1_w 
                    loss_d['norm_D_L1'] = _loss_validity_filter(norm_depth_L1, norm_depth, 'norm_depth_L1')
                if self.norm_DEpth_dot_w > 0:
                    norm_depth_dot = self.dot_prod(norm_depth, nom_tar)
                    norm_depth_dot *= self.norm_DEpth_dot_w 
                    loss_d['norm_D_dot'] = _loss_validity_filter(norm_depth_dot, norm_depth, 'norm_depth_dot')
            else:
                print('Norm target has no valid_idx in current batch...')
                loss_d['norm'] = torch.tensor(0.0, device=norm_depth.device)

        # RegNerf loss
        if self.reg_depth_w > 0 and kwargs['global_step'] > self.can_sched_start:
            d_pred = pred_unsup['depth']
            x123_idx = pred_unsup['x123_idx']
            reg_depth_loss = (d_pred[x123_idx['x1']] - d_pred[x123_idx['x2']]) ** 2 
            reg_depth_loss += (d_pred[x123_idx['x1']] - d_pred[x123_idx['x3']]) ** 2 
            reg_depth_loss = reg_depth_loss.mean()
            loss_d['reg_depth'] = _loss_validity_filter(reg_depth_loss, d_pred, 'reg_depth')

        # Normals clustering loss
        if (self.norm_D_C_ort_dot_w > 0) or\
            (self.norm_D_C_centr_dot_w > 0) or (self.norm_D_C_centr_L1_w > 0) or\
                (self.norm_D_C_can_dot_w > 0) or (self.norm_D_C_can_L1_w > 0):        
            step = kwargs['global_step']
            if (step <= self.can_sched_end) or (self.can_sched_end == -1):    
                norm_D_C = pred_unsup['norm_depth']        
                # Remove invalid predicted normals (mostly (0,0,0) invalid) 
                invalid_idx = torch.abs(norm_D_C).sum(-1) == 0.0
                invalid_idx += torch.isnan(norm_D_C).sum(-1) > 0
                invalid_idx += torch.isinf(norm_D_C).sum(-1) > 0
                norm_D_C = norm_D_C[torch.logical_not(invalid_idx)]

                find_opposite=True           
                clust_ass_new, clust_ass_orig, centrs_new = _normals_clustering(
                                                                norm_D_C.detach().contiguous().cpu().numpy(), 
                                                                device=norm_D_C.device,
                                                                K=20,
                                                                niter=20,
                                                                t_similar=1.0-self.norm_CAN_tres, 
                                                                merge_clusters=True, 
                                                                find_opposite=find_opposite)
                norm_D_C_ort = norm_D_C[clust_ass_new != 0].contiguous()
                clust_ass_new = clust_ass_new[clust_ass_new != 0]

                # Flip opposite, if any, to just have 3 clusters
                if find_opposite:
                    norm_D_C_ort[clust_ass_new < 0] *= -1.0
                    clust_ass_new[clust_ass_new < 0] *= -1

                # Keep only members close enough to cluster centroid
                discard_1 = (1.0 - (norm_D_C_ort[clust_ass_new == 1]*centrs_new[0].unsqueeze(0)).sum(-1)) > self.norm_CAN_tres
                discard_2 = (1.0 - (norm_D_C_ort[clust_ass_new == 2]*centrs_new[1].unsqueeze(0)).sum(-1)) > self.norm_CAN_tres
                discard_3 = (1.0 - (norm_D_C_ort[clust_ass_new == 3]*centrs_new[2].unsqueeze(0)).sum(-1)) > self.norm_CAN_tres
                clust_ass_new[clust_ass_new == 1][discard_1] = 0
                clust_ass_new[clust_ass_new == 2][discard_2] = 0
                clust_ass_new[clust_ass_new == 3][discard_3] = 0

                norm_D_C_ort = norm_D_C_ort[clust_ass_new != 0].contiguous()
                clust_ass_new = clust_ass_new[clust_ass_new != 0]

                # Calculate losses
                clust_1 = norm_D_C_ort[clust_ass_new == 1]
                clust_2 = norm_D_C_ort[clust_ass_new == 2]
                clust_3 = norm_D_C_ort[clust_ass_new == 3]
                c_1 = F.normalize(clust_1.mean(dim=0, keepdim=True), p=2.0, dim=-1)
                c_2 = F.normalize(clust_2.mean(dim=0, keepdim=True), p=2.0, dim=-1)
                c_3 = F.normalize(clust_3.mean(dim=0, keepdim=True), p=2.0, dim=-1)
                loss_ort_dot  = torch.abs((c_1*c_2).sum())
                loss_ort_dot += torch.abs((c_1*c_3).sum())
                loss_ort_dot += torch.abs((c_2*c_3).sum())
                loss_ort_dot /= 3.0
                loss_clust_dot  = 1.0 - (clust_1 * c_1).sum(dim=-1).mean() 
                loss_clust_dot += 1.0 - (clust_2 * c_2).sum(dim=-1).mean() 
                loss_clust_dot += 1.0 - (clust_3 * c_3).sum(dim=-1).mean() 
                loss_clust_dot /= 3.0
                loss_clust_l1  = torch.abs(clust_1 - c_1).sum(dim=-1).mean()
                loss_clust_l1 += torch.abs(clust_2 - c_2).sum(dim=-1).mean()
                loss_clust_l1 += torch.abs(clust_3 - c_3).sum(dim=-1).mean()
                loss_clust_l1 /= 3.0

                can_3 = torch.tensor([
                    [ 1.0,  0.0,  0.0],
                    [-1.0,  0.0,  0.0],
                    [ 0.0,  1.0,  0.0],
                    [ 0.0, -1.0,  0.0],
                    [ 0.0,  0.0,  1.0],
                    [ 0.0,  0.0, -1.0]], device=norm_D_C_ort.device)

                c_mat = torch.vstack([c_1, c_2, c_3])
                can_clust_sim = 1.0 - (c_mat.unsqueeze(1) * can_3.unsqueeze(0)).sum(-1)
                cond_clust = can_clust_sim <  self.norm_CAN_tres*3.0
                loss_can_dot = 0.0
                loss_can_l1 = 0.0
                if cond_clust.any():
                    indices = cond_clust.nonzero(as_tuple=True)
                    c_mat_sel = c_mat[indices[0]].contiguous()
                    can_3_sel = can_3[indices[1]].contiguous()
                    loss_can_dot = 1.0 - (c_mat_sel*can_3_sel).sum(dim=-1).mean()
                    loss_can_l1 = torch.abs(c_mat_sel-can_3_sel).sum(dim=-1).mean()
                    loss = self.w_sched(self.norm_D_C_can_dot_w, step)*loss_can_dot
                    loss_d['norm_D_C_can_dot'] = _loss_validity_filter(loss, norm_D_C_ort, 'norm_D_C_can_dot')
                    loss = self.w_sched(self.norm_D_C_can_L1_w, step)*loss_can_l1
                    loss_d['norm_D_C_can_L1'] = _loss_validity_filter(loss, norm_D_C_ort, 'norm_D_C_can_L1')
                
                loss = self.w_sched(self.norm_D_C_ort_dot_w, step)*loss_ort_dot
                loss_d['norm_D_C_ort_dot'] = _loss_validity_filter(loss, norm_D_C_ort, 'norm_D_C_ort_dot')
                loss = self.w_sched(self.norm_D_C_centr_dot_w, step)*loss_clust_dot
                loss_d['norm_D_C_centr_dot'] = _loss_validity_filter(loss, norm_D_C_ort, 'norm_D_C_centr_dot')
                loss = self.w_sched(self.norm_D_C_centr_L1_w, step)*loss_clust_l1
                loss_d['norm_D_C_centr_L1'] = _loss_validity_filter(loss, norm_D_C_ort, 'norm_D_C_centr_L1')

        if self.manhattan_nerf_w > 0:
            assert pred_w_gt['sem'].shape[1] == 3
            sem_WF_pred = pred_w_gt['sem'][pred_w_gt['x123_idx']['x1']]
            sem_wf_pred_softmax = torch.softmax(sem_WF_pred, dim=-1)
            sem_WF_target = target_gt['semantics_WF'][pred_w_gt['x123_idx']['x1']]

            weight_wf = torch.tensor([1.0, 1.0, 0.3])
            CE_WF = nn.CrossEntropyLoss(ignore_index=-1, weight=weight_wf.to(sem_WF_pred), label_smoothing=0.1)
            sem_WF_loss = self.sem_w * CE_WF(sem_WF_pred, sem_WF_target-1)
            loss_d['sem_WF'] = _loss_validity_filter(sem_WF_loss, sem_WF_pred, 'sem_WF')

            # They are selecting normal positions from GT
            wall_mask = sem_WF_target == 1
            floor_mask = sem_WF_target == 2

            norm_D = pred_w_gt['norm_depth'] 
            
            if kwargs['global_step'] > self.can_sched_start:
                joint_loss = 0.0
                if floor_mask.any() > 0:
                    floor_normals = norm_D[floor_mask]
                    floor_loss = (1.0 - floor_normals[:, 2]) # Eq.8
                    joint_floor_loss = (sem_wf_pred_softmax[floor_mask][:, 1] * floor_loss).mean() # Eq.13
                    joint_loss += joint_floor_loss
                
                if wall_mask.any() > 0:
                    wall_normals = norm_D[wall_mask]
                    wall_loss_vertical = wall_normals[:, 2].abs()
                    theta = kwargs['theta_WF']
                    #theta = torch.tensor(0.0).to(norm_D.device)
                    cos = wall_normals[:, 0] * torch.cos(theta) + wall_normals[:, 1] * torch.sin(theta)
                    wall_loss_horizontal = torch.min(cos.abs(), torch.min((1 - cos).abs(), (1 + cos).abs())) # Eq.9
                    wall_loss = wall_loss_vertical + wall_loss_horizontal
                    joint_wall_loss = (sem_wf_pred_softmax[wall_mask][:, 0] * wall_loss).mean() # Eq.13
                    joint_loss += joint_wall_loss
                
                if floor_mask.any() > 0 or wall_mask.any() > 0:
                    joint_loss *= self.manhattan_nerf_w
                    loss_d['norm_WF'] = _loss_validity_filter(joint_loss, norm_D, 'norm_WF')
            else:
                # Semantic score is unreliable in early training stage
                geo_loss = 0.

                if floor_mask.sum() > 0:
                    floor_normals = norm_D[floor_mask]
                    floor_loss = (1 - floor_normals[..., 2]).mean()
                    geo_loss += floor_loss
                
                if wall_mask.sum() > 0:
                    wall_normals = norm_D[wall_mask]
                    wall_loss_vertical = wall_normals[..., 2].abs().mean()
                    geo_loss += wall_loss_vertical

                if floor_mask.sum() > 0 or wall_mask.sum() > 0:
                    geo_loss *= self.manhattan_nerf_w
                    loss_d['norm_WF'] = _loss_validity_filter(geo_loss, norm_D, 'norm_WF')
 
        # Segmentation loss
        if self.sem_w and self.manhattan_nerf_w == 0:
            sem_pred = pred_w_gt['sem']
            sem_target = target_gt['semantics']
            sem_loss = self.sem_w * self.CE_L(sem_pred, sem_target)
            loss_d['sem'] = _loss_validity_filter(sem_loss, sem_pred, 'sem')
            

        # print(f'step= {kwargs["global_step"]}')
        # for k in loss_d:
        #     print(f'{k}: {loss_d[k]}')
        # print('\n')

        loss_d['total'] = sum(lo for lo in loss_d.values())

        # Quick fix to have zero gradient in case of all losses are invalid
        if loss_d['total'] == 0.0:
            loss_d['total'] = pred_w_gt['rgb'][0, 0] - pred_w_gt['rgb'][0, 0]

        return loss_d
