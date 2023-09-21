from torch.utils.data import Dataset
import torch
import einops
import numpy as np


class BaseDataset(Dataset):
    """
    Define length and sampling method
    """
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split

    def _triang_images_metadata(self, h, w, max_expand):
        assert self.ray_sampling_strategy in ['same_image_triang', 'all_images_triang', 'all_images_triang_val']

        self.H, self.W = h, w
        self.N = h * w

        self.max_expand = max_expand
        img_idx = np.arange(0, h*w, 1, dtype=np.int64)
        img_idx = einops.rearrange(img_idx, '(h w) -> h w', h=h)
        # ____|_x2_|____
        # _x3_|_x1_|____
        #     |    |  
        valid_x1_idx = img_idx[1:-1, 1:-1]
        valid_x2_idx = img_idx[:-2, 1:-1]
        valid_x3_idx = img_idx[1:-1, :-2]
        self.valid_idx = {
            'x1': einops.rearrange(valid_x1_idx, 'h w -> (h w)'),
            'x2': einops.rearrange(valid_x2_idx, 'h w -> (h w)'),
            'x3': einops.rearrange(valid_x3_idx, 'h w -> (h w)')}
        
    def _triang_patche_images_metadata(self, h, w, patch_size=8):

        assert self.ray_sampling_strategy in ['same_image_triang_patch', 'all_images_triang_patch']

        self.H, self.W = h, w
        self.N = h * w
        self.patch_size = patch_size

        patch_area = patch_size ** 2
        img_idx = np.arange(0, h*w, 1, dtype=np.int64)
        img_idx = einops.rearrange(img_idx, '(h w) -> h w', h=h)
        # Define all possible patch upper-left corners
        valid_corner_idx = img_idx[:-patch_size+1, :-patch_size+1]
        # Define all tiriangles inside a patch
        # (with an idx relative to the upper left idx of the patch)
        # ____|_x2_|____
        # _x3_|_x1_|____
        #     |    |  
        valid_patch_offset = img_idx[:patch_size, :patch_size]
        patch_idx_local = np.arange(0, patch_area, 1, dtype=np.int64)
        patch_idx_local = einops.rearrange(patch_idx_local, '(h w) -> h w', h=patch_size)
        valid_x1_offset_local = patch_idx_local[1:, 1:]
        valid_x2_offset_local = patch_idx_local[:-1, 1:]
        valid_x3_offset_local = patch_idx_local[1:, :-1]
        # Rearange as a vector
        self.valid_idx = {
            'patch_corners': einops.rearrange(valid_corner_idx, 'h w -> (h w)'),
            'patch_offsets': einops.rearrange(valid_patch_offset, 'h w -> (h w)'),
            'x1_offsets_local': einops.rearrange(valid_x1_offset_local, 'h w -> (h w)'),
            'x2_offsets_local': einops.rearrange(valid_x2_offset_local, 'h w -> (h w)'),
            'x3_offsets_local': einops.rearrange(valid_x3_offset_local, 'h w -> (h w)'),
            }

    def _generate_random_poses(self):
        random_poses, avg_pose = generate_random_poses(self.poses, 
                                                       self.xyz_cam_min,  
                                                       self.xyz_cam_max,  
                                                       10000)
        self.random_poses = random_poses

    def read_intrinsics(self):
        raise NotImplementedError

    def __len__(self):
        if self.split.startswith('train'):
            # TODO Training epoch hardcoded to 1k data points...
            return 1000
        return len(self.poses)

    def __getitem__(self, idx):
        sample = {}

        if self.split.startswith('train'):
            if self.random_tr_poses:
                assert self.ray_sampling_strategy in ['all_images_triang', 
                                                      'same_image_triang',
                                                      'all_images_triang_patch', 
                                                      'same_image_triang_patch',]
            # training pose is retrieved in train.py
            if self.ray_sampling_strategy == 'all_images': # randomly select images
                sample['img_idxs'] = np.random.choice(len(self.poses), self.batch_size)
                sample['pix_idxs'] = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)

            elif self.ray_sampling_strategy == 'same_image': # randomly select ONE image
                sample['img_idxs'] = np.random.choice(len(self.poses), 1)[0]
                sample['pix_idxs'] = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)

            elif self.ray_sampling_strategy in ['all_images_triang', 'same_image_triang']:
                n_triang = self.batch_size // 3

                if self.ray_sampling_strategy == 'all_images_triang':
                    if self.random_tr_poses:
                        n_triang = n_triang // 2
                        rnd_img_idxs = np.random.choice(len(self.random_poses), n_triang)
                        sample['rnd_img_idxs'] = np.repeat(rnd_img_idxs, 3, axis=0)
                    img_idxs = np.random.choice(len(self.poses), n_triang)
                    sample['img_idxs'] = np.repeat(img_idxs, 3, axis=0)
                elif self.ray_sampling_strategy == 'same_image_triang':
                    # Select one image
                    if self.random_tr_poses:
                        n_triang = n_triang // 2
                        rnd_pose_idx = np.random.choice(len(self.random_poses), 1)[0]
                        sample['rnd_img_idxs'] = rnd_pose_idx * np.ones(3*n_triang, dtype=np.int32)
                    img_idx = np.random.choice(len(self.poses), 1)[0]
                    sample['img_idxs'] = img_idx * np.ones(3*n_triang, dtype=np.int32)
                else:
                    raise NotImplementedError
                
                pix_idxs_tr = np.random.choice(self.valid_idx['x1'].shape[0], n_triang)
                pix_idx_x1 = self.valid_idx['x1'][pix_idxs_tr]
                pix_idx_x2 = self.valid_idx['x2'][pix_idxs_tr]
                pix_idx_x3 = self.valid_idx['x3'][pix_idxs_tr]
                # Expand unit triangle
                if self.max_expand > 0:
                    expand = self.max_expand
                    pix_idx_x1_new = (pix_idx_x1 + expand*self.W)
                    pix_idx_x1 = np.where(pix_idx_x1_new < self.N, pix_idx_x1_new, pix_idx_x1)

                    pix_idx_x2_new = (pix_idx_x2 - expand*self.W) 
                    pix_idx_x2 = np.where(pix_idx_x2_new >= 0, pix_idx_x2_new, pix_idx_x2)
                    
                    pix_idx_x3_new = pix_idx_x3 - expand
                    cond_x3 = (pix_idx_x3_new // self.W) == (pix_idx_x3 // self.W)
                    pix_idx_x3 = np.where(cond_x3, pix_idx_x3_new, pix_idx_x3)
                pix_idxs = np.concatenate([pix_idx_x1[:, None], pix_idx_x2[:, None], pix_idx_x3[:, None]], axis=1)
                sample['pix_idxs'] = einops.rearrange(pix_idxs, 'n s -> (n s)')
                
            elif self.ray_sampling_strategy in ['all_images_triang_patch', 'same_image_triang_patch']:
                patch_area = self.patch_size ** 2
                n_patches = self.batch_size // patch_area

                if self.ray_sampling_strategy == 'all_images_triang_patch':
                    if self.random_tr_poses:
                        n_patches = n_patches // 2
                        rnd_img_idxs = np.random.choice(len(self.random_poses), n_patches)
                        sample['rnd_img_idxs'] = np.repeat(rnd_img_idxs, patch_area, axis=0)
                    img_idxs = np.random.choice(len(self.poses), n_patches)
                    sample['img_idxs'] = np.repeat(img_idxs, patch_area, axis=0)
                elif self.ray_sampling_strategy == 'same_image_triang_patch':
                    if self.random_tr_poses:
                        n_patches = n_patches // 2
                        rnd_pose_idx = np.random.choice(len(self.random_poses), 1)[0]
                        sample['rnd_img_idxs'] = rnd_pose_idx * np.ones(patch_area*n_patches, dtype=np.int32)
                    # randomly select ONE image
                    img_idx = np.random.choice(len(self.poses), 1)[0]
                    sample['img_idxs'] = img_idx * np.ones(patch_area*n_patches, dtype=np.int32)
                else:
                    raise NotImplementedError
                
                patch_corner_idx = np.random.choice(self.valid_idx['patch_corners'].shape[0], n_patches)
                patch_offset_idx = self.valid_idx['patch_offsets']
                pix_idxs = patch_corner_idx[:, None] + patch_offset_idx[None:, ]
                sample['pix_idxs'] = einops.rearrange(pix_idxs, 'n s -> (n s)')
                sample['patch_area'] = patch_area
                sample['x1_offsets_local'] = self.valid_idx['x1_offsets_local']
                sample['x2_offsets_local'] = self.valid_idx['x2_offsets_local']
                sample['x3_offsets_local'] = self.valid_idx['x3_offsets_local']
            else:
                raise NotImplementedError

            # randomly select pixels
            rays = self.rays[sample['img_idxs'], sample['pix_idxs']]
            sample['rgb'] = rays[:, :3]
            if self.rays.shape[-1] == 4: # HDR-NeRF data
                sample['exposure'] = rays[:, 3:]
            if hasattr(self, 'labels'):
                for k in self.labels: 
                    sample[k] = self.labels[k][sample['img_idxs'], sample['pix_idxs']]

        else:
            sample['pose'] = self.poses[idx]
            sample['img_idxs'] = idx
            sample['img_id'] = self.img_ids[idx]
            if len(self.rays)>0: # if ground truth available
                rays = self.rays[idx]
                sample['rgb'] = rays[:, :3]
                if rays.shape[1] == 4: # HDR-NeRF data
                    sample['exposure'] = rays[0, 3] # same exposure for all rays
                if hasattr(self, 'labels'):
                    for k in self.labels: 
                        sample[k] = self.labels[k][idx]
            sample['ray_dirs_cc'] = self.directions

        return sample


def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)


def viewmatrix(lookdir, up, position, subtract_position=False):
    """Construct lookat view matrix."""
    vec2 = normalize((lookdir - position) if subtract_position else lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def poses_avg(poses):
    """New pose using average position, z-axis, and up vector of input poses."""
    position = poses[:, :3, 3].mean(0)
    z_axis = poses[:, :3, 2].mean(0)
    up = poses[:, :3, 1].mean(0)
    cam2world = viewmatrix(z_axis, up, position)
    return cam2world


def focus_pt_fn(poses):
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    # Z axis faces away from the cameras # TODO ????????????
    directions = - directions            # TODO ????????????
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def generate_random_poses(poses, xyz_cam_min, xyz_cam_max, n_poses=10000, random_pose_focusptjitter= False):
    dev = poses.device
    poses = poses[:, :3, :].clone().numpy()
    xyz_cam_min = xyz_cam_min.clone().numpy()
    xyz_cam_max = xyz_cam_max.clone().numpy()

    cam2world = poses_avg(poses)
    up = poses[:, :3, 1].mean(0)
    z_axis = focus_pt_fn(poses)
    random_poses = []
    for _ in range(n_poses):
        position = xyz_cam_min + (xyz_cam_max-xyz_cam_min) * (np.random.rand(3)*0.8 + 0.1)

        if random_pose_focusptjitter:
            z_axis_i = z_axis + np.random.randn(*z_axis.shape) * 0.125
        else:
            z_axis_i = z_axis
        # Z axis faces away from the cameras     
        vec2 = normalize(-(z_axis_i - position))
        vec0 = normalize(np.cross(up, vec2))
        vec1 = normalize(np.cross(vec2, vec0))
        #vec0 = cam2world[:3, 0]; vec1 = cam2world[:3, 1]; vec2 = cam2world[:3, 2]; 
        rnd_pose_i = np.stack([vec0, vec1, vec2, position], axis=1)
        #rnd_pose_i = viewmatrix(z_axis_i, up, position, False)
        #rnd_pose_i = np.concatenate([rnd_pose_i, np.array([[0., 0., 0., 1.]])], axis=0)
        random_poses.append(rnd_pose_i)
    random_poses = torch.as_tensor(np.asarray(random_poses)).to(dev).type(torch.float32)
    #cam2world = torch.from_numpy(np.concatenate([cam2world, np.array([[0., 0., 0., 1.]])], axis=0)).to(dev)
    return random_poses, cam2world
