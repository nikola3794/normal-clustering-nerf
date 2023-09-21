import os
import copy

import einops
import torch

from .base import BaseDataset
from .hypersim_src.scene import HypersimScene
from .hypersim_src.utils import generate_pointcloud, clip_depths_to_bbox


class HypersimDataset(BaseDataset):
    def __init__(self, root_dir, hparams, split='train', **kwargs):
        super().__init__(root_dir, split)

        self.ray_sampling_strategy = hparams.ray_sampling_strategy
        self.batch_size = hparams.batch_size
        self.random_tr_poses = hparams.random_tr_poses

        which_labels = ['depth'] if hparams.load_depth_gt else []
        which_labels += ['normals'] if hparams.load_norm_gt else []
        which_labels += ['normals_depth'] if hparams.load_norm_depth_gt else []
        which_labels += ['semantics'] if hparams.load_sem_gt else []
        which_labels += ['semantics_WF'] if hparams.load_sem_WF_gt else []

        # hypersim_root = os.sep.join(root_dir.split(os.sep)[:-1])
        # scene_metadata_path = os.path.join(hypersim_root, 'all_scenes_metadata.json')
        # TODO CHeck is this way of loading kay:
        scene_metadata_path = os.path.join('hypersim_src', 'metadata', 'all_scenes_metadata.json')
            
        scene = HypersimScene(
            scene_root_dir=root_dir,
            scene_metadata_path=scene_metadata_path,
            downscale_factor=hparams.downsample,
            which_labels=which_labels,
            which_cams=['cam_00'],
            which_split=split,
            split_factor=kwargs['split_factor'],
        )
        self.scene_name = scene.scene_name
        self.directions = scene.cam_model.ray_dirs_cc.clone()

        self.img_wh = (scene.W, scene.H)
                
        # Load image ids
        self.img_ids = scene.img_ids

        # Load rgb and poses
        self.ray_idxs = scene.cam_model.ray_idxs.clone()
        self.rays = scene.imgs.clone()
        self.rays = einops.rearrange(self.rays, 'b h w c -> b (h w) c')
        self.poses = scene.cam_model.poses.clone()

        # Load scene bondaries
        scene_bnd = copy.deepcopy(scene.scene_boundary)
        if 'xyz_cam1p5_min' in scene_bnd:
            self.xyz_min = scene_bnd['xyz_cam1p5_min'].clone()
            self.xyz_max = scene_bnd['xyz_cam1p5_max'].clone()
        else:
            self.xyz_min = scene_bnd['xyz_scene_min'].clone()
            self.xyz_max = scene_bnd['xyz_scene_max'].clone()
        
        self.shift = (self.xyz_max+self.xyz_min)/2
        self.scale = (self.xyz_max-self.xyz_min).max().item()/2 * 1.05 # enlarge a little        
            
        # Rescale the pose, because the scene is viewed in [-0.5, 0.5]
        self.poses[:, :3, 3] -= self.shift.unsqueeze(0)
        self.poses[:, :3, 3] /= 2*self.scale 

        self.xyz_cam_min = (scene_bnd['xyz_cam_min'].clone() - self.shift) / (2*self.scale)
        self.xyz_cam_max = (scene_bnd['xyz_cam_max'].clone() - self.shift) / (2*self.scale)

        # Load labels
        self.labels = {}
        for k, v in scene.labels.items():
            shape = 'b h w c -> b (h w) c' if len(v.shape) == 4 else 'b h w -> b (h w)'
            self.labels[k] = einops.rearrange(v.clone(), shape)
            if k in ['semantics', 'semantics_WF']:
                self.n_classes= scene.label_metadata[k]['n_valid_classes_scene']
        
        # Rotation offset
        if kwargs['R_offset'] is not None:
            self.poses[:, :3, :3] = kwargs['R_offset'] @ self.poses[:, :3, :3]
            self.poses[:, :3, 3:] = kwargs['R_offset'] @ self.poses[:, :3, 3:]
            # exact rule for yaw: sqrt(2)*cos((45-x)/180*math.pi)
            # maxinimal for all rotations is sqrt(3)
            adjust_const = 1.6 # approx for 30 degree offset
            self.poses[:, :3, 3] /= adjust_const
            self.scale = self.scale * adjust_const
            if 'normals' in self.labels:
                self.labels['normals'] = (kwargs['R_offset'] @ torch.transpose(self.labels['normals'], 1, 2))
                self.labels['normals'] = torch.transpose(self.labels['normals'], 1, 2)
            if 'normals_depth' in self.labels:
                self.labels['normals_depth'] = (kwargs['R_offset'] @ torch.transpose(self.labels['normals_depth'], 1, 2))
                self.labels['normals_depth'] = torch.transpose(self.labels['normals_depth'], 1, 2)

        # To align with training nerf script convention
        self.poses = self.poses[:, :3, :]

        # TODO: Better emphasize difference between this and classical intrinsic K
        # Load intrinsics
        self.K = (scene.cam_model.M_ndc_from_cam.clone(),
                  scene.cam_model.M_uv_from_ndc.clone(),
                  self.shift,
                  self.scale)
        
        if self.ray_sampling_strategy in ['same_image_triang', 'all_images_triang']:
            self._triang_images_metadata(h=scene.H, w=scene.W,
                                        max_expand=hparams.triang_max_expand)
            
        if self.ray_sampling_strategy in ['same_image_triang_patch', 'all_images_triang_patch']:
            self._triang_patche_images_metadata(h=scene.H, w=scene.W, patch_size=8)
        
            
        if 'depth' in self.labels:
            # Depth needs to be rescaled, if scene bounds have been rescaled
            if (self.xyz_min != scene_bnd['xyz_scene_min']).any() or \
                (self.xyz_max != scene_bnd['xyz_scene_max']).any():
                P_wc = generate_pointcloud(
                    ray_dirs_cc=scene.cam_model.ray_dirs_cc, 
                    poses=scene.cam_model.poses, 
                    depths=self.labels['depth'],
                    depth_type=scene.depth_type)
                self.labels['depth'] = clip_depths_to_bbox(
                    depths=self.labels['depth'], 
                    P_wc=P_wc, 
                    poses=scene.cam_model.poses,
                    xyz_min=self.xyz_min, 
                    xyz_max=self.xyz_max,)

            # Scale depth with respect to scene scaling
            self.labels['depth'] /= 2*self.scale
