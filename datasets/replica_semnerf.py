import copy

import einops

from .base import BaseDataset
from .replica_semnerf_src.scene import ReplicaSemNerfScene


class ReplicaSemNerfDataset(BaseDataset):
    def __init__(self, root_dir, hparams, split='train', **kwargs):
        super().__init__(root_dir, split)

        self.ray_sampling_strategy = hparams.ray_sampling_strategy
        self.batch_size = hparams.batch_size
        self.random_tr_poses = hparams.random_tr_poses

        which_labels = ['depth'] if hparams.load_depth_gt else []
        which_labels += ['normals_depth'] if hparams.load_norm_depth_gt else []
        which_labels += ['semantics'] if hparams.load_sem_gt else []
        which_labels += ['semantics_WF'] if hparams.load_sem_WF_gt else []

        scene = ReplicaSemNerfScene(
            scene_root_dir=root_dir,
            which_split=split,
            which_labels=which_labels,
            downscale_factor=hparams.downsample,
        )

        self.scene_name = scene.scene_name
        self.directions = scene.ray_dirs_cc.clone()
        #self.directions = F.normalize(self.directions, dim=-1)

        self.img_wh = (scene.W, scene.H)

        # Load scene bondaries
        scene_bnd = copy.deepcopy(scene.scene_boundary)
        self.xyz_min = scene_bnd['xyz_scene_min'].clone()
        self.xyz_max = scene_bnd['xyz_scene_max'].clone()
        self.shift = (self.xyz_max+self.xyz_min)/2
        self.scale = (self.xyz_max-self.xyz_min).max().item()/2 * 1.05 # enlarge a little
                
        # Load intrinsics
        self.K = scene.K

        # Load image ids
        self.img_ids = scene.img_ids

        # Load rgb and poses
        #self.ray_idxs = scene.cam_model.ray_idxs.clone()
        self.rays = scene.rgb_images.clone()
        self.rays = einops.rearrange(self.rays, 'b h w c -> b (h w) c')
        self.poses = scene.poses.clone()

        # Rescale the pose, because the scene is viewed in [-0.5, 0.5]
        self.poses[:, :3, 3] -= self.shift.unsqueeze(0)
        self.poses[:, :3, 3] /= 2*self.scale 
        # To align with training nerf script convention
        self.poses = self.poses[:, :3, :]

        self.xyz_cam_min = (scene_bnd['xyz_cam_min'].clone() - self.shift) / (2*self.scale)
        self.xyz_cam_max = (scene_bnd['xyz_cam_max'].clone() - self.shift) / (2*self.scale)

        # Load labels
        self.labels = {}
        for k, v in scene.labels.items():
            shape = 'b h w c -> b (h w) c' if len(v.shape) == 4 else 'b h w -> b (h w)'
            self.labels[k] = einops.rearrange(v.clone(), shape)
            if k == 'semantics':
                assert 'semantics_WF' not in self.labels
                self.n_classes= scene.label_metadata[k]['n_valid_classes_scene']
            if k == 'semantics_WF':
                assert 'semantics' not in self.labels
                self.n_classes= 3
        
        if self.ray_sampling_strategy in ['same_image_triang', 'all_images_triang']:
            self._triang_images_metadata(h=scene.H, w=scene.W,
                                        max_expand=hparams.triang_max_expand)
            
        if self.ray_sampling_strategy in ['same_image_triang_patch', 'all_images_triang_patch']:
            self._triang_patche_images_metadata(h=scene.H, w=scene.W, patch_size=8)
        
        if 'depth' in self.labels:
            # Scale depth with respect to scene scaling
            self.labels['depth'] /= 2*self.scale
