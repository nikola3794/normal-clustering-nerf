'''
Base code taken the following link and modified: 
https://github.com/zju3dv/manhattan_sdf/blob/main/lib/datasets/scannet.py
'''

import os
import torch
import numpy as np
from tqdm import tqdm
from typing import List
import cv2
import einops

import torch.nn.functional as F


WALL_SEMANTIC_ID = 80
FLOOR_SEMANTIC_ID = 160


class ScanNetMannhatanScene():
    def __init__(
        self, 
        scene_root_dir: str, 
        which_split: str,
        which_labels: List,
        downscale_factor: float = 1.0,
    ):
        self.scene_root_dir = scene_root_dir
        self.which_split = which_split
        self.scene_name = os.path.basename(self.scene_root_dir)
        assert os.path.exists(self.scene_root_dir)
        
        # Downscaling not implemented
        assert downscale_factor == 1.0

        image_dir = '{0}/images'.format(self.scene_root_dir)
        image_list = os.listdir(image_dir)
        image_list.sort(key=lambda _:int(_.split('.')[0]))

        # Dataset division
        half_step = 1
        if which_split == 'train':
            self.image_list = image_list[::half_step*2]
        elif which_split == 'test':
            self.image_list = image_list[half_step::half_step*2]
        else:
            raise NotImplementedError
        
        self.n_images = len(self.image_list)
        print(f'Loaded {self.n_images} {which_split} image ids. (half_step={half_step})')

        
        self.W_orig = 640; self.H_orig = 480
        self.W = self.W_orig; self.H = self.H_orig
        # # TODO downscale:
        # self.H = round(self.H_orig * self.downscale_factor)
        # self.W = round(self.W_orig * self.downscale_factor)
        
        # Load intrinsics
        intrinsics = np.loadtxt(f'{self.scene_root_dir}/intrinsic.txt')[:3, :3]
        # TODO downscale: intrinsics need to be scaled accordingly

        # Ray directions in camera coordinates
        X, Y = np.meshgrid(np.arange(self.W), np.arange(self.H))
        X = X.astype(np.float32); Y = Y.astype(np.float32)
        X += 0.5; Y += 0.5
        ray_dirs_uv = np.concatenate((X[:, :, None], Y[:, :, None], np.ones_like(X[:, :, None])), axis=-1)
        ray_dirs_cc = ray_dirs_uv @ np.linalg.inv(intrinsics).T
        ray_dirs_cc = torch.from_numpy(ray_dirs_cc.astype(np.float32))
        ray_dirs_cc = einops.rearrange(ray_dirs_cc, 'h w i -> (h w) i')
        self.depth_type = 'distance'
        if self.depth_type == 'z':
            # Normalize such that |z| = 1
            ray_dirs_cc = (ray_dirs_cc / torch.abs(ray_dirs_cc[:, 2:3]))
        elif self.depth_type == 'distance':
            # Normalize such that ||ray_dir||=1
            ray_dirs_cc = F.normalize(ray_dirs_cc, p=2, dim=-1)
        else:
            raise NotImplementedError
        self.ray_dirs_cc = ray_dirs_cc

        self.img_ids = []
        self.poses = []
        self.rgb_images = []
        self.labels = {}
        for k in which_labels:
            assert k in ['depth', 'semantics', 'semantics_WF']
            self.labels[k] = []
        for imgname in tqdm(self.image_list, desc='Loading dataset'):
            self.img_ids.append(imgname[:-4])
            c2w = np.loadtxt(f'{self.scene_root_dir}/pose/{imgname[:-4]}.txt')
            self.poses.append(c2w.astype(np.float32))

            rgb = cv2.imread(f'{self.scene_root_dir}/images/{imgname[:-4]}.png')
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = (rgb.astype(np.float32) / 255)
            self.rgb_images.append(rgb)

            if 'depth' in which_labels:
                depth_path = f'{self.scene_root_dir}/depth_colmap/{imgname[:-4]}.npy'
                if os.path.exists(depth_path):
                    depth_colmap = np.load(depth_path)
                    depth_colmap[depth_colmap > 2.0] = 0
                else:
                    depth_colmap = np.zeros((self.H, self.W), np.float32)
                if not depth_colmap.shape == (480, 640):
                    depth_colmap = np.zeros((self.H, self.W), np.float32)
                self.labels['depth'].append(depth_colmap.astype(np.float32))
            
            if ('semantics' in which_labels) or ('semantics_WF' in which_labels):
                semantic_deeplab = cv2.imread(f'{self.scene_root_dir}/semantic_deeplab/{imgname[:-4]}.png', -1)
                semantic_deeplab = semantic_deeplab
                wall_mask = semantic_deeplab == WALL_SEMANTIC_ID
                floor_mask = semantic_deeplab == FLOOR_SEMANTIC_ID
                bg_mask = ~(wall_mask | floor_mask)
                # TODO Lik this because 0 is void class...right?
                semantic_deeplab[wall_mask] = 1
                semantic_deeplab[floor_mask] = 2
                semantic_deeplab[bg_mask] = 3
                semantic_deeplab = semantic_deeplab.astype(np.int64)
                if not semantic_deeplab.shape == (480, 640):
                    semantic_deeplab = np.zeros((self.H, self.W), np.int64)
                if 'semantics' in which_labels:
                    self.labels['semantics'].append(np.copy(semantic_deeplab))
                if 'semantics_WF' in which_labels:
                    self.labels['semantics_WF'].append(np.copy(semantic_deeplab))
        
        # TODO
        self.label_metadata = {}
        for k in self.labels:
            if k in ['semantics', 'semantics_WF']:
                self.label_metadata[k] = {'n_valid_classes_scene': 3}
            elif k == 'depth':
                pass
            else:
                raise NotImplementedError
                
        # Merge everything in one numpy array ([N, 4, 4])
        self.poses = np.asarray(self.poses)
        self.poses = torch.from_numpy(self.poses) 
        # Merge everything in one numpy array ([N, H, W, 3])
        self.rgb_images = np.asarray(self.rgb_images)
        self.rgb_images = torch.from_numpy(self.rgb_images) 
        # TODO downscale
        # self.rgb_images = self._downscale_selected_label_all(self.rgb_images, 'rgb)
        for k in self.labels:
            # Merge everything in one numpy array ([N, H, W, ?])
            #dtype = self.labels[k][0].dtype
            self.labels[k] = np.asarray(self.labels[k])#.astype(dtype)
            # TODO _process_loaded_raw_label_all()
            self.labels[k] = torch.from_numpy(self.labels[k]) 
            # TODO downscale
            # self.labels[k] = self._downscale_selected_label_all(self.labels[k], k) 

        self.K = torch.FloatTensor(intrinsics)

        self.scene_boundary = {
            'xyz_scene_min': -1.2*torch.ones(3),
            'xyz_scene_max': 1.2*torch.ones(3),
            'xyz_cam_min': -1.2*torch.ones(3),
            'xyz_cam_max': 1.2*torch.ones(3),
        }
        
    # def _downscale_selected_label_all(self, labels_all, which_label):
    #     if (self.H != self.H_orig) or (self.W != self.W_orig):
    #         return _downscale_selected_label_all(
    #             labels_all=labels_all, 
    #             which_label=which_label, 
    #             H=self.H, 
    #             W=self.W
    #         )
    #     else:
    #         return labels_all
    

if __name__ == '__main__':
    scene = ScanNetMannhatanScene(
        scene_root_dir = 'SCENE_DIR',
        which_split = 'train',
        which_labels=['depth', 'semantics']
    )

    import os, sys
    sys.path.append(os.getcwd())
    from datasets.hypersim_src.draw_scene import draw_scene
    draw_scene(
        poses=scene.poses,
        ray_dirs_cc=scene.ray_dirs_cc,
        scale_cam=5e-2,
        RGBs=scene.rgb_images,
        depths=scene.labels['depth'],
        depth_type=scene.depth_type,
        scene_boundary=scene.scene_boundary,
        every_kth_frame=1,
        connect_cams=False,
        save_path=None, #os.path.join(SAVE_ROOT_DIR, f'{scene_n}.{cam_n}.jpg'),
        draw_now=True
    )