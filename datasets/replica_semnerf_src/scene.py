'''
Base code taken the following link and modified: 
https://github.com/Harry-Zhi/semantic_nerf
'''

import os, sys
import glob
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import math
from imgviz import label_colormap
import einops
import json

import os, sys
sys.path.append(os.getcwd())
from datasets.hypersim_src.utils import _extract_normals_from_depth_batch


class ReplicaSemNerfScene(Dataset):
    def __init__(
        self, 
        scene_root_dir: str, 
        which_labels: List[str], 
        which_split: str,
        #split_factor: float = 0.5,
        downscale_factor: float = 1.0,
    ):  
        self.which_labels = which_labels
        self.which_split = which_split
        self.downscale_factor = downscale_factor

        # Hardcoded: only use Sequence_1
        self.scene_name = os.path.basename(scene_root_dir)
        self.semantic_root_dir = os.path.join(os.path.dirname(scene_root_dir), 'semantic_info', self.scene_name)
        self.scene_root_dir = os.path.join(scene_root_dir, "Sequence_1")

        assert os.path.exists(self.scene_root_dir)

        traj_file = os.path.join(self.scene_root_dir, "traj_w_c.txt")
        self.rgb_dir = os.path.join(self.scene_root_dir, "rgb")
        self.depth_dir = os.path.join(self.scene_root_dir, "depth")  # depth is in mm uint
        self.semantic_class_dir = os.path.join(self.scene_root_dir, "semantic_class")
        # self.semantic_instance_dir = os.path.join(self.scene_root_dir, "semantic_instance")
        # if not os.path.exists(self.semantic_instance_dir):
        #     self.semantic_instance_dir = None

        self.depth_type = 'z'
        self.H_orig = 480; self.W_orig = 640;
        self.H = int(self.H_orig * downscale_factor); self.W = int(self.W_orig * downscale_factor);

        total_num = 900
        self.img_ids = list(range(0, total_num))
        self.poses = np.loadtxt(traj_file, delimiter=" ").reshape(-1, 4, 4)
        self.poses = self.poses.astype(np.float32)

        self.rgb_list = sorted(glob.glob(self.rgb_dir + '/rgb*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
        self.depth_list = sorted(glob.glob(self.depth_dir + '/depth*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
        self.semantic_list = sorted(glob.glob(self.semantic_class_dir + '/semantic_class_*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))
        # if self.semantic_instance_dir is not None:
        #     self.instance_list = sorted(glob.glob(self.semantic_instance_dir + '/semantic_instance_*.png'), key=lambda file_name: int(file_name.split("_")[-1][:-4]))

        # Intrinsic matrix
        self.hfov = 90
        # the pin-hole camera has the same value for fx and fy
        self.fx = self.W / 2.0 / math.tan(math.radians(self.hfov / 2.0))
        # self.fy = self.H / 2.0 / math.tan(math.radians(self.yhov / 2.0))
        self.fy = self.fx
        self.cx = (self.W - 1.0) / 2.0
        self.cy = (self.H - 1.0) / 2.0
        self.K = torch.tensor([[self.fx,  0.0,      self.cx],
                               [0.0,      self.fy,  self.cy],
                               [0.0,      0.0,      1.0]])

        # Ray directions in CC
        # pytorch's meshgrid has indexing='ij', we transpose to "xy" moode
        i, j = torch.meshgrid(torch.arange(self.W), torch.arange(self.H))
        i = i.t().float()
        j = j.t().float()
        size = [self.H, self.W]
        # i_batch = torch.empty(size)
        # j_batch = torch.empty(size)
        # i_batch[:, :, :] = i[None, :, :]
        # j_batch[:, :, :] = j[None, :, :]
        # "opencv" convention:
        x = (i - self.cx) / self.fx
        y = (j - self.cy) / self.fy
        z = torch.ones(size)
        self.ray_dirs_cc = torch.stack((x, y, z), dim=-1)  # shape of [B, H, W, 3]
        self.ray_dirs_cc = einops.rearrange(self.ray_dirs_cc, 'h w i -> (h w) i')

       # training samples       
        self.rgb_images = []
        self.labels = {}
        for k in which_labels:
            assert k in ['depth', 'semantics', 'semantics_WF', 'normals_depth']
            self.labels[k] = []
        for idx in self.img_ids:
            image = cv2.imread(self.rgb_list[idx])[:,:,::-1] / 255.0  # change from BGR uinit 8 to RGB float
            image = image.astype(np.float32)
            assert self.H_orig == image.shape[0]
            assert self.W_orig == image.shape[1]
            if (self.H_orig != self.H) or (self.W_orig != self.W):
                image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            self.rgb_images.append(image)

            if 'depth' in which_labels:
                depth = cv2.imread(self.depth_list[idx], cv2.IMREAD_UNCHANGED) / 1000.0  # uint16 mm depth, then turn depth from mm to meter
                depth = depth.astype(np.float32)
                if (self.H_orig != self.H) or (self.W_orig != self.W):
                    depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
                self.labels['depth'].append(depth)

            if ('semantics' in which_labels):
                semantic = cv2.imread(self.semantic_list[idx], cv2.IMREAD_UNCHANGED)
                semantic = semantic.astype(np.int64)
                # if self.semantic_instance_dir is not None:
                #     instance = cv2.imread(self.instance_list[idx], cv2.IMREAD_UNCHANGED) # uint16
                if (self.H_orig != self.H) or (self.W_orig != self.W):
                    semantic = cv2.resize(semantic, (self.W, self.H), interpolation=cv2.INTER_NEAREST).astype(np.int64)
                    # if self.semantic_instance_dir is not None:
                    #     instance = cv2.resize(instance, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
                self.labels['semantics'].append(semantic)

            if ('semantics_WF' in which_labels):
                semantic = cv2.imread(self.semantic_list[idx], cv2.IMREAD_UNCHANGED)
                semantic = semantic.astype(np.int64)
                # if self.semantic_instance_dir is not None:
                #     instance = cv2.imread(self.instance_list[idx], cv2.IMREAD_UNCHANGED) # uint16
                if (self.H_orig != self.H) or (self.W_orig != self.W):
                    semantic = cv2.resize(semantic, (self.W, self.H), interpolation=cv2.INTER_NEAREST).astype(np.int64)
                    # if self.semantic_instance_dir is not None:
                    #     instance = cv2.resize(instance, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)

                semantic_WF = np.zeros_like(semantic, dtype=np.int64)
                # Wall class 93 (window 97; )
                semantic_WF[semantic == 93] = 1
                # Floor class 40 (mat 50; ceeling 31)
                semantic_WF[semantic == 40] = 2

                FW_mask = (semantic_WF == 1) + (semantic_WF == 2)
                semantic_WF[np.logical_not(FW_mask)] = 3
                self.labels['semantics_WF'].append(semantic_WF)
                #metadata_dict['n_valid_classes_scene'] = 3


             # number of semantic classes, including the void class of 0
        
        depth_all = torch.from_numpy(np.asarray(self.labels['depth'])).clone()
        poses_all = torch.from_numpy(self.poses).clone()

        half_step = 6
        if which_split == 'train':
            self.img_ids = self.img_ids[::2*half_step]
            self.poses = self.poses[::2*half_step]
            self.rgb_images = self.rgb_images[::2*half_step]
            for k in self.labels:
                self.labels[k] = self.labels[k][::2*half_step]
        elif which_split == 'test':
            self.img_ids = self.img_ids[half_step::2*half_step]
            self.poses = self.poses[half_step::2*half_step]
            self.rgb_images = self.rgb_images[half_step::2*half_step]
            for k in self.labels:
                self.labels[k] = self.labels[k][half_step::2*half_step]
        else:
            raise NotImplementedError
        self.n_images = len(self.img_ids)
        print(f'Loaded {self.n_images} {which_split} image ids. (half_step={half_step})')

        self.label_metadata = {}
        for k in self.labels:
            if k in ['semantics', 'semantics_WF']:
                semantic_classes = np.unique(
                    np.concatenate(
                        (np.unique(self.labels[k]), 
                    np.unique(self.labels[k])))).astype(np.uint8)
                num_classes = semantic_classes.shape[0] 
                num_valid_classes = num_classes - 1 # exclude void class

                colour_map_np = label_colormap()[semantic_classes]
                json_class_mapping = os.path.join(self.semantic_root_dir, "info_semantic.json")
                with open(json_class_mapping, "r") as f:
                    annotations = json.load(f)
                ignore_label = -1
                class_name_string = ["void"] + [x["name"] for x in annotations["classes"]]
                classes_remap = np.arange(num_classes)

                # Remap classes
                for i, _ in enumerate(self.labels[k]):
                    for new_id, old_id in enumerate(semantic_classes):
                        self.labels[k][i][self.labels[k][i] == old_id] = new_id

                self.label_metadata[k] = {'class_ids_scene': semantic_classes,
                                          'n_classes_scene': num_classes,
                                          'n_valid_classes_scene': num_valid_classes,
                                          'id_to_colour_scene': colour_map_np}
            elif k == 'depth':
                pass
            elif k == 'normals_depth':
                pass
            else:
                raise NotImplementedError

        # Merge everything in one numpy array ([N, 4, 4])
        #self.poses = np.asarray(self.poses)
        self.poses = torch.from_numpy(self.poses) 
        # Merge everything in one numpy array ([N, H, W, 3])
        self.rgb_images = np.asarray(self.rgb_images)
        self.rgb_images = torch.from_numpy(self.rgb_images) 
        # self.rgb_images = self._downscale_selected_label_all(self.rgb_images, 'rgb)
        for k in self.labels:
            # Merge everything in one numpy array ([N, H, W, ?])
            #dtype = self.labels[k][0].dtype
            self.labels[k] = np.asarray(self.labels[k])#.astype(dtype)
            # TODO _process_loaded_raw_label_all()
            self.labels[k] = torch.from_numpy(self.labels[k]) 

        if 'normals_depth' in which_labels:
            assert 'depth' in self.labels
            print('Extracting normals from depth...')
            normals_gt_depth = _extract_normals_from_depth_batch(
                depth=self.labels['depth'],
                ray_dirs_cc=self.ray_dirs_cc,
                poses=self.poses[:, :3, :3]
            )
            self.labels['normals_depth'] = normals_gt_depth
        
        print('Extracting scene boudnary...')
        assert 'depth' in which_labels
        # Otherwise compute the scene boundary
        # Compute from all frames in the whole scene
        depths = einops.rearrange(depth_all.clone(), 'b h w -> b (h w) 1')
        P_cc = self.ray_dirs_cc * depths
        P_cc = torch.cat((P_cc, torch.ones_like(depths)), dim=-1)
        P_wc = poses_all @ torch.transpose(P_cc, 1, 2)
        P_wc = torch.transpose(P_wc, 1, 2)
        P_wc = P_wc[:, :, :3] / P_wc[:, :, 3:4]
        # # Remove undefined depth
        P_wc = P_wc[depths.squeeze(-1) != 0.0]
        # torch.set_printoptions(sci_mode=False)
        # Compute scene boundaries as the pointcloud boundaries
        xyz_scene_min = torch.tensor(
            (P_wc[:, 0].min().item(),
             P_wc[:, 1].min().item(), 
             P_wc[:, 2].min().item())
        )
        xyz_scene_max = torch.tensor(
            (P_wc[:, 0].max().item(),
             P_wc[:, 1].max().item(), 
             P_wc[:, 2].max().item())
        )
        # Compute camera boundaries
        translations = poses_all[:, :3, 3]
        xyz_cam_min = torch.tensor(
            (translations[:, 0].min().item(),
             translations[:, 1].min().item(), 
             translations[:, 2].min().item())
        )
        xyz_cam_max = torch.tensor(
            (translations[:, 0].max().item(),
             translations[:, 1].max().item(), 
             translations[:, 2].max().item())
        )        
        self.scene_boundary = {
            'xyz_scene_min': xyz_scene_min,
            'xyz_scene_max': xyz_scene_max,
            'xyz_cam_min': xyz_cam_min,
            'xyz_cam_max': xyz_cam_max,
        }

        #self.near, self.far = self.config["render"]["depth_range"]


if __name__ == '__main__':
    dataset_root = 'DATASET_DIR'
    scene_n = 'room_1'
    scene = ReplicaSemNerfScene(
        scene_root_dir = os.path.join(dataset_root, scene_n),
        which_split = 'train',
        which_labels=['depth', 'semantics_WF', 'normals_depth'],
        downscale_factor=1.0
    )

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
