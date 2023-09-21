import os
import json
from typing import List
import h5py
import random

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from pytorch3d.transforms import (
    matrix_to_euler_angles,
)

from .cam_model import HypersimCamModel
from .utils import (
    _make_img_id, 
    _split_img_id, 
    _load_image_single,
    _load_selected_label_raw_single, 
    _get_img_id_from_num,
    _convert_single_label_to_vis_format, 
    _downscale_selected_label_all,
    _process_loaded_raw_label_all, 
    _load_selected_label_raw_single,
    _make_filler_label,
    generate_pointcloud,
    _extract_normals_from_depth_batch,
)

class HypersimScene:
    def __init__(
        self, 
        scene_root_dir: str, 
        scene_metadata_path: str, 
        which_labels: List[str], 
        which_split: str,
        split_factor: float = 0.5,
        which_cams: List[str] = ['cam_00'],
        downscale_factor: float = 1.0,
    ):
        self.scene_root_dir = scene_root_dir
        self.scene_metadata_path = scene_metadata_path
        # TODO Sorted is current quickfix for depth to come before normals
        self.which_labels = sorted(which_labels)
        self.which_split = which_split
        self.split_factor = split_factor
        self.which_cams = which_cams
        self.downscale_factor = downscale_factor
        # TODO Implement depth_type as arguments
        self.depth_type = 'distance'

        # TODO Implement better configuration handling 
        # (hydra is probably a good way to go)
        
        self._load_scene_metadata()

        self._load_image_ids()

        self._keep_split_ids()

        self._create_cam_model()

        self._load_images()

        self._load_labels()

        self._correct_metric_units()
        
        self._calculate_scene_boundry()   

        self._calculate_cam_statistics()

        # TODO Place scene clipping (depth clipping) method here
          
        # TODO Parametrize tonemapping

    def _calculate_cam_statistics(self):
        euler_angles = []
        for pose_i in self.cam_model.poses:
            # yaw, pitch, roll convention (from -pi to +pi)
            euler_angles.append(matrix_to_euler_angles(pose_i[:3,:3], 'ZYX'))
        euler_angles = torch.stack(euler_angles, dim=0)
        euler_angles_mean = euler_angles.mean(dim=0)
        euler_angles_std  = euler_angles.std(dim=0)
        
    def _load_scene_metadata(self):
        '''
        Load scene statistics, meaning list of images for each camera trajectory.
        '''
        self.H_orig = 768
        self.W_orig = 1024

        self.H = round(self.H_orig * self.downscale_factor)
        self.W = round(self.W_orig * self.downscale_factor)

        # Extract scene name
        self.scene_name = os.path.basename(self.scene_root_dir)

        if self.scene_metadata_path is not None:
            if os.path.isfile(self.scene_metadata_path):
                print(f'Loading {self.scene_name} metadata from {self.scene_metadata_path}...')
                # If metadata file speicifed
                with open(self.scene_metadata_path, 'r') as f:
                    scene_stats = json.load(f)
                self.scene_metadata = scene_stats[self.scene_name]
                return

        print(f'Extracting {self.scene_name} metadata...')
        # Otherwise compute metadata
        images_root_dir = os.path.join(self.scene_root_dir, 'images')
        rgb_cams_list = [x.name for x in os.scandir(images_root_dir) 
                            if 'final_hdf5' in x.name]
        rgb_cams_list.sort()
        self.scene_metadata = {'cams': {}}
        # Go through all cameras
        for rgb_cam in rgb_cams_list:
            imgs_list = [x.name for x in os.scandir(os.path.join(images_root_dir, rgb_cam))]
            # Randomly shuffle image list, for easy data division later
            random.shuffle(imgs_list)
            random.shuffle(imgs_list)
            random.shuffle(imgs_list) # third time's a charm :)
            # Extract camera_name
            cam_name = '_'.join(rgb_cam.split('_')[1:3])
            self.scene_metadata['cams'][cam_name] = {'img_names': imgs_list}

    def _load_image_ids(self):
        '''Load images ids for all requested camera trajectories.'''
        img_ids_all = {}; cams_list = []

        print('Extracting cameras and their image lists')

        # TODO Hardcoded quick fix:
        # If a scenes does not have cam_00
        if self.which_cams == ['cam_00']:
            if 'cam_00' not in self.scene_metadata['cams']:
                self.which_cams = ['cam_01']
                print('Scene did not have cam_00, instead switched to cam_01...')

        # Go through every camera trajectory
        for cam_n in self.scene_metadata['cams']:
            if self.which_cams:
                if cam_n not in self.which_cams:
                    continue
            # Check if this camera is requested
            img_ids_cam_n = []
            # Load every image id from the trajectory
            for img_n in self.scene_metadata['cams'][cam_n]['img_names']:
                data_dir = os.path.join(self.scene_root_dir, 'images')
                img_path = os.path.join(data_dir, f'scene_{cam_n}_final_hdf5', img_n)
                # Skip non-existing or existing but corrupted image files
                if not os.path.isfile(img_path):
                    continue
                elif not h5py.is_hdf5(img_path):
                    continue
                frame_n = img_n.split(".")[1]
                img_ids_cam_n.append(_make_img_id(cam_n, frame_n))
            img_ids_all[cam_n] = img_ids_cam_n
            cams_list.append(cam_n)
        # Assert there are elements
        assert img_ids_all[list(img_ids_all.keys())[0]]
        # Store
        self.img_ids_all = img_ids_all
        self.cams_list = cams_list
        print(f'(Loaded {len(self.cams_list)} cameras.)')
        print(f'(Loaded {sum([len(img_ids_all[k]) for k in img_ids_all])} image ids in total.)')

    def _keep_split_ids(self):
        img_ids = []
        # Go through each cam and extract ids for specified split
        for cam_n in self.img_ids_all:
            img_ids_n = self.img_ids_all[cam_n]
            split_point = round(self.split_factor * len(img_ids_n))
            # Image ids should be randomly ordered already
            if self.which_split == 'train':
                img_ids_n = img_ids_n[:split_point]
            elif self.which_split == 'test':
                img_ids_n = img_ids_n[split_point:]
            elif self.which_split == 'all':
                img_ids_n = img_ids_n
            else:
                raise NotImplementedError
            # Sort only after division
            img_ids_n.sort()
            img_ids.extend(img_ids_n)
        # Store
        self.img_ids = img_ids

        print(f'Extracted the {self.which_split} and got {len(img_ids)} images left.')

    def _load_images(self):
        '''Load images from all camera trajectories.'''
        print('Loading all images...')
        data_dir = os.path.join(self.scene_root_dir, 'images')
        # Load images from all camera trajectories
        imgs = []
        # Go through every camera trajectory
        for img_id in self.img_ids:
            cam_n, frame_n = _split_img_id(img_id)
            # TODO Put as optional with a configurable parameter
            # TODO Vectorize the tonemap function
            rgb = _load_image_single(
                cam_name=cam_n, 
                frame_name=frame_n,
                data_dir=data_dir,
                apply_tonemap=True
            )
            imgs.append(rgb)
        # Merge everything in one numpy array ([N, H, W, 3])
        imgs = np.asarray(imgs)
        # Convert to torch tensor
        imgs = torch.from_numpy(imgs) 
        # Downscale_factor 
        imgs = self._downscale_selected_label_all(imgs, 'image')
        # Store
        self.imgs = imgs

    def _load_labels(self):
        '''Load requested labels for all loaded images and their metadata.'''
        print(f'Loading all labels {self.which_labels}...')
        self.labels = {}
        self.label_metadata = {}
        for label_name in self.which_labels:
            # Load labels and their metadata
            label_all, label_metadata= self._load_selected_label_all(label_name)
            # Store
            self.labels[label_name] = label_all
            self.label_metadata[label_name] = label_metadata

    def _load_selected_label_all(self, which_label):
        '''Load selected label for all loaded images'''
        if which_label == 'normals_depth':
            label_all = self._extract_normals_from_depth().numpy()
        else:
            label_all = []
            data_dir = os.path.join(self.scene_root_dir, 'images')
            for img_id in self.img_ids:
                cam_name, frame_name = _split_img_id(img_id)
                label = _load_selected_label_raw_single(
                    which_label, 
                    cam_name, 
                    frame_name,
                    data_dir
                )
                if label is None:
                    label = _make_filler_label(which_label, self.H, self.W)
                label_all.append(label)
            # Merge everything in one numpy array ([N, H, W, ?])
            label_all = np.asarray(label_all)
        # Postprocess loaded label and extract their metadata
        kwargs = {'depth_type': self.depth_type , 'scene_metadata': self.scene_metadata}
        label_all, label_metadata = _process_loaded_raw_label_all(
            label_all, 
            which_label,
            **kwargs
        )
        # Convert to torch tensor
        label_all = torch.from_numpy(label_all) 
        # Downscale_factor 
        label_all = self._downscale_selected_label_all(label_all, which_label)

        return label_all, label_metadata
    
    def _downscale_selected_label_all(self, labels_all, which_label):
        if (self.H != self.H_orig) or (self.W != self.W_orig):
            return _downscale_selected_label_all(
                labels_all=labels_all, 
                which_label=which_label, 
                H=self.H, 
                W=self.W
            )
        else:
            return labels_all
    
    def _create_cam_model(self):
        print('Creating camera model...')
        self.cam_model = HypersimCamModel(
            scene_root_dir = self.scene_root_dir,
            img_ids = self.img_ids,
            cams_list = self.cams_list,
            scene_name = self.scene_name,
            depth_type = self.depth_type,
            H_scaled = self.H,
            W_scaled = self.W,
        ) 
    
    def _extract_normals_from_depth(self):
        assert 'depth' in self.labels
        
        print('Extracting normals from depth...')
        normals_gt_depth = _extract_normals_from_depth_batch(
            depth=self.labels['depth'],
            ray_dirs_cc=self.cam_model.ray_dirs_cc,
            poses=self.cam_model.poses[:, :3, :3]
        )
        return normals_gt_depth
        
    def _correct_metric_units(self):
        print('Correcting metric units...')
        if self.cam_model.metric_mode == 'meters':
            pass
        elif self.cam_model.metric_mode == 'asset_units':
            # Convert depth from meters to asset units
            if 'depth' in self.labels:
                self.labels['depth'] /= self.cam_model.m_per_asset_unit
        else:
            raise NotImplementedError  
    
    def _calculate_scene_boundry(self):
        # Try to load first
        if 'scene_boundary' in self.scene_metadata:
            print('Loading scene boudnary from scene metadata...')
            self.scene_boundary = self.scene_metadata['scene_boundary']
            for k in self.scene_boundary:
                self.scene_boundary[k] = torch.tensor(self.scene_boundary[k])
            return

        if 'depth' not in self.labels:
            print('Skipping scene boundary, NO DEPTH.')
            self.scene_boundary = None
            # Cant compute boundaries without depth
            return
        
        print('Extracting scene boudnary...')
        # Otherwise compute the scene boundary
        depths = rearrange(self.labels['depth'], 'b h w -> b (h w) 1')
        P_wc = generate_pointcloud(
            ray_dirs_cc=self.cam_model.ray_dirs_cc, 
            poses=self.cam_model.poses, 
            depths=depths,
            depth_type=self.depth_type)
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
        translations = self.cam_model.poses[:, :3, 3]
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

        # Put initial scene boundaries, by expanding the space around camera boundaries.
        # On each axis: expand by going away from camera bounds with a*current_bound
        # (clip so that it doesnt go out of the pointcloud boundaries)
        xyz_cam_scale = xyz_cam_max - xyz_cam_min
        # # Assert that the z axis is the heigth axis
        # assert xyz_cam_scale.min() == xyz_cam_scale[2]        
        xyz_min_tmp = xyz_scene_min.clone()
        xyz_max_tmp = xyz_scene_max.clone()
        A = 1.5
        xyz_min_tmp[:2] = torch.maximum(xyz_min_tmp[:2], (xyz_cam_min-A*xyz_cam_scale)[:2])
        xyz_max_tmp[:2] = torch.minimum(xyz_max_tmp[:2], (xyz_cam_max+A*xyz_cam_scale)[:2])
        # Discard all points not in calculated bounds from the pointcloud
        # Then shrink new bounds to the remaining pointcloud
        valid_idx = torch.logical_and(
            (P_wc >= xyz_min_tmp.unsqueeze(0)), 
            (P_wc <= xyz_max_tmp.unsqueeze(0))
        )
        valid_idx = valid_idx.min(dim=1)[0]
        P_wc_valid = P_wc[valid_idx]
        if P_wc_valid.nelement() == 0:
            print(f'Could not compute xyz_cam1p5 for {self.scene_name}')
        else:
            self.scene_boundary[f'xyz_cam1p5_min'] = torch.tensor(
                (P_wc_valid[:, 0].min().item(),
                P_wc_valid[:, 1].min().item(), 
                P_wc_valid[:, 2].min().item())
            )
            self.scene_boundary['xyz_cam1p5_max'] = torch.tensor(
                (P_wc_valid[:, 0].max().item(),
                P_wc_valid[:, 1].max().item(), 
                P_wc_valid[:, 2].max().item())
            )
        return

    def _get_z_limits(self):
        min_z = 0.1; max_z = 10.0
        if 'depth' in self.label_metadata:
            min_z = self.label_metadata['depth']['min_depth_z']
            max_z = self.label_metadata['depth']['max_depth_z']
        return min_z, max_z

    def _convert_single_label_to_vis_format(self, label, which_label):
        return _convert_single_label_to_vis_format(
            label=label, 
            label_metadata=self.label_metadata[which_label], 
            which_label=which_label
        )

    ############################################################################
                            # TODO: Necessary? #
    ############################################################################
    def get_img(self, cam_num, frame_num):
        img = None
        img_id = _get_img_id_from_num(cam_num, frame_num)
        for i, img_id_i in enumerate(self.img_ids):
            if img_id == img_id_i:
                img = self.imgs[i]
        return img
    
    def get_label(self, which_label, cam_num, frame_num):
        label = None
        img_id = _get_img_id_from_num(cam_num, frame_num)
        for i, img_id_i in enumerate(self.img_ids):
            if img_id == img_id_i:
                label = self.labels[which_label][i]
        return label   

    ############################################################################
             