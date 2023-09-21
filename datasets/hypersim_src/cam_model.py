from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from einops import rearrange

from .utils import *
from .utils import _load_m_per_asset_unit, _load_cam_poses_single_cam, _split_img_id


class HypersimCamModel:
    def __init__(
        self, 
        scene_root_dir: str,
        scene_name: str,
        img_ids: List[str],
        cams_list: List[str],
        depth_type: str,
        H_scaled: int,
        W_scaled: int,
    ):
        self.scene_root_dir = scene_root_dir
        self.img_ids = img_ids
        self.cams_list = cams_list
        self.scene_name = scene_name
        self.depth_type = depth_type
        self.H = H_scaled
        self.W = W_scaled

        self._load_cam_params()

        self._load_cam_poses()

        self._create_ray_directions_cam_coord()

    def _load_cam_params(self):
        '''Load useful camera parameters.'''
        self.coord_convention = 'opengl'

        self.n_pix = self.H * self.W
        self.aspect_ratio = self.W / self.H

        # url = ('https://raw.githubusercontent.com/apple/ml-hypersim/main/'
        # 'contrib/mikeroberts3000/metadata_camera_parameters.csv')
        #df_camera_parameters_all = pd.read_csv(url, index_col="scene_name")
        csv_path = './datasets/hypersim_src/metadata/metadata_camera_parameters.csv'
        df_camera_parameters_all = pd.read_csv(csv_path, index_col="scene_name")
        df_ = df_camera_parameters_all.loc[self.scene_name]

        # Load meters per asset units
        self.m_per_asset_unit = _load_m_per_asset_unit(self.scene_root_dir)

        # Metric mode
        self.metric_mode = 'asset_units'

        # Matrix to go from the uv space to x,y,z (cam coordinates)
        self.M_cam_from_uv = torch.FloatTensor([
            [df_['M_cam_from_uv_00'], df_['M_cam_from_uv_01'], df_['M_cam_from_uv_02']],
            [df_['M_cam_from_uv_10'], df_['M_cam_from_uv_11'], df_['M_cam_from_uv_12']],
            [df_['M_cam_from_uv_20'], df_['M_cam_from_uv_21'], df_['M_cam_from_uv_22']],
        ])
        # # (scale to meters)
        # self.M_cam_from_uv *= self.m_per_asset_unit
        # Matrix to go from the x, y, z space (cam coordinates) to the uv space 
        self.M_ndc_from_cam = torch.FloatTensor([
            [df_['M_proj_00'], df_['M_proj_01'], df_['M_proj_02'], df_['M_proj_03']],
            [df_['M_proj_10'], df_['M_proj_11'], df_['M_proj_12'], df_['M_proj_13']],
            [df_['M_proj_20'], df_['M_proj_21'], df_['M_proj_22'], df_['M_proj_23']],
            [df_['M_proj_30'], df_['M_proj_31'], df_['M_proj_32'], df_['M_proj_33']],
        ])
        # # (scale to meters)
        # self.M_ndc_from_cam /= self.m_per_asset_unit
        self.M_uv_from_ndc = torch.FloatTensor(
            [[0.5*(self.W-1),  0,              0,   0.5*(self.W-1)],
             [0,              -0.5*(self.H-1), 0,   0.5*(self.H-1)],
             [0,               0,              0.5, 0.5],
             [0,               0,              0,   1.0]])
        ########################################################################
        # # Useful info at:
        # # https://stackoverflow.com/questions/11277501/how-to-recover-view-space-position-given-view-space-depth-value-and-ndc-xy/46118945#46118945
        # #self.hfov_rad = 2.0*np.arctan(1.0/df_['M_proj_11'])
        # self.wfov_rad = df_['settings_camera_fov']
        # self.hfov_rad = 2.0 * np.arctan(self.H * np.tan(self.wfov_rad/2.0) / self.W)
        # # the pin-hole camera has the same value for fx and fy
        # # Should the original f be first computed and then downscaled?
        # # Seems not since the absolute H,W + angle dose the job
        # self.fx = self.W / 2.0 / math.tan(self.wfov_rad / 2.0)
        # self.fy = self.H / 2.0 / math.tan(self.hfov_rad / 2.0)
        # # Info that indicates an unusual focal length can be found here:
        # # https://github.com/apple/ml-hypersim/issues/9
        # # self.fx = 886.81; self.fy = self.fx
        # self.cx = (self.W - 1.0) / 2.0
        # self.cy = (self.H - 1.0) / 2.0

        # self.poses_staticcam = None
        ########################################################################

    def _load_cam_poses(self):
        '''Load camera poses for all camera trajectories'''

        # Load camera poses for all cameras
        cam_poses_all_cams = self._load_cam_poses_all_cams()

        # Reorder camera poses to match the self.img_ids list
        poses = self._reorder_cam_poses_with_img_ids(
            cam_poses_all_cams=cam_poses_all_cams
        )

        # Store
        self.poses = poses
    
    def _load_cam_poses_all_cams(self):
        '''Load all camera poses for all cameras'''
        cam_poses_all_cams = {}
        for cam_n in self.cams_list:
            cam_poses = _load_cam_poses_single_cam(
                scene_root_dir=self.scene_root_dir, 
                cam_name=cam_n, 
            )
            # # (scale to meters)
            #cam_poses[:3, 3] *= self.m_per_asset_unit
            cam_poses_all_cams[cam_n] = cam_poses
        return cam_poses_all_cams

    def _reorder_cam_poses_with_img_ids(self, cam_poses_all_cams):
        '''
        Take camera poses which area loaded for all cameras.
        Reorder them to mathc the self.img_ids list
        '''
        poses = []
        # Select onl the camera poses of loaded images
        for img_id in self.img_ids:
            cam_name, frame_name = _split_img_id(img_id)
            frame_i = int(frame_name)

            cam_poses = cam_poses_all_cams[cam_name]
            # Pose index for current cam which corresponds to img_id
            if cam_poses['frame_idx'][frame_i] == frame_i:
                cam_n_i = frame_i
            else:
                cam_n_i_where = np.where(cam_poses['frame_idx'] == frame_i)
                assert len(cam_n_i_where) == 1
                cam_n_i = cam_n_i_where[0].item()
            
            poses.append(cam_poses['poses'][cam_n_i])
        # Merge everything in one numpy array ([N, H, W, 3])
        poses = torch.stack(poses)

        return poses

    def _create_ray_directions_cam_coord(self):
        '''
        Create pixel center positions (ray directions) in the camera coordinates.
        '''
        # Create grid of pixel center positions in the uv space
        u_min  = -1.0; u_max  = 1.0
        v_min  = -1.0; v_max  = 1.0
        half_du = 0.5 * (u_max - u_min) / self.W
        half_dv = 0.5 * (v_max - v_min) / self.H
        u_linspace = np.linspace(u_min + half_du, u_max - half_du, self.W)
        v_linspace = np.linspace(v_min + half_dv, v_max - half_dv, self.H)
        # Reverse vertical coordinate because [H=0,W=0] 
        # corresponds to (u=-1, v=1).
        u, v = np.meshgrid(u_linspace, v_linspace[::-1])
        # Add 3rd coordinate and reshape to vector
        uvs_2d = np.dstack((u, v, np.ones_like(u)))
        uvs_2d = torch.FloatTensor(uvs_2d)
        ray_dirs_uv = rearrange(uvs_2d, 'h w xyz-> (h w) xyz')

        # Create grid of pixel center indices in the uv space
        # Then reshape to vectors
        u_idx_linspace = np.arange(0, self.W)
        v_idx_linspace = np.arange(0, self.H)
        u_idx, v_idx = np.meshgrid(u_idx_linspace, v_idx_linspace)
        idx_2d = np.dstack((u_idx, v_idx))
        idx_2d = torch.IntTensor(idx_2d)
        ray_idxs = rearrange(idx_2d, 'h w i-> (h w) i')

        # Transfer pixel center positions from uv to the camera space
        # (???) If seems that the near clipping plane (=image plane) 
        # corresponds to |M_cam_from_uv_22|, in the case of regular pinhole camera (???)
        # Very often, the near clipping plane has distance 1.0 in asset units, in that case. 
        ray_dirs_cc = self.M_cam_from_uv @ ray_dirs_uv.T
        # Pixel center points in camera coordinates [N, HW, 3]
        ray_dirs_cc = ray_dirs_cc.T
        # Normalize with respect to depth type
        if self.depth_type == 'z':
            # Normalize such that |z| = 1
            ray_dirs_cc = (ray_dirs_cc / torch.abs(ray_dirs_cc[:, 2:3]))
        elif self.depth_type == 'distance':
            # Normalize such that ||ray_dir||=1
            ray_dirs_cc = F.normalize(ray_dirs_cc, p=2, dim=-1)
        else:
            raise NotImplementedError

        # Store
        self.ray_dirs_cc = ray_dirs_cc
        self.ray_dirs_uv = ray_dirs_uv
        self.ray_idxs = ray_idxs