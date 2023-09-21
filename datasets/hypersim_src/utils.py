import os

import h5py

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import einops

from imgviz import label_colormap, depth2rgb

################################################################################
                             # Load single raw #
################################################################################

def _load_image_single(cam_name, frame_name, data_dir, apply_tonemap):
    '''
    Load image and apply appropriate tonemaping.
    '''   
    img_path = os.path.join(
        data_dir,
        f'scene_{cam_name}_final_hdf5',
        f'frame.{frame_name}.color.hdf5'
    )

    rgb = h5py.File(img_path, 'r')['dataset'][:].astype('float32')

    if apply_tonemap:
        render_entity_id_path = os.path.join(
            data_dir, 
            f'scene_{cam_name}_geometry_hdf5',
            f'frame.{frame_name}.render_entity_id.hdf5'
        )
        render_entity_id = h5py.File(render_entity_id_path, 'r')['dataset'][:]
        render_entity_id = render_entity_id.astype('int32')

        rgb = _scene_generated_images_tonemap(rgb, render_entity_id)

    return rgb


def _load_selected_label_raw_single( 
    which_label, 
    cam_name, 
    frame_name, 
    data_dir
):
    '''Load selected label corresponding to a single image'''
    if which_label in ['semantics', 'semantics_WF']:
        return _load_sem_raw_single(cam_name, frame_name, data_dir)
    elif which_label == 'depth':
        return _load_depth_raw_single(cam_name, frame_name, data_dir)
    elif which_label == 'normals':
        return _load_norm_raw_single(cam_name, frame_name, data_dir)
    else:
        raise AssertionError


def _load_sem_raw_single(cam_name, frame_name, data_dir):
    '''Load semantic mask corresponding to a single image'''
    semantics_path = os.path.join(
        data_dir,
        f'scene_{cam_name}_geometry_hdf5',
        f'frame.{frame_name}.semantic.hdf5'
    )
    if not _is_valid_label_file(semantics_path):
        return None
    semantics_raw = h5py.File(semantics_path, 'r')['dataset'][:]
    semantics_raw = semantics_raw.astype('int64')
    return semantics_raw


def _load_depth_raw_single(cam_name, frame_name, data_dir):
    '''Load a depth map correspodning to a single image'''
    depth_path = os.path.join(
        data_dir,
        f'scene_{cam_name}_geometry_hdf5',
        f'frame.{frame_name}.depth_meters.hdf5'
    )
    if not _is_valid_label_file(depth_path):
        return None
    depth_dist_raw =  h5py.File(depth_path, 'r')['dataset'][:]
    depth_dist_raw = depth_dist_raw.astype('float32')
    return depth_dist_raw


def _load_norm_raw_single(cam_name, frame_name, data_dir):
    '''Load surface normals correspodning to a single image'''
    normals_path = os.path.join(
        data_dir,
        f'scene_{cam_name}_geometry_hdf5',
        f'frame.{frame_name}.normal_bump_world.hdf5'
    )
    # There are multiple options for normals:
    # .normal_bump_world (bump mapping, world space)
    # .normal_bump_cam (bump mapping, camera space)
    # .normal_world (no bump mapping, world space)
    # .normal_cam (no bump mapping, camera space)
    if not _is_valid_label_file(normals_path):
        return None
    normals_raw =  h5py.File(normals_path, 'r')['dataset'][:]
    normals_raw = normals_raw.astype('float32')
    return normals_raw


def _make_filler_label(which_label, H, W):
    '''In case of corrupted or non-existing label file, create a filler label.'''
    if which_label == 'semantics':
        return -1 * np.ones((H, W), dtype=np.int64)
    elif which_label == 'depth':
        return np.zeros((H, W), dtype=np.float32)
    elif which_label == 'normals':
        return np.zeros((H, W, 3), dtype=np.float32)
    else:
        raise AssertionError


def _is_valid_label_file(file_path):
    if not os.path.isfile(file_path):
        return False
    elif not h5py.is_hdf5(file_path):
        return False
    else:
        return True


################################################################################
                             # Process multiple #
################################################################################


def _process_loaded_raw_label_all(label_all, which_label, **kwargs):
    '''Postprocess loaded raw labels and extract metadata.'''
    if which_label in ['semantics', 'semantics_WF']:
        fn = _process_raw_sem_all
    elif which_label == 'depth':
        fn = _process_raw_depth_all
    elif which_label in ['normals', 'normals_depth']:
        fn = _process_raw_norm_all
    else:
        raise NotImplementedError

    kwargs['which_label'] = which_label
    label_all, label_metadata = fn(label_all, **kwargs)
    return label_all, label_metadata


def _process_raw_sem_all(semantics_all, **kwargs):
    scene_metadata = kwargs['scene_metadata']

    # Move the void class from -1 to the empty 0 value
    semantics_all[semantics_all==-1] = 0

    if 'semantic_metadata' in scene_metadata:
        # Load semantic information from metadata
        # This avoids having different class ids in the train and test split
        metadata_dict = scene_metadata['semantic_metadata'].copy()
        for k in metadata_dict:
            if isinstance(metadata_dict[k], list):
                metadata_dict[k] = np.array(metadata_dict[k]) 
    else:
        # General semanti info
        # Load NYU40 classes info
        url = ("https://raw.githubusercontent.com/apple/ml-hypersim/main/"
        "code/cpp/tools/scene_annotation_tool/semantic_label_descs.csv")
        nyu40_labels_df = pd.read_csv(url)
        class_names_nyu40 = [' void'] + list(nyu40_labels_df.iloc[:, 1].values)
        n_classes_nyu40 = len(class_names_nyu40)
        # Create colour map
        # TODO: Take RGB values that NYU defined(can be found in nyu40_labels_df)
        id_to_colour_nyu40 = label_colormap(n_classes_nyu40)

        # Per scene semantic info
        # Unique semantic class ids, including the void class of 0
        class_ids_scene = np.unique(semantics_all).astype(np.uint8)
        # Number of semantic classes, including the void class of 0
        n_classes_scene = class_ids_scene.shape[0]
        n_valid_classes_scene = n_classes_scene - 1
        id_to_colour_scene = id_to_colour_nyu40[class_ids_scene]
        id_to_colour_scene_torch = torch.from_numpy(id_to_colour_scene)
        # Remap labels
        # TODO: Plot semantic label legend (semantic_nerf code)
        metadata_dict = {
            'class_ids_scene': class_ids_scene,
            'n_classes_scene': n_classes_scene,
            'n_valid_classes_scene': n_valid_classes_scene,
            'id_to_colour_scene': id_to_colour_scene,
            #'id_to_colour_scene_torch': id_to_colour_scene_torch,
            #'class_names_nyu40': class_names_nyu40,
            #'n_classes_nyu40': n_classes_nyu40,
            #'id_to_colour_nyu40': id_to_colour_nyu40
        }

    if kwargs['which_label'] == 'semantics':
        # Give new class IDs to go from [0, C)
        for new_id, old_id in enumerate(metadata_dict['class_ids_scene']):
            semantics_all[semantics_all == old_id] = new_id
    elif kwargs['which_label'] == 'semantics_WF':
        url = ("https://raw.githubusercontent.com/apple/ml-hypersim/main/"
        "code/cpp/tools/scene_annotation_tool/semantic_label_descs.csv")
        nyu40_labels_df = pd.read_csv(url)
        class_names_nyu40 = [' void'] + list(nyu40_labels_df.iloc[:, 1].values)
        if 1 not in metadata_dict['class_ids_scene']:
            print('[semantics_WF] WALL CLASS DOESNT EXIST!')
        if 2 not in metadata_dict['class_ids_scene']:
            print('[semantics_WF] FLOOR CLASS DOESNT EXIST!')
        
        # Merge window into wall class
        semantics_all[semantics_all == 9] = 1
        # # Merge mirror into wall class
        # semantics_all[semantics_all == 19] = 1
        # Merge floormat into floor class
        semantics_all[semantics_all == 20] = 2

        FW_mask = (semantics_all == 1) + (semantics_all == 2)
        semantics_all[np.logical_not(FW_mask)] = 3
        metadata_dict['n_valid_classes_scene'] = 3

        # semantics_all[np.logical_not(FW_mask)] = 0
        # metadata_dict['n_valid_classes_scene'] = 0

        # import matplotlib.pyplot as plt
        # import matplotlib
        # sem_colormap = label_colormap(3)
        # matplotlib.image.imsave('cam_00.0000_semantics_WF_GT.png', sem_colormap[semantics_all[0]])
        # plt.imshow(sem_colormap[semantics_all[0]])
        # plt.show()
    else:
        raise NotImplementedError

    return semantics_all, metadata_dict


def _process_raw_norm_all(normals_all, **kwargs):
    #normals_all = _interpolate_zero_normals(normals_all)
    normals_all = _interpolate_nan_normals(normals_all)
    return normals_all, {}


def _process_raw_depth_all(depth_all, **kwargs):
    # Replace nan values with regional averages
    depth_all = _interpolate_nan_depth(depth_all)

    # min_depth_dist = np.min(depth_all).item()
    # max_depth_dist = np.max(depth_all).item()
    
    if kwargs['depth_type'] == 'z':
        # Convert depth from distance to center of projection convention
        # into the z-plane convention.
        im_plane_norm2_inv_mul_focal = _depth_to_z_helpers()
        depth_all = depth_all * np.expand_dims(im_plane_norm2_inv_mul_focal, axis=0)
    elif kwargs['depth_type'] == 'distance':
        pass
    else:
        raise NotImplementedError

    # min_depth_z = np.min(depth_all).item()
    # max_depth_z = np.max(depth_all).item()
    metadata_dict = {}
    # metadata_dict = {
    #     'min_depth_dist': min_depth_dist,
    #     'max_depth_dist': max_depth_dist,
    #     'min_depth_z': min_depth_z,
    #     'max_depth_z': max_depth_z
    # }
    return depth_all, metadata_dict


def _interpolate_nan_depth(depth_all):
    '''Replaces all NaN depth value with a local window mean if makes sense'''
    depth_all[np.isnan(depth_all)] = 0.0
    return depth_all
    # VAR_TRESH = 0.1
    # for nan_loc in np.argwhere(np.isnan(depth_all)):
    #     i, h, w = nan_loc
    #     win_margin = 2
    #     h_low = max(0, h.item() - win_margin)
    #     h_high = min(depth_all.shape[1], h.item() + win_margin)
    #     w_low = max(0, w.item() - win_margin)
    #     w_high = min(depth_all.shape[2], w.item() + win_margin)
    #     mean_region = depth_all[i, h_low:h_high+1, w_low:w_high+1]
    #     mean_region = mean_region[np.invert(np.isnan(mean_region))]
    #     if (np.var(mean_region) < VAR_TRESH and (len(mean_region) > 2)):
    #         depth_all[i, h, w] = np.mean(mean_region)
    #     else:
    #         # TODO : Put -1 as a non-selection index
    #         depth_all[i, h, w] = 0.0
    # return depth_all


def _interpolate_nan_normals(normals_all):
    '''Replaces all NaN normal vectors with a local window mean if makes sense'''
    zero_vect = np.zeros(3, dtype=normals_all.dtype)
    normals_all[np.isnan(np.abs(normals_all).sum(axis=-1))] = zero_vect
    return normals_all 
    # VAR_TRESH = 0.05
    # for nan_loc in np.argwhere(np.isnan(normals_all.sum(axis=-1))):
    #     i, h, w = nan_loc
    #     win_margin = 2
    #     h_low = max(0, h.item() - win_margin)
    #     h_high = min(normals_all.shape[1], h.item() + win_margin)
    #     w_low = max(0, w.item() - win_margin)
    #     w_high = min(normals_all.shape[2], w.item() + win_margin)
    #     mean_region = normals_all[i, h_low:h_high+1, w_low:w_high+1]
    #     # Remove nan from window
    #     mean_region = mean_region[np.invert(np.isnan(mean_region.sum(axis=-1)))]
    #     # Remove zero vectors from window
    #     mean_region = mean_region[np.invert(mean_region.sum(axis=-1) == 0.0)]
    #     var = np.var(mean_region[:,0])
    #     var += np.var(mean_region[:,1])
    #     var += np.var(mean_region[:,2])
    #     var /= 3.0
    #     if (var < VAR_TRESH and (len(mean_region) > 2)):
    #         normals_all[i, h, w] = np.mean(mean_region, axis=0)
    #     else:
    #         # TODO : Put -1 as a non-selection index
    #         normals_all[i, h, w] = np.zeros(3, dtype=mean_region.dtype)
    # return normals_all


# def _interpolate_zero_normals(normals_all):
#     '''Replaces all zero vector normals with a local window mean'''
#     normalize = lambda v: v / np.sqrt(np.sum(v**2))
#     VAR_TRESH = 0.05
#     for zero_loc in np.argwhere(normals_all.sum(axis=-1) == 0.0):
#         i, h, w = zero_loc
#         win_margin = 2
#         h_low = max(0, h.item() - win_margin)
#         h_high = min(normals_all.shape[1], h.item() + win_margin)
#         w_low = max(0, w.item() - win_margin)
#         w_high = min(normals_all.shape[2], w.item() + win_margin)
#         mean_region = normals_all[i, h_low:h_high+1, w_low:w_high+1]
#         mean_region = mean_region[np.invert(mean_region.sum(axis=-1) == 0.0)]
#         var = np.var(mean_region[:,0])
#         var += np.var(mean_region[:,1])
#         var += np.var(mean_region[:,2])
#         var /= 3.0
#         if (var < VAR_TRESH and (len(mean_region) > 2)):
#             norm_mean = np.mean(mean_region)
#             normals_all[i, h, w] = normalize(norm_mean)
#         else:
#             # TODO : Put -1 as a non-selection index
#             normals_all[i, h, w] = 0.0
#     return normals_all


def _downscale_selected_label_all(labels_all, which_label, H, W):
    new_size = (H, W)

    if which_label == 'image':
        labels_all = F.interpolate(
            labels_all.permute(0,3,1,2,), 
            size = new_size,
            mode='bilinear'
        ).permute(0,2,3,1)

    elif which_label == 'depth':
        # Nearest neighbor instead of bilinear, 
        # in order to retain missing elements (zeros)
        labels_all =  F.interpolate(
            torch.unsqueeze(labels_all, dim=1), 
            size = new_size,
            mode='nearest' #'bilinear'
        ).squeeze(1)

    elif which_label in ['normals', 'normals_depth']:
        # Nearest neighbor instead of bilinear, 
        # in order to retain missing elements (zero vectors)
        labels_all = F.interpolate(
            labels_all.permute(0,3,1,2), 
            size = new_size,
            mode='nearest' #'bilinear'
        ).permute(0,2,3,1)
        # Normalize normals to unit length, because
        # unit length gets ruined after interpolation.
        # But keep the zero vector missing element code.
        with torch.no_grad():
            zero_idx = (labels_all.abs().sum(-1) != 0.0).unsqueeze(-1)
            labels_all_norm = F.normalize(labels_all, p=2.0, dim=-1)
            labels_all = torch.where(zero_idx, labels_all, labels_all_norm)

    elif which_label == 'semantics':
        labels_all = F.interpolate(
            torch.unsqueeze(labels_all, dim=1).float(), 
            size = new_size,
            mode='nearest'
        ).squeeze(1).type(labels_all.dtype)

    return labels_all


################################################################################
                    # Load camera trajectory & related #
################################################################################


def _load_cam_poses_single_cam(scene_root_dir, cam_name):
    cam_dir = os.path.join(scene_root_dir, '_detail', cam_name)
    trans_f = os.path.join(cam_dir, 'camera_keyframe_positions.hdf5')
    rot_f = os.path.join(cam_dir, 'camera_keyframe_orientations.hdf5')
    frame_idx_f = os.path.join(cam_dir, 'camera_keyframe_frame_indices.hdf5')

    translations = h5py.File(trans_f, 'r')['dataset'][:]
    rotations = h5py.File(rot_f, 'r')['dataset'][:]
    frame_idx = h5py.File(frame_idx_f, 'r')['dataset'][:]

    # look_at_pos_f = os.path.join(cam_dir, 'camera_keyframe_look_at_positions.hdf5')
    # look_at_pos = h5py.File(look_at_pos_f, 'r')['dataset'][:]

    translations = torch.from_numpy(translations).type(torch.FloatTensor)
    rotations = torch.from_numpy(rotations).type(torch.FloatTensor)

    poses = _construct_pose_matrices(translations, rotations)
    
    cam_poses = {
        'poses': poses,
        'frame_idx': frame_idx}
    return cam_poses


def _construct_pose_matrices(translations, rotations):
# Construct pose matrices
    P = torch.cat(
        (rotations, translations.unsqueeze(-1)),
        dim=2)
    tmp = torch.zeros((P.shape[0], 1, 4))
    tmp[:, 0, -1] = 1.0
    P =  torch.cat((P, tmp), dim=1)
    return P


def _construct_pose_matrix(translation, rotation):
# Construct pose matrices
    P = torch.cat(
        (rotation, translation.unsqueeze(-1)),
        dim=1)
    tmp = torch.zeros((1, 4))
    tmp[0, -1] = 1.0
    P =  torch.cat((P, tmp), dim=0)
    return P


def _load_m_per_asset_unit(scene_root_dir):
    '''Load conversion factor from scene units to meters'''
    scene_metadata_file_path  = os.path.join(
        scene_root_dir, 
        '_detail', 
        'metadata_scene.csv'
    )
    scene_metadata = pd.read_csv(scene_metadata_file_path)
    tmp_idx = (scene_metadata['parameter_name'] == 'meters_per_asset_unit')
    m_per_asset_unit = scene_metadata.loc[tmp_idx, 'parameter_value'].iloc[0]
    return m_per_asset_unit


################################################################################
                # 3D stuff #
################################################################################


def generate_pointcloud(ray_dirs_cc, poses, depths, depth_type):
    # Pixel center positions in camera coordinates, 
    # unprojected by depth
    ray_dirs_cc = ray_dirs_cc.clone().unsqueeze(0)
    depths = depths.clone()
    if depths.dim() == 2:
        depths = depths.unsqueeze(-1)
    else:
        assert depths.dim() == 3

    if depth_type == 'z':
        # Normalize such that |z| = 1
        ray_dirs_cc = (ray_dirs_cc / torch.abs(ray_dirs_cc[:, :, 2:3]))
    elif depth_type == 'distance':
        # Normalize such that ||ray_dir||=1
        ray_dirs_cc = F.normalize(ray_dirs_cc, p=2, dim=-1)
    else:
        raise NotImplementedError
    P_cc = ray_dirs_cc * depths
    P_cc = torch.cat((P_cc, torch.ones_like(depths)), dim=-1)
    P_wc = poses @ torch.transpose(P_cc, 1, 2)
    P_wc = torch.transpose(P_wc, 1, 2)
    P_wc = P_wc[:, :, :3] / P_wc[:, :, 3:4]

    return P_wc


def clip_depths_to_bbox(depths, P_wc, poses, xyz_min, xyz_max):
    assert depths.dim() == 2
    xyz_min = xyz_min.clone().unsqueeze(0).unsqueeze(0)
    xyz_max = xyz_max.clone().unsqueeze(0).unsqueeze(0)

    P_wc_bnd = P_wc.clone()
    P_wc_bnd = torch.where(P_wc > xyz_max, xyz_max, P_wc_bnd)
    P_wc_bnd = torch.where(P_wc < xyz_min, xyz_min, P_wc_bnd)
    S = (P_wc_bnd - poses[:, None, :3, 3]) / (P_wc - poses[:, None, :3, 3])
    S = torch.where((depths.unsqueeze(-1) == 0.0), torch.ones_like(S), S)
    S = torch.min(S, dim=-1, keepdim=True)[0]
    S = S[...,0]
    depths = depths * S
    return depths

@torch.cuda.amp.autocast(dtype=torch.float32)
def _extract_normals_from_ray_batch(
    rays_o:   torch.Tensor, 
    rays_d:   torch.Tensor, 
    depth:    torch.Tensor,
    x123_idx: dict):
    # Get points in WC
    P_wc = rays_o + rays_d * depth.unsqueeze(-1)
    # Sample triangle points P1, P2, P3 for each pixel neighbourhood
    # ____|_P2_|____
    # _P3_|_P1_|____
    #     |    |     
    # N, _ = P_wc.shape
    # assert N % 3 == 0
    # N = N // 3
    # P1_wc = P_wc[:N]
    # P2_wc = P_wc[N:2*N]
    # P3_wc = P_wc[2*N: 3*N]
    P1_wc = P_wc[x123_idx['x1']]
    P2_wc = P_wc[x123_idx['x2']]
    P3_wc = P_wc[x123_idx['x3']]

    # Extract normas as a cross product over the local pixel triangle
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7335535&tag=1
    normals_extracted = torch.cross((P2_wc-P1_wc), (P3_wc-P1_wc), dim=-1)
    # Normalize to init length
    normals_extracted = F.normalize(normals_extracted, p=2.0, dim=-1)
    # # TODO Turned off
    # # There are two possible normals for both sides of the surface.
    # # The surface side which the camera sees gives the correct normal
    # # https://pcl.readthedocs.io/projects/tutorials/en/latest/normal_estimation.html
    # condition = torch.sum(normals_extracted * (rays_o[:N] - P1_wc), dim=-1, keepdim=True)
    # normals_extracted = torch.where(condition > 0.0, normals_extracted, -1.0 * normals_extracted)
    
    # # Assign the same normal to P1, P2 and P3
    # normals_extracted = normals_extracted.repeat(3, 1)
    
    return normals_extracted

@torch.cuda.amp.autocast(dtype=torch.float32)
def _extract_normals_from_depth_batch(
    depth: torch.Tensor,         # b h w
    ray_dirs_cc: torch.Tensor,   # (hw) 3
    poses: torch.Tensor):        # b 4 4    
    # Clone, to not modify content
    depth = depth.clone()
    ray_dirs_cc = ray_dirs_cc.clone()
    poses = poses.clone()

    # Get shape
    B, H, W = depth.shape

    # Rearange to compatible shapes for a vectorized matmul
    depth = einops.rearrange(depth, 'b h w -> b (h w) 1')
    ray_dirs_cc = einops.rearrange(ray_dirs_cc, 'hw d -> 1 hw d')
    
    # Create pointcloud in camera coordinates
    P_cc = ray_dirs_cc * depth

    # Another option is to go to WC first and not have to switch at the end
    # P_cc = torch.cat((P_cc, torch.ones_like(depths)), dim=-1)
    # P_wc = poses @ torch.transpose(P_cc, 1, 2)
    # P_wc = torch.transpose(P_wc, 1, 2)
    # # Unnecessary, they should be all ones anyways
    # P_wc = P_wc[:, :, :3] / P_wc[:, :, 3:4] 
    # P_wc = einops.rearrange(P_wc, 'b (h w) c -> b h w c', h=H)

    # Sample triangle points P1, P2, P3 for each pixel neighbourhood
    # ____|_P2_|____
    # _P3_|_P1_|____
    #     |    |     
    P_cc = einops.rearrange(P_cc, 'b (h w) d -> b h w d', h=H)
    P1_cc = P_cc[:, 1:-1, 1:-1]
    P2_cc = P_cc[:, :-2, 1:-1]
    P3_cc = P_cc[:, 1:-1, :-2]
    # Extract normas as a cross product over the local pixel triangle
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7335535&tag=1
    normals_extracted = torch.cross((P2_cc-P1_cc), (P3_cc-P1_cc), dim=-1)
    # Normalize extracted normals to unit length
    normals_extracted = F.normalize(normals_extracted, p=2.0, dim=-1)
    
    # TODO Turned off
    # # There are two possible normals for both sides of the surface.
    # # The surface side which the camera sees gives the correct normal
    # # https://pcl.readthedocs.io/projects/tutorials/en/latest/normal_estimation.html
    # vp = torch.zeros((1,1,1,3)).to(depth.device) # Viewpoint # TODO Is the camera centered?
    # condition = torch.sum(normals_extracted * (vp - P1_cc), dim=-1, keepdim=True)
    # # Flip normals so they correspond to surfaces the camera can see
    # normals_extracted = torch.where(condition > 0.0, normals_extracted, -1.0 * normals_extracted)

    # Rotate normal from CC to WC 
    # (No need to translate, we just want the vector orientation)
    normals_extracted = einops.rearrange(normals_extracted, 'b h w d -> b d (h w)')
    normals_extracted = poses[:, :3,:3] @ normals_extracted
    normals_extracted = einops.rearrange(normals_extracted, 'b d (h w) -> b h w d', h=H-2, w=W-2)
    
    # Pad with zeros (ignore code) because the 1-thich border 
    # could not have normals estiamted.
    normals_extracted = normals_extracted.permute(0, 3, 1, 2)
    normals_extracted = torch.nn.ZeroPad2d(1)(normals_extracted)
    normals_extracted = normals_extracted.permute(0, 2, 3, 1)

    # TODO Put 0 normals for invalid depth
    depth = einops.rearrange(depth, 'b (h w) 1 -> b h w', h=H)
    invalid_idx = (depth == 0.0) + torch.isnan(depth) + torch.isinf(depth)
    normals_extracted[invalid_idx] = torch.zeros(3, device=depth.device)

    return normals_extracted

################################################################################
                # Convert single label to visualizable format #
################################################################################
    

def _convert_single_label_to_vis_format(label, label_metadata, which_label):
    import cv2
    def depth2img(depth):
        depth = (depth-depth.min())/(depth.max()-depth.min())
        depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                    cv2.COLORMAP_TURBO)

        return depth_img
    label_vis = label.clone().cpu().detach()

    if which_label in ['semantics', 'semantics_WF']:
        label_vis =  torch.tensor(label_metadata['id_to_colour_scene'])[label.long()]
    elif which_label == 'depth':
        label_vis = depth2img(label_vis.numpy())
        # label_vis = depth2rgb(
        #     label_vis.numpy(),
        #     min_value=label_metadata['min_depth_z'], 
        #     max_value=label_metadata['max_depth_z']
        # )
        label_vis = torch.from_numpy(label_vis)
    elif which_label in ['normals', 'normals_depth']:
        #pass
        label_vis = (label_vis + 1.0) * 0.5
    else:
        raise NotImplemented

    return label_vis


################################################################################
                             # Image id convention #
################################################################################


def _make_img_id(cam_name, frame_name):
    return f'{cam_name}.{frame_name}'


def _get_img_id_from_num(cam_num, frame_num):
    return f'cam_{cam_num:02d}.{frame_num:04d}'


def _split_img_id(img_id):
    cam_name, frame_name = img_id.split('.')
    return cam_name, frame_name


################################################################################
                             # Misc #
################################################################################


def vecs_to_unit_np(vecs):
    '''Normalize a nd-array of vectors to unit lengh'''
    # Vectors are contained at the last dimension
    norm = np.linalg.norm(
        vecs,
        ord=2,
        axis=-1,
        keepdims=True
    )
    return vecs / norm


def _scene_generated_images_tonemap(rgb_color, render_entity_id):
    '''
    Code taken from:
    https://github.com/apple/ml-hypersim/blob/main/code/python/tools/scene_generate_images_tonemap.py
            
    Compute brightness according to "CCIR601 YIQ" method, 
    use CGIntrinsics strategy for tonemapping, see [1,2]
    [1] https://github.com/snavely/pbrs_tonemapper/blob/master/tonemap_rgbe.py
    [2] https://landofinterruptions.co.uk/manyshades
    '''
    assert (render_entity_id != 0).all()
    # standard gamma correction exponent
    gamma                             = 1.0/2.2   
    inv_gamma                         = 1.0/gamma
    # we want this percentile brightness value in the unmodified image...
    percentile                        = 90
    # ...to be this bright after scaling        
    brightness_nth_percentile_desired = 0.8       

    valid_mask = render_entity_id != -1
    
    if np.count_nonzero(valid_mask) == 0:
        # if there are no valid pixels, then set scale to 1.0
        scale = 1.0 
    else:
        # "CCIR601 YIQ" method for computing brightness
        brightness = 0.3*rgb_color[:,:,0] + 0.59*rgb_color[:,:,1] + 0.11*rgb_color[:,:,2] 
        brightness_valid = brightness[valid_mask]

        # if the kth percentile brightness value in the unmodified image is 
        # less than this, set the scale to 0.0 to avoid divide-by-zero
        eps                               = 0.0001
        brightness_nth_percentile_current = np.percentile(brightness_valid, percentile)

        if brightness_nth_percentile_current < eps:
            scale = 0.0
        else:

            # Snavely uses the following expression in the code at 
            # https://github.com/snavely/pbrs_tonemapper/blob/master/tonemap_rgbe.py:
            # scale = np.exp(np.log(brightness_nth_percentile_desired)*inv_gamma 
            #         - np.log(brightness_nth_percentile_current))
            #
            # Our expression below is equivalent, but is more intuitive, 
            # because it follows more directly from the expression:
            # (scale*brightness_nth_percentile_current)^gamma = 
            #  = brightness_nth_percentile_desired

            scale = np.power(brightness_nth_percentile_desired, inv_gamma) 
            scale /= brightness_nth_percentile_current

    rgb_color_tm = np.power(np.maximum(scale*rgb_color,0), gamma)
    
    return np.clip(rgb_color_tm, 0, 1)


def _depth_to_z_helpers():
    '''
    Create helper values to convert depth from a distance to camera center
    to a z-plane.
    Code taken from: 
    https://github.com/apple/ml-hypersim/issues/9#issuecomment-754935697
    '''
    h_orig = 768 
    w_orig = 1024
    focal_flt = 886.81 # For full resolution image
    w_lim_min = (-0.5 * w_orig) + 0.5
    w_lim_max = (0.5 * w_orig) - 0.5
    im_plane_X = np.linspace(w_lim_min, w_lim_max, w_orig)
    im_plane_X = im_plane_X.reshape(1, w_orig).repeat(h_orig, 0)
    im_plane_X = im_plane_X.astype(np.float32)[:, :, None]

    h_lim_min = (-0.5 * h_orig) + 0.5
    h_lim_max = (0.5 * h_orig) - 0.5
    im_plane_Y = np.linspace(h_lim_min, h_lim_max, h_orig)
    im_plane_Y = im_plane_Y.reshape(h_orig, 1).repeat(w_orig, 1)
    im_plane_Y = im_plane_Y.astype(np.float32)[:, :, None]

    im_plane_Z = np.full([h_orig, w_orig, 1], focal_flt, np.float32)
    im_plane = np.concatenate([im_plane_X, im_plane_Y, im_plane_Z], 2)
    im_plane_norm2_inv = 1.0 / np.linalg.norm(im_plane, 2, 2)
    depth_helpers = {
        'h_orig': h_orig,
        'w_orig': w_orig,
        'focal_flt': focal_flt,
        'im_plane': im_plane,
        'im_plane_norm2_inv': im_plane_norm2_inv,
        'im_plane_norm2_inv_mul_focal': im_plane_norm2_inv * focal_flt
    }
    return depth_helpers['im_plane_norm2_inv_mul_focal']

    
################################################################################
                             # End #
################################################################################