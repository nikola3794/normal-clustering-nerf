'''
Taken and  modified from NERF--: 
https://github.com/ActiveVisionLab/nerfmm/blob/main/utils/vis_cam_traj.py
(They took it and modified from NeRF++: 
https://github.com/Kai-46/nerfplusplus)
'''

import os, sys
sys.path.append(os.getcwd())

from datasets.hypersim_src.utils import generate_pointcloud

import imageio

import numpy as np
import torch

import open3d as o3d


def connect_points(points, color):
    N = len(points)
    lines = np.zeros((N-1, 2))       # 8 lines per frustum
    color = np.repeat(np.array(color)[np.newaxis,:], N, axis=0)

    for i, _ in enumerate(points):
        if i == 0:
            continue
        lines[i-1, 0] = i-1
        lines[i-1, 1] = i

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(points)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector(color)

    return lineset


def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N*11, 3))      # 11 vertices per frustum
    merged_lines = np.zeros((N*17, 2))       # 17 lines per frustum
    merged_colors = np.zeros((N*17, 3))      # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i*11:(i+1)*11, :] = frustum_points
        merged_lines[i*17:(i+1)*17, :] = frustum_lines + i*11
        merged_colors[i*17:(i+1)*17, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset

def pointcloud2open3d(pointcloud_list):
    for i, (p_wc, rgb) in enumerate(pointcloud_list):
        if i == 0:
            merged_points = p_wc
            merged_rbg = rgb
        else:
            merged_points = np.concatenate((merged_points, p_wc), axis=0)
            merged_rbg = np.concatenate((merged_rbg, rgb), axis=0)

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(merged_points)
    pointcloud.colors = o3d.utility.Vector3dVector(merged_rbg)

    return pointcloud
    


def get_camera_frustum_opengl_coord(
    C2W, 
    ray_dirs_cc,
    scale_cam,
    pix_plane_color=[1., 0., 0.],
    ext_plane_color=[0., 0., 1.],
):
    # Camera internals
    w_min = ray_dirs_cc[:, 0].min(); w_max = ray_dirs_cc[:, 0].max()
    w_mid = (w_min + w_max) / 2.0
    h_min = ray_dirs_cc[:, 1].min(); h_max = ray_dirs_cc[:, 1].max()
    h_mid = (h_min + h_max) / 2.0
    z_neg = ray_dirs_cc[:, 2].min() # Negative

    center_points = np.array([[0., 0., 0., 1.0]]) 
    pix_plane_points = np.array([[w_min, h_max,  z_neg, 1.0],   # top-left image corner
                                [w_max, h_max,   z_neg, 1.0],   # top-right image corner
                                [w_max, h_min,  z_neg, 1.0],   # bottom-right image corner
                                [w_min, h_min, z_neg, 1.0]])  # bottom-left image corner
    ext_plane_points = np.copy(pix_plane_points)
    ext_plane_points[:, :3] *= 2.0
    ray_coef = 3.0
    ray_points = np.array([[w_mid, h_mid,  z_neg, 1.0],   
                            [ray_coef*w_mid, ray_coef*h_mid, ray_coef*z_neg, 1.0]])  
    # Frustum points (11, :)
    frustum_points = np.concatenate(
        (center_points, pix_plane_points, ext_plane_points, ray_points),
        axis=0
    )
    frustum_points[:,:3] *= scale_cam

    # Pixel plane (8, :)
    pix_plane_lines = [[0, i] for i in range(1, 5)] # Center to pix plane
    pix_plane_lines += [[i, (i+1)] for i in range(1, 4)] + [[4, 1]] # Pix plane
    pix_plane_colors = [pix_plane_color]*len(pix_plane_lines)

    # Extended plane (8, :)
    ext_plane_lines = [[i, i+4] for i in range(1, 5)] # Pix plane to extended plane
    ext_plane_lines += [[i, (i+1)] for i in range(5, 8)] + [[8, 5]] # Extended plane
    ext_plane_colors = [ext_plane_color]*len(ext_plane_lines)

    # Ray (1, :)
    ray_line = [[9, 10]]
    ray_color = [pix_plane_color]

    # (17, :)
    frustum_lines = np.array(pix_plane_lines + ext_plane_lines + ray_line) 
    frustum_colors = np.array(pix_plane_colors + ext_plane_colors + ray_color)

    # transform view frustum from camera space to world space
    frustum_points = np.matmul(C2W.numpy(), frustum_points.T).T  # (11, :)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]  # (11, :)  remove homogenous coordinate
    return frustum_points, frustum_lines, frustum_colors


def get_pointcloud_opengl_coord(ray_dirs_cc, depths, depth_type, C2W, RGBs, height_lim):
    N = ray_dirs_cc.shape[0]
    depths = torch.reshape(torch.clone(depths), (N, 1))
    RGBs = torch.reshape(torch.clone(RGBs), (N, 3))

    P_wc = generate_pointcloud(
        ray_dirs_cc, 
        C2W.unsqueeze(0), 
        depths.unsqueeze(0), 
        depth_type)
    P_wc = P_wc[0]

    idx_select = P_wc[:, 2] < height_lim
    P_wc = P_wc[idx_select]
    RGBs = RGBs[idx_select]

    return P_wc, RGBs


def draw_scene(
    poses,
    ray_dirs_cc,
    scale_cam,
    RGBs,
    depths,
    depth_type,
    scene_boundary,
    every_kth_frame=20, 
    save_path=None,
    draw_now=False,
    connect_cams=True,
    draw_pointcloud=True,
    coord='opengl',
    hight_lim=True,
    additional_geometries = None,
):
    xyz_scene_min = scene_boundary['xyz_scene_min'].clone()
    xyz_scene_max = scene_boundary['xyz_scene_max'].clone()
    xyz_cam_min = scene_boundary['xyz_cam_min'].clone()
    xyz_cam_max = scene_boundary['xyz_cam_max'].clone()
    if 'xyz_cam1p5_min' in scene_boundary:
        xyz_min = scene_boundary['xyz_cam1p5_min'].clone()
        xyz_max = scene_boundary['xyz_cam1p5_max'].clone()
    else:
        xyz_min = scene_boundary['xyz_scene_min'].clone()
        xyz_max = scene_boundary['xyz_scene_max'].clone()


    pix_plane_color=[0.0, 0.0, 0.0]
    ext_plane_color=[0.0, 0.0, 1.0]
    cam_traj_color=[0.0, 0.0, 0.0]

    C2Ws = poses
    frames = list(range(C2Ws.shape[0]))
    # TODO
    #frames = [37]
    #every_kth_frame = 1
    # TODO
    #factors = int(C2Ws.shape[0] / every_kth_frame)
    if hight_lim:
        hight_lim = xyz_scene_min[2] + (xyz_scene_max[2] - xyz_scene_min[2])*0.95
    else:
        hight_lim = xyz_scene_min[2] + (xyz_scene_max[2] - xyz_scene_min[2])*1.05

    frustum_list = []
    pointcloud_list = []
    if coord == 'opengl':
        for i in frames:
            frustum = get_camera_frustum_opengl_coord(
                C2Ws[i],
                ray_dirs_cc, 
                scale_cam,
                pix_plane_color=pix_plane_color,
                ext_plane_color=ext_plane_color
            )
            frustum_list.append(frustum)
            if draw_pointcloud and (i % every_kth_frame ==0):
                pointcloud = get_pointcloud_opengl_coord(
                    ray_dirs_cc,
                    depths[i],
                    depth_type,
                    C2Ws[i],
                    RGBs[i],
                    height_lim=hight_lim
                )
                pointcloud_list.append(pointcloud)
    else:
        raise NotImplementedError
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame()
    coord_scale = (xyz_cam_max - xyz_cam_min).max().cpu().item()/10.0
    draw_geometries = [coord.scale(coord_scale, center=coord.get_center())]

    frustums_geometry = frustums2lineset(frustum_list)
    draw_geometries.append(frustums_geometry)

    if draw_pointcloud:
        pointcloud_geometry = pointcloud2open3d(pointcloud_list)
        draw_geometries.append(pointcloud_geometry)
    
    if connect_cams:
        connect_cam_seq = connect_points(C2Ws[frames, :3, 3], cam_traj_color)
        draw_geometries.append(connect_cam_seq)

    if additional_geometries is not None:
        draw_geometries.append(additional_geometries)

    bbox_scene = o3d.geometry.AxisAlignedBoundingBox(np.array(xyz_scene_min), np.array(xyz_scene_max))
    bbox_scene.color = np.array([1., 0., 0.])
    draw_geometries.append(bbox_scene)
    bbox_cam = o3d.geometry.AxisAlignedBoundingBox(np.array(xyz_cam_min), np.array(xyz_cam_max))
    bbox_cam.color = np.array([0., 0., 1.])
    draw_geometries.append(bbox_cam)
    bbox = o3d.geometry.AxisAlignedBoundingBox(np.array(xyz_min), np.array(xyz_max))
    bbox.color = np.array([0., 1., 0.])
    draw_geometries.append(bbox)

    if save_path:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        for gmtry in draw_geometries:
            vis.add_geometry(gmtry)
            vis.update_geometry(gmtry)
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(True); 
        imageio.imsave(save_path, img)
        #vis.capture_screen_image('tmp2.jpg', do_render=False) # Doesnt work
        vis.destroy_window()

    if draw_now:
        o3d.visualization.draw_geometries(draw_geometries)

    return draw_geometries  # this is an o3d geometry object.

if __name__ == "__main__":

    from datasets.hypersim_src.scene import HypersimScene

    SCENE_DIR = 'PATH_TO_SCENE'
    SCENE_METADATA_PATH = 'PATH_TO_METADATA_JSON'

    which_labels = ['depth', 'normals']
    dataset = HypersimScene(
        scene_root_dir=SCENE_DIR,
        scene_metadata_path=SCENE_METADATA_PATH,
        downscale_factor=1.0,
        which_labels=which_labels,
        which_cams=['cam_00'],
        which_split='test',
    )
    
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
        # Z axis faces away from the cameras
        directions = - directions
        m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
        mt_m = np.transpose(m, [0, 2, 1]) @ m
        focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
        return focus_pt

    def generate_random_poses(poses, xyz_cam_min, xyz_cam_max, n_poses=10000, random_pose_focusptjitter = True, random_pose_radius = 1.0):
        dev = poses.device
        poses = poses[:, :3, :].clone().numpy()
        xyz_cam_min = xyz_cam_min.numpy()
        xyz_cam_max = xyz_cam_max.numpy()

        # positions = poses[:, :3, 3]
        # radii = np.percentile(np.abs(positions), 100, 0)
        # radii = np.concatenate([radii, [1.]])
        cam2world = poses_avg(poses)
        up = poses[:, :3, 1].mean(0)
        z_axis = focus_pt_fn(poses)
        random_poses = []
        for _ in range(n_poses):
            position = xyz_cam_min + (xyz_cam_max-xyz_cam_min) * (np.random.rand(3)*0.8 + 0.1)
            # t = radii * np.concatenate([
            #     2 * random_pose_radius * (np.random.rand(3) - 0.5), [1,]])
            #position = cam2world @ t

            if random_pose_focusptjitter:
                z_axis_i = z_axis + np.random.randn(*z_axis.shape) * 0.125
            else:
                z_axis_i = z_axis

            vec2 = normalize(-(z_axis - position))
            vec0 = normalize(np.cross(up, vec2))
            vec1 = normalize(np.cross(vec2, vec0))
            #vec0 = cam2world[:3, 0]; vec1 = cam2world[:3, 1]; vec2 = cam2world[:3, 2]; 
            rnd_pose_i = np.stack([vec0, vec1, vec2, position], axis=1)
            #rnd_pose_i = viewmatrix(z_axis_i, up, position, False)
            rnd_pose_i = np.concatenate([rnd_pose_i, np.array([[0., 0., 0., 1.]])], axis=0)
            random_poses.append(rnd_pose_i)
        random_poses = torch.as_tensor(np.asarray(random_poses)).to(dev).type(torch.float32)
        cam2world = torch.from_numpy(np.concatenate([cam2world, np.array([[0., 0., 0., 1.]])], axis=0)).to(dev)
        return random_poses, cam2world
    
    #np.random.seed(37)

    random_poses, avg_pose = generate_random_poses(dataset.cam_model.poses, 
                                                   dataset.scene_boundary['xyz_cam_min'],  
                                                   dataset.scene_boundary['xyz_cam_max'],  
                                                   10000)

    frustum_list = []
    for i in range(10):
        frustum = get_camera_frustum_opengl_coord(
            random_poses[i],
            dataset.cam_model.ray_dirs_cc, 
            6.0,
            pix_plane_color=[0.0, 0.0, 0.0],
            ext_plane_color=[1.0, 0.0, 0.0],
        )
        frustum_list.append(frustum)
    frustum = get_camera_frustum_opengl_coord(
        avg_pose,
        dataset.cam_model.ray_dirs_cc, 
        6.0,
        pix_plane_color=[0.0, 0.0, 0.0],
        ext_plane_color=[0.0, 1.0, 0.0],
    )
    frustum_list.append(frustum)
    frustums_geometry = frustums2lineset(frustum_list)

    draw_scene(
        poses=dataset.cam_model.poses,
        ray_dirs_cc=dataset.cam_model.ray_dirs_cc,
        scale_cam=3.0,
        RGBs=dataset.imgs,
        depths=dataset.labels['depth'],
        depth_type=dataset.depth_type,
        scene_boundary=dataset.scene_boundary,
        every_kth_frame=1,
        save_path=None, #os.path.join(SAVE_ROOT_DIR, f'{scene_n}.{cam_n}.jpg'),
        draw_now=True,
        additional_geometries=frustums_geometry
    )
