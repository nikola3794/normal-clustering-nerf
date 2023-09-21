import json
import os
import sys
from tqdm import tqdm

sys.path.append(os.getcwd())
from datasets.hypersim_src.scene import HypersimScene
from datasets.hypersim_src.draw_scene import draw_scene

HYPERSIM_ROOT = 'HYPESIM_ROOT_DIR'
SAVE_ROOT_DIR = 'OUTPUT_DIR'
SCENE_METADATA_PATH = 'SCENE_METADATA_PATH'


if __name__ == '__main__':
    with open(SCENE_METADATA_PATH, 'r') as f:
        scene_metadata = json.load(f)
    for scene_n in tqdm(scene_metadata['scenes']):
        # # TODO : Temporary quickfix, remvoe later to be able to recompute
        # if 'scene_boundary' in scene_metadata['scenes'][scene_n]:
        #     print("Already recorded, skip")
        #     continue

        cam_list = list(scene_metadata['scenes'][scene_n]['cams'].keys())
        # assert len(cam_list) == 1
        cam_list.sort()
        # TODO Hardcoded
        # Always pick the first available camera 
        # (should be cam_00 or cam_01)
        cam_n = cam_list[0]

        # TODO skip already recorded
        save_path = os.path.join(SAVE_ROOT_DIR, f'{scene_n}.{cam_n}.jpg')
        if os.path.isfile(save_path):
            print('Already saved')
            continue

        scene_path = os.path.join(HYPERSIM_ROOT, scene_n)
        which_labels = ['semantics' ,'depth', 'normals']
        dataset = HypersimScene(
            scene_root_dir=scene_path,
            scene_metadata_path=SCENE_METADATA_PATH,
            downscale_factor=1.0,
            which_labels=which_labels,
            which_cams=[cam_n],
            which_split='all',
            split_factor=1.0
        )
        scene_boundary = dataset.scene_boundary

        draw_scene(
            poses=dataset.cam_model.poses,
            ray_dirs_cc=dataset.cam_model.ray_dirs_cc,
            extend_scale=5.0,
            ray_len=10.0,
            RGBs=dataset.imgs,
            depths=dataset.labels['depth'],
            depth_type=dataset.depth_type,
            scene_boundary=dataset.scene_boundary,
            every_kth_frame=1,
            save_path=save_path,
            draw_now=False
        )