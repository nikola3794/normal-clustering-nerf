import os, sys
import json
from tqdm import tqdm

sys.path.append(os.getcwd())
from datasets.hypersim_src.scene import HypersimScene

HYPERSIM_ROOT = 'HYPESIM_ROOT_DIR'
OUTPUT_FILE_PATH = 'OUTPUT_JSON_PATH'


if __name__ == '__main__':
    # Careful with overwriting, it takes long to compute
    assert not os.path.isfile(OUTPUT_FILE_PATH)

    # Loadd all scene dirs
    scene_dirs = [x.path for x in os.scandir(HYPERSIM_ROOT) if os.path.isdir(x)]
    scene_dirs.sort()

    print(f'{len(scene_dirs)} scenes loaded.\n')
    
    scene_boundaries = {}
    # Go through all loaded scenes
    for scene_dir in tqdm(scene_dirs):

        scene_n = os.path.basename(scene_dir)
        rgb_cams_list = [x.name for x in os.scandir(os.path.join(scene_dir, 'images')) if 'final_hdf5' in x.name]
        rgb_cams_list.sort()
        # TODO Hardcoded
        # Always pick the first available camera 
        # (should be cam_00 or cam_01)
        rgb_cam = rgb_cams_list[0]
        cam_n = '_'.join(rgb_cam.split('_')[1:3])

        scene_path = os.path.join(HYPERSIM_ROOT, scene_n)
        dataset = HypersimScene(
            scene_root_dir=scene_path,
            scene_metadata_path=None,
            downscale_factor=1.0,
            which_labels=['depth'],
            which_cams=[cam_n],
            which_split='all',
            split_factor=1.0,
        )
        scene_boundary = dataset.scene_boundary
        for k in scene_boundary:
            scene_boundary[k] = scene_boundary[k].cpu().tolist()
        scene_boundaries[scene_n] = scene_boundary

        # Rewrite json with computed scene boundries
        with open(OUTPUT_FILE_PATH, 'w') as f:
            json.dump(scene_boundaries, f)

    print('Finished')