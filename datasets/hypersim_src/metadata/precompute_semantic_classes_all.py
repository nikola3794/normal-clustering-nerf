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

    semantic_info = {'general': {}, 'scenes': {}}
    # Go through all loaded scenes
    for scene_dir_i in tqdm(scene_dirs):
        # Extract scene and camera name
        scene_n = os.path.basename(scene_dir_i)
        rgb_cams_list = [x.name for x in os.scandir(os.path.join(scene_dir_i, 'images')) if 'final_hdf5' in x.name]
        rgb_cams_list.sort()
        # TODO Hardcoded
        # Always pick the first available camera 
        # (should be cam_00 or cam_01)
        rgb_cam = rgb_cams_list[0]
        cam_n = '_'.join(rgb_cam.split('_')[1:3])

        which_labels = ['semantics']
        dataset = HypersimScene(
            scene_root_dir=os.path.join(HYPERSIM_ROOT, scene_n),
            scene_metadata_path=None,
            downscale_factor=1.0,
            which_labels=which_labels,
            which_cams=[cam_n],
            which_split='all',
            split_factor=1.0,
        )

        semantic_metadata_i = dataset.label_metadata['semantics']
        semantic_info['scenes'][scene_n]={
            'class_ids_scene': semantic_metadata_i['class_ids_scene'].tolist(),
            'n_classes_scene': semantic_metadata_i['n_classes_scene'],
            'n_valid_classes_scene': semantic_metadata_i['n_valid_classes_scene'],
            'id_to_colour_scene': semantic_metadata_i['id_to_colour_scene'].tolist()} 

        # TODO: Quick dumb solution, rewrite every time
        semantic_info['general'] = {
            'class_names_nyu40': semantic_metadata_i['class_names_nyu40'],
            'id_to_colour_nyu40': semantic_metadata_i['id_to_colour_nyu40'].tolist(),
            'n_classes_nyu40': semantic_metadata_i['n_classes_nyu40']}    

        # Rewrite json with computed semantic classes
        with open(OUTPUT_FILE_PATH, 'w') as f:
            json.dump(semantic_info, f)

    print('Finished')


