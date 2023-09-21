import os
import json
import random
# If this has to be reran, we want the same image and scene division
random.seed(37)

HYPERSIM_ROOT = 'HYPESIM_ROOT_DIR'
OUTPUT_FILE_PATH = 'OUTPUT_JSON_PATH'

SCENE_BONDARIES_JSON = 'PATH_TO_BOUNDARY_JSON'
SCENE_SEMANTIC_METADAT_JSON = 'PATH_TO_SEMANTIC_METADATA_JSON'


if __name__ == '__main__':
    # Careful with overwriting, you dont want to overwrite random division
    # which serves as a train/set split later for all experiments
    # assert not os.path.isfile(OUTPUT_FILE_PATH)

    # Loadd all scene dirs
    scene_dirs = [x.path for x in os.scandir(HYPERSIM_ROOT) if os.path.isdir(x)]
    scene_dirs.sort()
    print(f'{len(scene_dirs)} scenes loaded.\n')

    # Load dict with all scene boundaries
    if os.path.isfile(SCENE_BONDARIES_JSON):
        with open(SCENE_BONDARIES_JSON, 'r') as f:
            scene_boudnaries = json.load(f)
    else:
        scene_boudnaries = None
    # Load dict with all scene semantic classes
    if os.path.isfile(SCENE_SEMANTIC_METADAT_JSON):
        with open(SCENE_SEMANTIC_METADAT_JSON, 'r') as f:
            scene_semantic_classes = json.load(f)
    else:
        scene_semantic_classes = None

    metadata_all_scenes = {}
    # Go through all loaded scenes
    for scene_dir in scene_dirs:
        # Extract scene and camera name
        scene_n = os.path.basename(scene_dir)
        
        # Create metedata structure
        metadata_all_scenes[scene_n] = {'cams': {}}
        if scene_boudnaries is not None:
            metadata_all_scenes[scene_n]['scene_boundary'] = scene_boudnaries[scene_n]
        if scene_semantic_classes is not None:
            metadata_all_scenes[scene_n]['semantic_metadata'] = scene_semantic_classes['scenes'][scene_n]

        rgb_cams_list = [x.name for x in os.scandir(os.path.join(scene_dir, 'images')) if 'final_hdf5' in x.name]
        rgb_cams_list.sort()
        for rgb_cam in rgb_cams_list:
            cam_n = '_'.join(rgb_cam.split('_')[1:3])

            # Load a list of all RGB images from the selected camera
            imgs_list = [x.name for x in os.scandir(os.path.join(scene_dir, 'images', rgb_cam))]
            n_img = len(imgs_list)
            #imgs_list.sort()
            # Randomly shuffle image list, so that different meaningfull 
            # data division ratios can be made at runtime
            random.shuffle(imgs_list)
            random.shuffle(imgs_list)
            random.shuffle(imgs_list)

            metadata_all_scenes[scene_n]['cams'][cam_n] = {'img_names': imgs_list}

    print(f'Image lists for different scenes are shuffled.')
    print(f'This way, meaningful different data division can be made during runtime. \n')

    # Save to a .json file
    with open(OUTPUT_FILE_PATH, 'w') as f:
        json.dump(metadata_all_scenes, f)
        print(f'Metadata of all scenes saved at: {OUTPUT_FILE_PATH}')