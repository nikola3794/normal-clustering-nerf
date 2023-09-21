import os
import json
import random
# If this has to be reran, we want the same image and scene division
random.seed(37)

import pandas as pd

HYPERSIM_ROOT = 'HYPESIM_ROOT_DIR'
OUTPUT_FILE_PATH = 'OUTPUT_JSON_PATH'

# Filtering criteria
SCENE_BLACKLIST = [
    'ai_003_001', # Only black
]
MIN_IMGS_PER_CAM = 90
N_BEST_PSNR = 150
EXP_RES_PATH = 'EXPERIMENT_CSV_PATH'
N_TRAIN_SCENES = 100


if __name__ == '__main__':
    # Careful with overwriting, you dont want to overwrite random division
    # which serves as a train/set split later for all experiments
    assert not os.path.isfile(OUTPUT_FILE_PATH)

    # Load all scene metadata
    with open(os.path.join(HYPERSIM_ROOT, 'all_scenes_metadata.json'), 'r') as f:
        all_scene_metadata = json.load(f)

    # Load provided experiment results
    exp_res_df = pd.read_csv(EXP_RES_PATH)

    selected_scenes = []
    psnr_dict = {'scene_name': [], 'psnr': []}
    # Go through all loaded scenes
    for scene_n in all_scene_metadata:
        scene_selected = True
        scene_metadata = all_scene_metadata[scene_n]

        # Extract scene and camera name
        cams_list = list(scene_metadata['cams'].keys()).copy()
        cams_list.sort()
        # TODO Hardcoded
        # Always pick the first available camera 
        # (should be cam_00 or cam_01)
        cam_n = cams_list[0]

        # Skip blacklisted scenes
        if scene_n in SCENE_BLACKLIST:
            scene_selected = False

        # Load a list of all RGB images from the selected camera
        imgs_list = scene_metadata['cams'][cam_n]['img_names']
        n_img = len(imgs_list)
        
        # Skip all scenes which have a low number of images for the seleted camera
        if n_img < MIN_IMGS_PER_CAM:
            scene_selected = False
        
        # Skip scenes where xyz_cam1p5 boudns counld not be computed
        if 'xyz_cam1p5_min' not in scene_metadata['scene_boundary']:
            scene_selected = False
        elif 'xyz_cam1p5_max' not in scene_metadata['scene_boundary']:
            scene_selected = False

        # TODO : More selection filters?
        
        # Extract PSNR from the current scens experiment
        curr_exp = exp_res_df.loc[exp_res_df['param/scene_name'] == scene_n]
        if curr_exp.shape[0] == 0:        
            # If there is no experiment for this scene, 
            # it probably crashed so give bad psnr
            curr_psnr = -1.0
        elif curr_exp.shape[0] == 1:
            # Extract SPNR
            curr_psnr = curr_exp['metric/rgb/psnr'].values.item()
        else:
            # Make sure only one experiment is recorded for this scene
            raise AssertionError

        if scene_selected:
            psnr_dict['scene_name'].append(scene_n)
            psnr_dict['psnr'].append(curr_psnr)

            selected_scenes.append(scene_n)

    print(f'{len(selected_scenes)} scenes remaining after following criteria:')
    print(f'Blacklist: {SCENE_BLACKLIST}')
    print(f'Minimum {MIN_IMGS_PER_CAM} images per camera.')
    print(f'Required to have xyz_cam1p5 boudns.\n')

    # Extract N best scenes with respect to provided experiment PSNR
    psnr_df = pd.DataFrame.from_dict(psnr_dict)
    psnr_df = psnr_df.sort_values('psnr', ascending=False)
    psnr_df = psnr_df.head(N_BEST_PSNR)
    worst_psnr = psnr_df['psnr'].values[-1]
    scenes_good_psnr = psnr_df['scene_name'].values.tolist()
    scenes_good_psnr.sort()
    selected_scenes = [x for x in selected_scenes if x in scenes_good_psnr]
    print(f'{len(selected_scenes)} scenes remaining after following criteria:')
    print(f'N scenes with best experiment PSNR (worst psnr: {worst_psnr})\n')
    
    # Make a division of scenes into train and val set
    all_selected_scenes = selected_scenes.copy()
    random.shuffle(all_selected_scenes)
    random.shuffle(all_selected_scenes)
    random.shuffle(all_selected_scenes)
    selected_scenes_train = all_selected_scenes[:N_TRAIN_SCENES]
    selected_scenes_train.sort()
    selected_scenes_val = all_selected_scenes[N_TRAIN_SCENES:]
    selected_scenes_val.sort()
    print(f'Made a scene division:')
    print(f'{len(selected_scenes_train)} training scenes')
    print(f'{len(selected_scenes_val)} validation scenes')


    selected_scenes_info = {
        'selected_scenes': selected_scenes,
        'train_split': selected_scenes_train,
        'val_split': selected_scenes_val,
    }

    # Save to a .json file
    with open(OUTPUT_FILE_PATH, 'w') as f:
        json.dump(selected_scenes_info, f)
        print(f'Information about selected scenes saved at: {OUTPUT_FILE_PATH}')