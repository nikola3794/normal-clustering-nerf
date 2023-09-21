import os
import json
import pandas as pd

HL_TASK_ROOT = '/home/nipopovic/MountedDirs/hl_task_prediction/hl_task_pred_root/data'
OUTPUT_ROOT = os.path.join(HL_TASK_ROOT, 'data_sets/Hypersim_targz_partitions')

RES_DIR = '/home/nipopovic/MountedDirs/euler/work_specta/experiment_logs/ngp_mt/batch_extracted_results/_extracted_detailed_info'
JUST_RGB_PATH = '2022_10_31_91_just_rgb_Edho_finished_all.csv'
RGB_CAN_PATH = '2022_10_31_91_rgb_can_fTOL_finished_all.csv'

def print_grp(rgb_df, rgb_can_df, scene_list, grp_name):
    rgb_A_df = rgb_df.loc[rgb_df['param/scene_name'].isin(scene_list)]
    rgb_can_A_df = rgb_can_df.loc[rgb_can_df['param/scene_name'].isin(scene_list)]
    print(f'Group {grp_name}: (rgb vs. rgb + canonical)')
    print(f'---------------------------------------------------------------')
    print(f'    len={len(scene_list)}')
    print(f'    psnr:       {rgb_A_df["metric/rgb/psnr"].mean():.3f}')
    print(f'    psnr:       {rgb_can_A_df["metric/rgb/psnr"].mean():.3f}')
    print(f'    psnr:       {(rgb_can_A_df["metric/rgb/psnr"]-rgb_A_df["metric/rgb/psnr"]).mean():.3f}')
    print(f'---------------------------------------------------------------')
    print(f'    ssim:       {rgb_A_df["metric/rbg/ssim"].mean():.3f}')
    print(f'    ssim:       {rgb_can_A_df["metric/rbg/ssim"].mean():.3f}')
    print(f'---------------------------------------------------------------')
    print(f'    depth_abs:  {rgb_A_df["metric/depth/abs"].mean():.3f}')
    print(f'    depth_abs:  {rgb_can_A_df["metric/depth/abs"].mean():.3f}')
    print(f'---------------------------------------------------------------')
    print(f'    depth_rmse: {rgb_A_df["metric/depth/rmse"].mean():.3f}')
    print(f'    depth_rmse: {rgb_can_A_df["metric/depth/rmse"].mean():.3f}')
    print(f'---------------------------------------------------------------')
    print(f'    norm_mean:  {rgb_A_df["metric/norm_depth/ang_err_mean"].mean():.3f}')
    print(f'    norm_mean:  {rgb_can_A_df["metric/norm_depth/ang_err_mean"].mean():.3f}')
    print(f'---------------------------------------------------------------')
    print(f'    norm_med:   {rgb_A_df["metric/norm_depth/ang_err_median"].mean():.3f}')
    print(f'    norm_med:   {rgb_can_A_df["metric/norm_depth/ang_err_median"].mean():.3f}')
    print(f'---------------------------------------------------------------')

rgb_df = pd.read_csv(os.path.join(RES_DIR, JUST_RGB_PATH))
rgb_can_df = pd.read_csv(os.path.join(RES_DIR, RGB_CAN_PATH))


LIST_A = [
    'ai_044_004',
    'ai_016_010',
    'ai_047_003', 
    'ai_018_001',
    'ai_004_002',
    'ai_037_005', 
    'ai_018_002',
    'ai_017_008', # bad bounds
    'ai_018_003', 
    'ai_018_005', # non manhattan roof
    'ai_002_001', # not good bounds
    'ai_048_004',
    'ai_003_009', 
    'ai_054_004',
    'ai_008_006', # bad bounds
    'ai_002_010',
    'ai_005_010',
    'ai_003_005',
    'ai_053_006',
    'ai_018_006', # in empty space
]
print_grp(rgb_df, rgb_can_df, LIST_A, 'A')


LIST_B = [
    'ai_019_002',
    'ai_008_004',
    'ai_041_005',
    'ai_029_005',
    'ai_017_006',
    'ai_016_002',
    'ai_018_010',
    'ai_044_008',
    'ai_042_002',
    'ai_022_005', 
    'ai_053_017', 
    'ai_054_002',
    'ai_037_007',
    'ai_022_002',
    #'ai_039_003', # smewhat wierd bounds
    'ai_002_005', 
    'ai_046_004',
    'ai_002_003', 
    'ai_052_006',
    'ai_043_010', 
    'ai_003_008',
]
print_grp(rgb_df, rgb_can_df, LIST_B, 'B')

LIST_C = [
    'ai_004_004',
    'ai_012_004', 
    'ai_008_002', 
    'ai_018_008',
    'ai_008_009',
    'ai_004_008',
    'ai_017_002',
    'ai_043_002',
    'ai_012_005',
    'ai_053_012',
]
print_grp(rgb_df, rgb_can_df, LIST_C, 'C')


path_a = os.path.join(OUTPUT_ROOT, 'hypersim_A_scenes.json')
with open(path_a, 'w') as f:
    json.dump(LIST_A,f)
path_b = os.path.join(OUTPUT_ROOT, 'hypersim_B_scenes.json')
with open(path_b, 'w') as f:
    json.dump(LIST_B,f)
path_c = os.path.join(OUTPUT_ROOT, 'hypersim_C_scenes.json')
with open(path_c, 'w') as f:
    json.dump(LIST_C,f)