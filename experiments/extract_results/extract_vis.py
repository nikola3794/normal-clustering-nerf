import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import shutil


# Hypersim - paper
# 'ai_018_003': ['cam_00.0003']
# 'ai_018_002': ['cam_00.0048']
# Hypersim - supplementary
# 'ai_004_002': ['cam_00.0068']
# 'ai_008_006': ['cam_00.0097'] 
# 'ai_018_001': ['cam_00.0048'] 
# 'ai_018_006': ['cam_00.0010'] 
# 'ai_037_005': ['cam_00.0065'] 
# 'ai_047_003': ['cam_00.0014'] 
# 'ai_054_004': ['cam_00.0021'] 


# Replica - paper
# 'room_0': ['834']
# Replica - supplementary
# 'room_1': ['342']
# 'room_2': ['78']
# 'room_0': ['630']
# 'office_2': ['102']
#

EXP_LIST = [
    # rgb
    '2022_11_10_A_exp_VIS_A_mOPP',
    # rgb + ManNerf
    '2022_11_10_A_exp_VIS_A_yrcN',
    # rgb + clustering
    '2022_11_10_A_exp_VIS_A_bvcE',
    '2022_11_10_A_exp_VIS_A_cOxi',
]
EXP_NAME = [
    'RGB',
    'ManNerf',
    'Ours',
    'Ours2'
]
DATASET = 'Hypersim'
# for i,_ in enumerate(EXP_LIST):
#     EXP_NAME[i] = f"{EXP_NAME[i]}_{EXP_LIST[i].split('_')[-1]}"


# EXP_LIST = [
#     # rgb
#     '2022_11_10_replica_sem_vis_RsXQ',
#     # Manhattan
#     # 5e-4
#     '2022_11_10_replica_sem_vis_VrVN', # TODO
#     # ours
#     '2022_11_10_replica_sem_vis_tqSl', # 5e-4
#     '2022_11_10_replica_sem_vis_our_search_ygbQ', #3e-4
# ]
# EXP_NAME = [
#     'RGB',
#     'ManNerf',
#     'Ours',
#     'Ours',
# ]
# DATASET = 'Replica_2'


WHICH_LABELS = [
    'rgb',
    'depth',
    'normals'
]
EXP_ROOT_DIR = '/home/nipopovic/MountedDirs/euler/work_specta/experiment_logs/ngp_mt'
EXP_ROOT_DIR = os.path.join(EXP_ROOT_DIR, 'cvpr23_results', 'visuals_A') #'visuals_A' 'visuals_replica_sem' 'supp_loss_w'
SAVE_DIR = os.path.join(EXP_ROOT_DIR, 'saved')


SCENE_N = 'ai_004_002'
SAVE_IDS = ['cam_00.0007','cam_00.0027','cam_00.0033','cam_00.0041']
PLOT_ALL = False


# ScanNet
# '0050_00': ['1030', '1050', '1090', '1150', '1950', '1970', '2010', '2030', '3310', '3330', '3430', '3710', '3730', '3870']
# '1050', '1150', '2010', '3710', '3870' # depth - 3710
# 1150 (main paper), 3710 (supp + depth, first), 2010 (supp, last)
# '0580_00': ['110', '130', '170', '4610', '1590', '1810', '1830', '4190', '4410', '4430']
# '110', '4190', '4430'
# 110 (supp, second last), 4430 (supp, second)
SAVE_IDS = ['110', '4430']
EXP_ROOT_DIR = '/home/nipopovic/MountedDirs/euler/work_specta/experiment_logs/ngp_mt'
EXP_ROOT_DIR = os.path.join(EXP_ROOT_DIR, 'cvpr23_results', 'visuals_scannet') #'visuals_A' 'visuals_replica_sem' 'supp_loss_w'
SAVE_DIR = os.path.join(EXP_ROOT_DIR, 'saved')
SCENE_N = '0580_00'
PLOT_ALL = False
DATASET = 'ScanNet'
EXP_LIST = [
    # rgb
    '2023_03_07_ScanNet_6_vis_alpc',
    # # rgb + ManNerf
    '2023_03_07_ScanNet_6_vis_LDID',
    # rgb + clustering
    #'2023_03_07_ScanNet_6_vis_jOtT',
    '2023_03_07_ScanNet_6_vis_ejdi',
]
EXP_NAME = [
    'RGB',
    'ManNerf',
    #'Ours_7e-3',
    'Ours',
]

################################################################################

if DATASET == "Hypersim":
    all_img_ids = os.listdir(os.path.join(EXP_ROOT_DIR, EXP_LIST[0], SCENE_N, 'results'))
    all_img_ids = ['_'.join(x.split('_')[:2]) for x in all_img_ids]
    all_img_ids = sorted(list(set(all_img_ids)))
elif DATASET == "Replica_2":
    all_img_ids = os.listdir(os.path.join(EXP_ROOT_DIR, EXP_LIST[0], SCENE_N, 'results'))
    all_img_ids = [x.split('_')[0] for x in all_img_ids]
    all_img_ids = sorted(list(set(all_img_ids)))
elif DATASET == "ScanNet":
    all_img_ids = os.listdir(os.path.join(EXP_ROOT_DIR, EXP_LIST[0], SCENE_N, 'results'))
    all_img_ids = [x.split('_')[0] for x in all_img_ids]
    all_img_ids = sorted(list(set(all_img_ids)))



# # Replica room_0 - 834
# h_min = 100
# h_max = 480
# w_min = 70
# w_max = 70 + 507
h_min = 0
h_max = -1
w_min = 0
w_max = -1
n_exp = len(EXP_LIST)
n_lab = len(WHICH_LABELS)
for img_id in all_img_ids:
    if not PLOT_ALL and img_id not in SAVE_IDS:
        continue
    fig, axs = plt.subplots(n_lab, n_exp+1,  figsize=(18, 6))
    for i, exp in enumerate(EXP_LIST):
        res_dir = os.path.join(EXP_ROOT_DIR, exp, SCENE_N, 'results')
        for j, lab in enumerate(WHICH_LABELS):
            lab_pred = lab
            lab_GT = f'{lab}_GT'
            if lab in ['normals', 'normals_depth']:
                lab_pred = 'norm_depth'
            pred_name = f'{img_id}_{lab_pred}.png'
            vis_pred_path = os.path.join(res_dir, pred_name)
            if not os.path.isfile(vis_pred_path):
                continue
            vis_pred = (cv2.imread(vis_pred_path)[:,:,::-1] / 255.0).astype(np.float32)
            vis_pred = vis_pred[h_min:h_max, w_min:w_max]
            if img_id in SAVE_IDS:
                save_path = os.path.join(SAVE_DIR, f'{EXP_NAME[i]}_{pred_name}')
                plt.imsave(save_path, vis_pred)
                #shutil.copy(vis_pred_path, save_path)
            axs[j][i].imshow(vis_pred)
            if j == 0:
                axs[j][i].set_title(EXP_NAME[i])
            if i == n_exp-1:# TODO
                gt_name = f'{img_id}_{lab_GT}.png'
                vis_GT_path = os.path.join(res_dir, gt_name)
                if not os.path.isfile(vis_GT_path):
                    continue
                vis_pred_GT = (cv2.imread(vis_GT_path)[:,:,::-1] / 255.0).astype(np.float32)
                vis_pred_GT = vis_pred_GT[h_min:h_max, w_min:w_max]
                axs[j][n_exp].imshow(vis_pred_GT)
                if img_id in SAVE_IDS:
                    save_path = os.path.join(SAVE_DIR, f'GT_{gt_name}')
                    plt.imsave(save_path, vis_pred_GT)
                    #shutil.copy(vis_GT_path, save_path)
    axs[0][-1].set_title(img_id)
    plt.show()
