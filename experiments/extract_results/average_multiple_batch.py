import os, sys
sys.path.append(os.getcwd())


import pandas as pd
import numpy as np

from experiments.extract_results.utils_results import (
    load_all_res_batch,
    keep_only_overpalling_scenes,
    reduce_multiple_batches,
    merge_averaged_batches
)


if __name__ == '__main__':    
    WORK_SPECTA_DIR = '/home/nipopovic/MountedDirs/euler/work_specta'
    EXPERIMENT_LOGS = os.path.join(WORK_SPECTA_DIR, 'experiment_logs', 'ngp_mt')
    BATCH_EXP_DIR = os.path.join(EXPERIMENT_LOGS, 'batch_experiments')
    RESULTS_ROOT = os.path.join(EXPERIMENT_LOGS, 'batch_extracted_results')
    
    # # ############################  hypersim_ABC  ################################
    # BATCH_EXP_DIR = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'hypersim_ABC')
    # RESULTS_ROOT = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'hypersim_ABC')
    # # A
    # EXP_NAMES = [
    #     # rgb
    #     '2022_11_05_cvpr_table_1_A_eWnV',
    #     # RegNeRF
    #     '2023_05_30_iccv_rebut_3_A_vHxT',
    #     # rgb + mannhattan 
    #     '2022_11_07_tmp_ManNerf_search_A_GBeW', # 5e-4
    #     '2023_03_08_rerun_ABC_A_sRSK',  # 5e-4
    #     '2022_11_08_manhattan_search_A_iaik', # 1e-4
    #     # rgb + mannhattan + can
    #     '2022_11_07_tmp_ManNerf_search_A_KyvJ', # 5e-3
    #     '2022_11_08_manhattan_search_A_vMaI', # 1e-2 
    #     # rbg + clustering
    #     '2022_11_05_cvpr_table_1_A_iMoO',
    #     # rgb + clustering + centr. canonical
    #     '2022_11_09_A_ours_MF_repeat_A_brKj',
    #     # MonoSDF
    #     '2023_05_31_iccv_rebut_6Mono_A_yNUb', 
    # ]
    # # B
    # EXP_NAMES = [
    #     # rgb
    #     '2022_11_05_cvpr_table_1_B_ynCv',
    #     # rgb + mannhattan 
    #     '2022_11_07_tmp_ManNerf_search_B_epew', # 5e-4
    #     '2022_11_08_manhattan_search_B_pHKZ', # 1e-4
    #     # rgb + mannhattan + can
    #     '2022_11_07_tmp_ManNerf_search_B_wYxg',  #5e-3
    #     '2022_11_08_manhattan_search_B_oXBv',  # 1e-2
    #     # rgb + clustering
    #     '2022_11_05_cvpr_table_1_B_IeqF',
    #     # rgb + clustering + centr. canonical
    #     '2022_11_06_cvpr_table_1_B_ROal',  
    # ]
    # # C
    # EXP_NAMES = [
    #     # rgb
    #     '2022_11_05_cvpr_table_1_C_OPoI',
    #     # rgb + mannhattan 
    #     '2022_11_07_tmp_ManNerf_search_C_rpFr', #5e-4
    #     '2022_11_08_manhattan_search_C_Zpel', #1e-4
    #     # rgb + mannhattan + can
    #     '2022_11_07_tmp_ManNerf_search_C_RBqw', # 5e-3
    #     '2022_11_08_manhattan_search_C_OCel', # 1e-2
    #     # rgb + clustering
    #     '2022_11_05_cvpr_table_1_C_hIyi',
    #     # rgb + clustering + centr. canonical
    #     '2022_11_06_cvpr_table_1_C_twkm',
    #     # MonoSDF
    #     '2023_05_31_iccv_rebut_7Mono_C_eEUb', 
    # ]

    # # # ##########################  hypersim_all  ###############################
    # BATCH_EXP_DIR = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'hypersim_all')
    # RESULTS_ROOT = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'hypersim_all')
    # EXP_NAMES = [
    #     # rgb
    #     '2022_11_09_cvpr_all_peob', #435/435
    #     # ManhattanNerf
    #     #'2022_11_09_cvpr_all_BXbr', #195/435
    #     # rgb + clustering
    #     '2022_11_09_cvpr_all_GeOC', #434/435
    # ]

    # # ############################  ablation_A  ##################################
    # BATCH_EXP_DIR = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'ablation')
    # RESULTS_ROOT = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'ablation')
    # # A
    # EXP_NAMES = [
    #     # rgb
    #     '2022_11_05_cvpr_table_1_A_eWnV',
    #     # rgb + centrs.
    #     '2022_11_06_cvpr_ablation_1_A_saDH',
    #     # rgb + orthog.
    #     '2022_11_06_cvpr_ablation_1_A_TCJj',
    #     # rbg + (centrs. + orthog.)
    #     '2022_11_05_cvpr_table_1_A_iMoO',
    #     # rgb + (centrs. + orthog.) + centr. canonical
    #     '2022_11_09_A_ours_MF_repeat_A_brKj',
    #     # rgb + (centrs. + orthog.); no start
    #     '2022_11_06_cvpr_ablation_1_A_miSj',
    #     # rgb + (centrs. + orthog.); no grow
    #     '2022_11_06_cvpr_ablation_1_A_nKgf',
    # ]
    
    # ##########################  rotation_offset  ###############################
    # BATCH_EXP_DIR = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'rotation_offset')
    # RESULTS_ROOT = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'rotation_offset')
    # EXP_NAMES = [
    #     # yaw/pitch/roll by 0.01
    #     '2022_11_07_cvpr_offset_A_DfXh',
    #     # yaw/pitch/roll by 3.0
    #     '2022_11_07_cvpr_rot3_A_IfPI',
    #     # yaw/pitch/roll by 6.0
    #     '2022_11_07_cvpr_offset_A_CcLX',
    #     # yaw/pitch/roll by 9.0
    #     '2022_11_07_cvpr_offset_A_pYYc',
    #     # yaw/pitch/roll by 12.0
    #     '2022_11_07_cvpr_offset_A_hkKm',
    #     # yaw/pitch/roll by 15.0
    #     '2022_11_07_cvpr_offset_A_zIlH',
    #     # yaw/pitch/roll by 18.0
    #     '2022_11_07_cvpr_offset_A_IbfR',
    #     # yaw/pitch/roll by 21.0
    #     '2022_11_07_cvpr_offset_A_ESiT',
    #     # yaw/pitch/roll by 24.0
    #     '2022_11_07_cvpr_offset_A_LbxK',
    #     # yaw/pitch/roll by 27.0
    #     '2022_11_07_cvpr_offset_A_jdqG',
    #     # yaw/pitch/roll by 30.0
    #     '2022_11_07_cvpr_offset_A_fAkv',
    # ]

    # # ##########################  treshold_sensitivity  ###############################
    # BATCH_EXP_DIR = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'treshold_sensitivity')
    # RESULTS_ROOT = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'treshold_sensitivity')
    # EXP_NAMES = [
    #     '2022_11_08_tresh_ablation_search_A_QxZQ',
    #     '2022_11_08_tresh_ablation_search_A_SUqA',
    #     '2022_11_08_tresh_ablation_search_A_uEtW',
    #     '2022_11_05_cvpr_table_1_A_iMoO',
    #     '2022_11_08_tresh_ablation_search_A_yVyx',
    #     '2022_11_08_tresh_ablation_search_A_ZIvR',
    #     '2022_11_08_tresh_ablation_search_A_lXSb',
    # ]       

    # # # ##########################  visuals_A  ###############################
    # BATCH_EXP_DIR = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'visuals_A')
    # RESULTS_ROOT = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'visuals_A')
    # EXP_NAMES = [
    #     # rgb
    #     '2022_11_10_A_exp_VIS_A_mOPP',
    #     # rgb + ManNerf
    #     '2022_11_10_A_exp_VIS_A_yrcN',
    #     # rgb + ManNerf + can
    #     '2022_11_10_A_exp_VIS_A_KSTp',
    #     # rgb + clustering
    #     '2022_11_10_A_exp_VIS_A_bvcE',
    #     '2022_11_10_A_exp_VIS_A_cOxi',
    # ]    
    
    # ##########################  visuals_replica_sem  ###############################
    # BATCH_EXP_DIR = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'visuals_replica_sem')
    # RESULTS_ROOT = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'visuals_replica_sem')
    # EXP_NAMES = [
    #     # rgb
    #     '2022_11_10_replica_Sem_search_IMmn', # TODO
    #     # sem 
    #     # 4e-2
    #     '2022_11_10_replica_sem_vis_RtrH',
    #     # Manhattan
    #     # 5e-4
    #     '2022_11_10_replica_Sem_search_Fgpx', # TODO
    #     # ours
    #     '2022_11_10_replica_sem_vis_tqSl', # 5e-4
    #     '2022_11_10_replica_sem_vis_our_search_BJnO', #2e-4 (0.01)
    #     '2022_11_10_replica_sem_vis_our_search_HwnG', #2e-4 (0.06)
    #     '2022_11_10_replica_sem_vis_our_search_eGMv', #2e-4 (0.125)
    #     '2022_11_10_replica_sem_vis_our_search_zKhF', #1e-4
    #     '2022_11_10_replica_sem_vis_our_search_ygbQ', #3e-4
    # ]

    # # # ##########################  replica_sem  ###############################
    # BATCH_EXP_DIR = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'replica_sem')
    # RESULTS_ROOT = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'replica_sem')
    # EXP_NAMES = [
    #     # rgb
    #     '2022_11_10_replica_Sem_search_IMmn',

    #     # rgb + clustering
    #     #5e-4
    #     '2022_11_10_replica_Sem_search_rsoh',

    #     # Manhattan Nerf
    #     # 5e-4
    #     '2022_11_10_replica_Sem_search_Fgpx',
    # ]
 
    #####################################################################################
    #######################################################################################    

    # ##########################  supp_loss_w  ###############################
    # BATCH_EXP_DIR = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'supp_loss_w')
    # RESULTS_ROOT = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'supp_loss_w')
    # EXP_NAMES = [
    #     #'2022_11_14_supp_diff_w_A_grqM', #w=1e-5
    #     '2022_11_14_supp_diff_w_A_GOUB', #w=2.5e-5
    #     '2022_11_14_supp_diff_w_A_NSxq', #w=5e-5
    #     '2022_11_14_supp_diff_w_A_aBZI', #w=7.5e-5
    #     '2022_11_14_supp_diff_w_A_IRzf', #w=1e-4
    #     '2022_11_14_supp_diff_w_A_ZvEm', #w=2.5e-4
    #     '2022_11_14_supp_diff_w_A_edxh', #w=5e-4
    #     '2022_11_14_supp_diff_w_A_vhqZ', #w=7.5e-4
    #     '2022_11_14_supp_diff_w_A_SIJY', #w=1e-3
    #     '2022_11_14_supp_diff_w_A_fkUj', #w=2.5e-3
    #     '2022_11_14_supp_diff_w_A_MmoV', #w=5e-3
    #     '2022_11_14_supp_diff_w_A_ySme', #w=7.5e-3
    #     '2022_11_14_supp_diff_w_A_ZXsr', #w=1e-2
    #     '2022_11_14_supp_diff_w_A_foGh', #w=2.5e-2
    #     '2022_11_14_supp_diff_w_A_kynJ', #w=5e-2
    #     '2022_11_15_supp_loss_w_A_hkiD', #w=7.5e-2
    #     '2022_11_14_supp_diff_w_A_iRBQ', #w=1e-1
    # ]

    # ##########################  supp_triang  ###############################
    # BATCH_EXP_DIR = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'supp_triang')
    # RESULTS_ROOT = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'supp_triang')
    # EXP_NAMES = [
    #     '2022_11_05_cvpr_table_1_A_iMoO', #t=0
    #     '2022_11_14_supp_triang_A_WyGX', #t=1
    #     '2022_11_14_supp_triang_A_ksyL', #t=2
    #     '2022_11_14_supp_triang_A_nTNo', #t=3
    #     '2022_11_15_supp_triang_size_A_fIeH', #t=4
    #     '2022_11_14_supp_triang_A_HJgc', #t=5
    #     '2022_11_14_supp_triang_A_KofW', #t=6
    #     '2022_11_14_supp_triang_A_DcHP', #t=7
    #     '2022_11_14_supp_triang_A_mClN', #t=8
    #     '2022_11_14_supp_triang_A_sfjN', #t=9
    #     '2022_11_14_supp_triang_A_vkjb', #t=10
    # ]

    # # ##########################  supp_tresh_clust  ###############################
    # BATCH_EXP_DIR = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'supp_tresh_clust')
    # RESULTS_ROOT = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'supp_tresh_clust')
    # EXP_NAMES = [
    #     '2022_11_14_supp_tresh_clust_A_KVzK', #t=0.0025
    #     '2022_11_14_supp_tresh_clust_A_EWMp', #t=0.005
    #     '2022_11_14_supp_tresh_clust_A_PQDJ', #t=0.0075
    #     '2022_11_14_supp_tresh_clust_A_Oqzf', #t=0.01
    #     '2022_11_14_supp_tresh_clust_A_sJky', #t=0.02
    #     '2022_11_14_supp_tresh_clust_A_kCtI', #t=0.03
    #     '2022_11_14_supp_tresh_clust_A_zNqO', #t=0.05
    #     '2022_11_14_supp_tresh_clust_A_Zqpv', #t=0.075
    #     '2022_11_14_supp_tresh_clust_A_ueNN', #t=0.1
    #     '2022_11_14_supp_tresh_clust_A_djse', #t=0.125
    #     '2022_11_14_supp_tresh_clust_A_UIqr', #t=0.15
    #     '2022_11_14_supp_tresh_clust_A_LYhU', #t=0.2
    #     '2022_11_14_supp_tresh_clust_A_azSq', #t=0.25
    #     '2022_11_14_supp_tresh_clust_A_wrRL', #t=0.3
    #     '2022_11_15_supp_clust_tresh_A_IIUK', #t=0.4
    #     '2022_11_15_supp_clust_tresh_A_DOqI', #t=0.5
    # ]

    # # ##########################  supp_ray_sampling  ###############################
    # BATCH_EXP_DIR = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'supp_ray_sampling')
    # RESULTS_ROOT = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'supp_ray_sampling')
    # EXP_NAMES = [
    #     '2022_11_14_supp_ray_sampling_A_tgzF', #same_image
    #     '2022_11_05_cvpr_table_1_A_eWnV',      #same_image
    #     #'2022_11_14_supp_ray_sampling_A_ceJo', #all_images
    #     '2022_11_14_supp_ray_sampling_A_KnTQ', #all_images_triang
    # ]

    #####################################################################################
    #######################################################################################  

    # # # ##########################  scannet  ###############################
    # BATCH_EXP_DIR = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'scannet')
    # RESULTS_ROOT = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'scannet')
    # EXP_NAMES = [
    #     # lr=1e-2, NO-triang
    #     '2023_01_28_ScanNet_1_just_rgb_wOCd',
    #     # RegNeRF
    #     '2023_05_30_iccv_rebuttal_scannet_3_lVnV',
    #     # lr=1e-2, man_w=5e-3 
    #     '2023_01_29_ScanNet_2_mansdf_EuCB', 
    #     # lr=1e-2, clust=1e-2
    #     '2023_01_27_ScanNet_1_just_rgb_eTSv',
    #     ##########################################
    #     # lr=1e-2, d_w=1e-1, NO-triang
    #     '2023_01_28_ScanNet_1_just_rgb_zaTZ',
    #     # lr=1e-2, d_w=1e-1, man_w=1e-2 
    #     '2023_01_29_ScanNet_2_mansdf_UphM', 
    #     # lr=1e-2, d_w=1e-1, clust=1e-2
    #     '2023_01_27_ScanNet_1_just_rgb_HqVa',
    # ]
    
    # # # ##########################  scannet_detailed  ###############################
    # BATCH_EXP_DIR = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'scannet_detailed')
    # RESULTS_ROOT = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'scannet_detailed')
    # EXP_NAMES = [
    #     # depth_w=0, ours_w=0
    #     '2023_03_07_ScanNet_7_hyperparameter_kyly',
    #     '2023_01_28_ScanNet_1_just_rgb_wOCd',
    #     # depth_w=0, man_w=5e-3 
    #     '2023_01_29_ScanNet_2_mansdf_EuCB', 
    #     # depth_w=0, ours_w=7e-4
    #     '2023_03_07_ScanNet_7_hyperparameter_YtDw',
    #     # depth_w=0, ours_w=5e-4
    #     '2023_03_07_ScanNet_7_hyperparameter_hUaX',
    #     # depth_w=0, ours_w=1e-3
    #     '2023_03_07_ScanNet_7_hyperparameter_edMJ',
    #     # depth_w=0, ours_w=2e-3
    #     '2023_03_07_ScanNet_7_hyperparameter_ObAn',
    #     # depth_w=0, ours_w=5e-3
    #     '2023_03_07_ScanNet_7_hyperparameter_nUkL',
    #     # depth_w=0, ours_w=7e-3
    #     '2023_03_07_ScanNet_7_hyperparameter_fnYI',
    #     # depth_w=0, ours_w=1e-2
    #     '2023_03_07_ScanNet_7_hyperparameter_lLpF',
    #     '2023_01_27_ScanNet_1_just_rgb_eTSv',

    #     # depth_w=0.1, ours_w=0
    #     '2023_03_07_ScanNet_7_hyperparameter_wblv',
    #     '2023_01_28_ScanNet_1_just_rgb_zaTZ',
    #     # depth_w=0.1, man_w=2e-3
    #     '2023_03_08_ScanNet_7_hyperparameters_PJQJ',
    #     # depth_w=0.1, man_w=7e-3
    #     '2023_03_08_ScanNet_7_hyperparameters_qJSO',
    #     # depth_w=0.1, man_w=1e-2  
    #     '2023_01_29_ScanNet_2_mansdf_UphM', 
    #     # depth_w=0.1, ours_w=5e-3
    #     '2023_03_07_ScanNet_7_hyperparameter_WntK',
    #     # depth_w=0.1, ours_w=1e-2
    #     '2023_03_07_ScanNet_7_hyperparameter_XKKA',
    #     '2023_01_27_ScanNet_1_just_rgb_HqVa',
    #     # depth_w=0.1, ours_w=2e-2
    #     '2023_03_07_ScanNet_7_hyperparameter_ggSM',
    #     # depth_w=0.1, ours_w=5e-2
    #     '2023_03_07_ScanNet_7_hyperparameter_LsHW',

    #     # depth_w=0.5, ours_w=0
    #     '2023_03_07_ScanNet_7_hyperparameter_PcYi',
    #     # depth_w=0.5, man_w=7e-3
    #     '2023_03_08_ScanNet_7_hyperparameters_NBfT',
    #     # depth_w=0.5, man_w=5e-3
    #     '2023_03_08_ScanNet_7_hyperparameters_QWhG',
    #     # depth_w=0.5, man_w=1e-2
    #     '2023_03_08_ScanNet_7_hyperparameters_OdTM',
    #     # depth_w=0.5, ours_w=5e-3
    #     '2023_03_07_ScanNet_7_hyperparameter_jnbC',
    #     # depth_w=0.5, ours_w=1e-2
    #     '2023_03_07_ScanNet_7_hyperparameter_uixQ',
    #     # depth_w=0.5, ours_w=2e-2
    #     '2023_03_07_ScanNet_7_hyperparameter_FDry',
    #     # depth_w=0.5, ours_w=5e-2
    #     '2023_03_07_ScanNet_7_hyperparameter_OZbF',

    #     # depth_w=1, ours_w=0
    #     '2023_03_07_ScanNet_7_hyperparameter_SMUP',
    #     # depth_w=1, man_w=5e-3
    #     '2023_03_08_ScanNet_7_hyperparameters_DMLO',
    #     # depth_w=1, man_w=7e-3
    #     '2023_03_08_ScanNet_7_hyperparameters_KWCY',
    #     # depth_w=1, man_w=2e-2
    #     '2023_03_08_ScanNet_7_hyperparameters_Srqp',
    #     # depth_w=1, ours_w=5e-3
    #     '2023_03_07_ScanNet_7_hyperparameter_oOgs',
    #     # depth_w=1, ours_w=1e-2
    #     '2023_03_07_ScanNet_7_hyperparameter_sbdT',
    #     # depth_w=1, ours_w=2e-2
    #     '2023_03_07_ScanNet_7_hyperparameter_jyyl',
    #     # depth_w=1, ours_w=2e-2
    #     '2023_03_07_ScanNet_7_hyperparameter_CRCI',
    #     # depth_w=1, ours_w=5e-2
    #     '2023_03_07_ScanNet_7_hyperparameter_zZQQ',
    # ]

    # ##########################  hypersim_A_sparsity  ###############################
    BATCH_EXP_DIR = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'hypersim_A_sparsity')
    RESULTS_ROOT = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'hypersim_A_sparsity')
    EXP_NAMES = [
        # N_sparse=-1
        '2023_01_29_rebuttal_Sparse_A_A_yUHt',
        #N_sparse=-1, clust_w=2e-3  
        '2023_01_29_rebuttal_Sparse_A_A_eMxM',
        
        # N_sparse=3
        '2023_01_29_rebuttal_Sparse_A_A_HfHB',
        # N_sparse=3, clust_w=2e-3
        '2023_01_29_rebuttal_Sparse_A_A_IMRv',

        # N_sparse=6
        '2023_01_29_rebuttal_Sparse_A_A_CBAv',
        # N_sparse=6 ,man_w=1e-2
        '2023_01_31_rebuttal_Sparse_A_A_LDhC',
        # N_sparse=6, clust_w=2e-3
        '2023_01_29_rebuttal_Sparse_A_A_KRZv',

        # N_sparse=9
        '2023_01_29_rebuttal_Sparse_A_A_scRJ',
        # N_sparse=9 ,man_w=1e-2
        '2023_01_31_rebuttal_Sparse_A_A_RTWa',
        # N_sparse=9, clust_w=2e-3
        '2023_01_29_rebuttal_Sparse_A_A_ODyb',

        # N_sparse=12
        '2023_01_29_rebuttal_Sparse_A_A_NiyA',
        # N_sparse=12 ,man_w=1e-2
        '2023_01_31_rebuttal_Sparse_A_A_OtBK',
        # N_sparse=12, clust_w=2e-3
        '2023_01_29_rebuttal_Sparse_A_A_QJtx',
    ]  


    # # ##########################  visuals_scannet  ###############################
    # BATCH_EXP_DIR = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'visuals_scannet')
    # RESULTS_ROOT = os.path.join(EXPERIMENT_LOGS, 'cvpr23_results', 'visuals_scannet')
    # EXP_NAMES = [
    #     # Baseline
    #     '2023_03_07_ScanNet_6_vis_alpc',
    #     # ManDF
    #     '2023_03_07_ScanNet_6_vis_LDID',
    #     # Ours
    #     '2023_03_07_ScanNet_6_vis_jOtT',
    #     # Ours
    #     '2023_03_07_ScanNet_6_vis_ejdi',
    # ]
    
    #####################################################################################
    #######################################################################################  


    print(f'\n')
    
    # Collect all results from each batch experiment
    all_batches_all_res = {}
    for exp_name_i in EXP_NAMES:
        batch_exp_root = os.path.join(BATCH_EXP_DIR, exp_name_i)
        res_all_i, failed_all_i = load_all_res_batch(
                                                batch_exp_root=batch_exp_root, 
                                                output_csv_dir=RESULTS_ROOT,
                                                attempt_csv_load=False)
        all_batches_all_res[exp_name_i] = res_all_i

    # For each batch, keep only scenes that appear in every batch
    all_batches_all_res, keep_scenes = keep_only_overpalling_scenes(all_batches_all_res)

    # TODO Further filter with respect to a list of scenes
    # TODO Think about how to handle if scenes from the scene list are missing
    
    # Mean every batch and pack in a dict of df's
    reduction = 'mean'
    all_batch_mean_res = reduce_multiple_batches(all_batches_all_res, reduction=reduction)
    # Compile meaned batches into df's (mean and hyperparameters)
    mean_metrics_df, hyperparam_consistent, _ = \
        merge_averaged_batches(all_batch_mean_res, reduction=reduction, output_csv_dir=RESULTS_ROOT)

    # Mean every batch and pack in a dict of df's
    reduction = 'median'
    all_batch_mean_res = reduce_multiple_batches(all_batches_all_res, reduction=reduction)
    # Compile meaned batches into df's (mean and hyperparameters)
    mean_metrics_df, hyperparam_consistent, _ = \
        merge_averaged_batches(all_batch_mean_res, reduction=reduction, output_csv_dir=RESULTS_ROOT)

