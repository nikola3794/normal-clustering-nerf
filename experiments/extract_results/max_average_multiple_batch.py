import os, sys
sys.path.append(os.getcwd())


import pandas as pd
import numpy as np

from experiments.extract_results.utils_results import (
    load_all_res_batch,
    keep_only_overpalling_scenes,
    reduce_multiple_batches,
    merge_averaged_batches,
    _split_df_consistent_NONconsistent
)


if __name__ == '__main__':    
    # WORK_SPECTA_DIR = '/home/nipopovic/MountedDirs/hl_task_prediction/hl_task_pred_root/data/nikola'    
    # BATCH_EXP_DIR = os.path.join(WORK_SPECTA_DIR, 'experiment_logs', 'euler_backup_17_10_22_', 'ngp_mt', 'batch_experiments')
    # RESULTS_ROOT = os.path.join(BATCH_EXP_DIR, '_extracted_results')

    WORK_SPECTA_DIR = '/home/nipopovic/MountedDirs/euler/work_specta'
    EXPERIMENT_LOGS = os.path.join(WORK_SPECTA_DIR, 'experiment_logs', 'ngp_mt')
    BATCH_EXP_DIR = os.path.join(EXPERIMENT_LOGS, 'batch_experiments')
    RESULTS_ROOT = os.path.join(EXPERIMENT_LOGS, 'batch_extracted_results')

    EXP_NAMES = [
        #
        '2023_02_26_ScanNet_3_samae_image_foig',
        #
        '2023_02_26_ScanNet_3_samae_image_CxvJ',
        #
        '2023_02_26_ScanNet_3_samae_image_BXKC',
        #
        '2023_02_26_ScanNet_3_samae_image_odel',
        #
        '2023_02_26_ScanNet_3_samae_image_drmd',
        #
        '2023_02_26_ScanNet_3_samae_image_Hlib',
        #
        '2023_02_26_ScanNet_3_samae_image_vbXr',
        #
        '2023_02_26_ScanNet_3_samae_image_wgJy',
        #
        '2023_02_26_ScanNet_3_samae_image_SYKO',
        #
        '2023_02_26_ScanNet_3_samae_image_kKAa',
    ]

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
    
    # Average multiple batches and pack in a dict of df's
    all_batch_mean_res = reduce_multiple_batches(all_batches_all_res, reduction='mean')
    # Compile averaged batches into df's (mean and hyperparameters)
    mean_metrics_df, hyperparam_consistent, _ = \
        merge_averaged_batches(all_batch_mean_res, reduction='mean', output_csv_dir=RESULTS_ROOT)

    # Comile dataframe that consists of the best performing scene 
    # from the  provided batch.
    scene_names = list(all_batches_all_res.values())[0].loc[:, 'param/scene_name'].values
    scene_maximums = {x: {'max': 0.0, 'df_column': None} for x in scene_names}
    for k in all_batches_all_res:
        df_k = all_batches_all_res[k]
        for s in scene_maximums:
            row_s = df_k.loc[df_k['param/scene_name'] == s]
            psnr_s = row_s['metric/rgb/psnr'].item()
            if psnr_s > scene_maximums[s]['max']:
                scene_maximums[s]['max'] = psnr_s
                scene_maximums[s]['df_column'] = row_s
    all_res_best_batch_scene = [scene_maximums[s]['df_column'] for s in scene_maximums]
    all_res_best_batch_scene = pd.concat(all_res_best_batch_scene)

    _, all_res_best_batch_scene = \
        _split_df_consistent_NONconsistent(all_res_best_batch_scene)
    
    # mean_psnr_best_batch_scene = [scene_maximums[s]['max'] for s in scene_maximums]
    # mean_psnr_best_batch_scene = sum(mean_psnr_best_batch_scene) / len(mean_psnr_best_batch_scene)
    # tmp = pd.DataFrame({'metric/rgb/psnr': [mean_psnr_best_batch_scene]})
    # print(mean_psnr_best_batch_scene)
    mean = all_res_best_batch_scene.mean(axis=0)
    mean_df = pd.DataFrame([mean.values.tolist()], columns=all_res_best_batch_scene.columns.values.tolist()[:mean.size])
    all_res_best_batch_scene = pd.concat([all_res_best_batch_scene, mean_df])
    print(mean)

    output_csv_dir = RESULTS_ROOT
    hyperparam_consistent_save_path = os.path.join(output_csv_dir, 'max_scenes_from_batch.csv')
    print(f'Saved...')
    print(f'{hyperparam_consistent_save_path}')
    all_res_best_batch_scene.to_csv(
        hyperparam_consistent_save_path, 
        float_format='%.4f',
        index=False)



