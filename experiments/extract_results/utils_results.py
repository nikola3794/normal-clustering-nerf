import os
from itertools import chain
import pandas as pd
import numpy as np


BLACKLISTED_SCENES = [
    'ai_003_001', # Scene is completely black
    'ai_026_008', # Just some far away ceeling, call cameras looking up
    # Replica_sem blacklist
    'office_1',
    'office_4',
    'office_0',
    # ScanNet_manhattansdf
    '0084_00',
]
SCENE_RES_FNAME = 'results.csv'


def load_all_res_batch(batch_exp_root, output_csv_dir=None, attempt_csv_load=False):
    def _print_final_stats(res_all, failed_all):
        print(f'Number of successfull scenes: {res_all.shape[0]}')
        print(f'Number of failed scenes: {failed_all.shape[0]}')
        print(f'\n')

    assert batch_exp_root[-1] != os.sep

    exp_dir_name = os.path.basename(batch_exp_root)
    print(f'Compiling all results from : {exp_dir_name}')
    
    if output_csv_dir is not None:
        # Attempt to load if the results were compiled already
        assert os.path.isdir(output_csv_dir)
        res_all_path = os.path.join(output_csv_dir, '_extracted_detailed_info', exp_dir_name+'_finished_all.csv')
        failed_all_path = os.path.join(output_csv_dir, '_extracted_detailed_info', exp_dir_name+'_failed_all.csv')
        if os.path.isfile(res_all_path) and os.path.isfile(failed_all_path) and (attempt_csv_load is True):
            print(f'Already found all batch results and loading in a .csv file.')
            print(f'({res_all_path})')
            res_all = pd.read_csv(res_all_path)
            failed_all = pd.read_csv(failed_all_path)
            _print_final_stats(res_all, failed_all)
            return res_all, failed_all

    res_all = None
    # Iterate through all experiments
    already_loaded_scenes = []
    failed_exp = []
    for exp_n in os.listdir(batch_exp_root):
        # Check does a results file exist
        exp_dir = os.path.join(batch_exp_root, exp_n)
        assert os.path.isdir(exp_dir)
        res_path = os.path.join(exp_dir, SCENE_RES_FNAME)
        if not os.path.isfile(res_path):
            # Make a note of failed ones (ones without a results file)
            # And skip if they failed
            failed_exp.append(exp_n)
            continue

        # Concatanate results of all successfull scenes
        res_current = pd.read_csv(res_path)
        scene_n = res_current['param/scene_name'].values
        assert len(scene_n)
        scene_n = scene_n.item()
        if scene_n not in already_loaded_scenes:
            res_all = res_current if res_all is None else pd.concat([res_all, res_current])
            already_loaded_scenes.append(scene_n)
        else:
            # Discard scenes already loaded (double experiments)
            print(f'{scene_n} already has results, discarded...')
    
    # # If there are double experiments, and the first failed, and the second succeeded,
    # # then the experiment needs to be removed from list of fails
    # failed_exp_new = []
    # failed_new_scenes = []
    # for f_exp in failed_exp:
    #     double=False
    #     for s_exp in res_all['param/scene_name'].values.tolist():
    #         if s_exp in f_exp:
    #             double=True
    #             break
    #     if not double:
    #         failed_exp_new.append(f_exp)
    #         failed_new_scenes.append(s_exp)
    # failed_exp = failed_exp_new
    
    failed_exp.sort()

    # Sort results by scene names
    res_all = res_all.sort_values(by=['param/scene_name'])
    # Make dataframe of failed scenes and sort by scene name
    failed_all = pd.DataFrame(data={"param/exp_name": failed_exp})
    failed_all = failed_all.sort_values(by=['param/exp_name'])

    # Remove blacklisted scenes
    for blacklist_scene in BLACKLISTED_SCENES:
        res_all = res_all[res_all['param/scene_name']!=blacklist_scene]
    print(f'Removed blacklisted scenes: {BLACKLISTED_SCENES}')

    if output_csv_dir is not None:
        # Save results to .csv
        res_all.to_csv(res_all_path, float_format='%.6f', index=False)
        failed_all.to_csv(failed_all_path, index=False)
        print('Saved compiled results to a .csv')
        print(f'({res_all_path})')
            
    _print_final_stats(res_all, failed_all)

    return res_all, failed_all


def reduce_one_batch(res_all, exp_name, reduction, output_csv_dir=None):
    # Divide metric results, info and params for all eperiments of a scene
    metric_cols = [col for col in res_all if col.startswith('metric/')]
    hyperparam_cols = [col for col in res_all if (col.startswith('param/') or col.startswith('info/'))]
    all_metrics_i = res_all[metric_cols]
    all_hyperparam_i = res_all[hyperparam_cols]

    # Mean of metrics
    if reduction == 'mean':
        reduced_metrics = all_metrics_i.mean(axis=0)
    elif reduction == 'median':
        reduced_metrics = all_metrics_i.median(axis=0)
    else:
        raise NotImplementedError
    reduced_metrics = pd.DataFrame(data=reduced_metrics.values[np.newaxis,...], columns=reduced_metrics.index)

    # Extract hypperparameters and info with consistent values across rows
    hyperparam_consistent, _ = _split_df_consistent_NONconsistent(all_hyperparam_i)

    # Save to csv
    if output_csv_dir:
        if reduction == 'mean':
            output_csv_path_reduced = os.path.join(output_csv_dir, '_extracted_detailed_info', f'{exp_name}_finished_mean.csv')
        elif reduction == 'median':
            output_csv_path_reduced = os.path.join(output_csv_dir, '_extracted_detailed_info', f'{exp_name}_finished_median.csv')
        else:
            raise NotImplementedError
        reduced_metrics.to_csv(output_csv_path_reduced, float_format='%.6f', index=False)
        output_csv_path_hyperparam = os.path.join(output_csv_dir, '_extracted_detailed_info', f'{exp_name}_finished_hyperparam.csv')
        hyperparam_consistent.to_csv(output_csv_path_hyperparam, index=False)

    return reduced_metrics, hyperparam_consistent


def reduce_multiple_batches(all_batches_all_res, reduction):
    all_batch_reduced_res = dict()
    # Average eaxh batch experiment results
    for k in all_batches_all_res:
        reduced_metrics_k_df, hyperparam_k_df = reduce_one_batch(
            res_all=all_batches_all_res[k], 
            exp_name=k, 
            reduction=reduction,
            output_csv_dir=None
        )
        all_batch_reduced_res[k] =  {
            'reduced_metrics_df': reduced_metrics_k_df,
            'hyperparams_df': hyperparam_k_df,
            'n_exp': all_batches_all_res[k].shape[0]}
    return all_batch_reduced_res


def merge_averaged_batches(all_batch_reduced_res, reduction, output_csv_dir=None):
    # Extract all metric names (if exists at least in one batch)
    metric_names = set()
    for k in all_batch_reduced_res:
        reduced_metrics_k_df = all_batch_reduced_res[k]["reduced_metrics_df"]
        metric_names |= set(reduced_metrics_k_df.columns.tolist())
    metric_names = sorted(list(metric_names))

    # Create a dataframe with compiled batch reduced results
    reduced_metrics_dict = {'exp_name': [], 'step': [], 'n_exp': []}
    reduced_metrics_dict.update({k: [] for k in metric_names})
    for k in all_batch_reduced_res:
        reduced_metrics_k_df = all_batch_reduced_res[k]["reduced_metrics_df"]
        reduced_metrics_dict['exp_name'].append(k)
        reduced_metrics_dict['n_exp'].append(all_batch_reduced_res[k]["n_exp"])
        step = all_batch_reduced_res[k]["hyperparams_df"]["info/step"].item()
        reduced_metrics_dict['step'].append(step)
        for col_name in metric_names:
            if col_name in reduced_metrics_k_df:
                reduced_metrics_dict[col_name].append(reduced_metrics_k_df[col_name].item())
            else:
                reduced_metrics_dict[col_name].append(None)
    reduced_metrics_df = pd.DataFrame(reduced_metrics_dict)
    
    # Create a dataframe with compiled batch hyperparameters
    hyperparams_df_list = [all_batch_reduced_res[k]['hyperparams_df'] for k in all_batch_reduced_res]
    hyperparams_df = pd.concat(hyperparams_df_list, axis=0, ignore_index=True)
    hyperparam_consistent, hyperparam_non_consistent = \
        _split_df_consistent_NONconsistent(hyperparams_df)

    # Merge the results and non-consistent hyperparameters
    if 'param/exp_name' in hyperparam_non_consistent.columns:
        hyperparam_non_consistent = hyperparam_non_consistent.drop(columns=['param/exp_name'])
    reduced_metrics_df = pd.concat([reduced_metrics_df.reset_index(drop=True),
                                 hyperparam_non_consistent.reset_index(drop=True)]
                                , axis=1)

    # Save to .csv
    if output_csv_dir is not None:
        print(f'\n')
        if reduction == 'mean':
            save_path = os.path.join(output_csv_dir, 'multi_batch_mean.csv')
        elif reduction == 'median':
            save_path = os.path.join(output_csv_dir, 'multi_batch_median.csv')
        else:
            raise NotImplementedError
        reduced_metrics_df.to_csv(save_path, float_format='%.6f', index=False)
        print(f'Saved merged averaged batches to:')
        print(f'{save_path}')

        hyperparam_consistent_save_path = os.path.join(output_csv_dir, 'multi_batch_hyperparams_consistent.csv')
        print(f'Saved merged averaged batches to:')
        print(f'{hyperparam_consistent_save_path}')
        hyperparam_consistent.to_csv(
            hyperparam_consistent_save_path, 
            index=False)
        # hyperparam_non_consistent_save_path = os.path.join(output_csv_dir, 'multi_batch_hyperparams_non_consistent.csv')
        # print(f'Saved merged averaged batches to:')
        # print(f'{hyperparam_non_consistent_save_path}')
        # hyperparam_non_consistent.to_csv(
        #     hyperparam_non_consistent_save_path, 
        #     index=False)

        print(f'\n')
    
    return reduced_metrics_df, hyperparam_consistent, hyperparam_non_consistent


def keep_only_overpalling_scenes(batch_all_res_dict):
    # Collect a list of valid scenes for every batch as well
    valid_scenes_lists = []
    for k in batch_all_res_dict:
        valid_scenes_lists.append(batch_all_res_dict[k]['param/scene_name'].tolist())
    # List of scenes that appear at least in one batch
    all_appearing_scenes = set(list(chain(*valid_scenes_lists)))


    # Find scenes which are contained in all batch experiment results
    discard_scenes = set()
    for scene_lst in valid_scenes_lists:
        discard_scenes |= (all_appearing_scenes - set(scene_lst))
    keep_scenes = list(all_appearing_scenes - discard_scenes)

    # In every compiled batch experiment result dataframe:
    # Keep only scenes contained in keep_scenes
    for k in batch_all_res_dict:
        df = batch_all_res_dict[k]
        batch_all_res_dict[k] = df[df['param/scene_name'].isin(keep_scenes)]
    
    print(f'\nKept only {len(keep_scenes)} overlapping scenes for each batch.\n\n')

    return batch_all_res_dict, keep_scenes

def _split_df_consistent_NONconsistent(df):
    df_consistent = df.copy(deep=True)
    df_non_consistent = df.copy(deep=True)
    consistent_cols = []
    non_consistents_cols = []
    for col_name in df.columns:
        if (df[col_name] == df[col_name].iloc[0]).all():
            consistent_cols.append(col_name)
        else:
            # print(f'Dropping non consistent column: {col_name}')
            non_consistents_cols.append(col_name)
    
    # Extract consistent columns
    df_consistent = df_consistent.drop(columns=non_consistents_cols)
    # Select first row of consisten columns, because all are the same
    df_consistent = df_consistent.iloc[0:1]

    # Extract non-consistent columns
    df_non_consistent = df_non_consistent.drop(columns=consistent_cols)

    return df_consistent, df_non_consistent


if __name__ == '__main__':

    WORK_SPECTA_DIR = '/home/nipopovic/MountedDirs/euler/work_specta'
    EXP_ROOT_DIR = os.path.join(WORK_SPECTA_DIR, 'experiment_logs', 'ngp_mt', 'batch_experiments')
    EXP_DIR = '2022_10_09_tmp14_all_all_trng'
    RESULTS_ROOT = os.path.join(EXP_ROOT_DIR, '_extracted_results')

    res_all_i, failed_all_i = load_all_res_batch(
        batch_exp_root=os.path.join(EXP_ROOT_DIR, EXP_DIR), 
        output_csv_dir=RESULTS_ROOT)

    mean_metrics_i, hyperparam_i = reduce_one_batch(
        res_all=res_all_i, 
        exp_name=EXP_DIR,
        output_csv_dir=RESULTS_ROOT)

    a = 1
