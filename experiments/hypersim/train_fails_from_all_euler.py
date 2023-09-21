import os
import json
import time
import random
import string

import sys
sys.path.append(os.getcwd())
from experiments.hypersim.hyperparameters import hyperparameter_str


TAR_GZ_PARTITIONS_ROOT = '/cluster/work/cvl/specta/data/Hypersim_targz_partitions'
SCENES_METADATA_PATH = os.path.join(TAR_GZ_PARTITIONS_ROOT, 'all_scenes_metadata.json')

JOB_SCRIPT_PATH = 'tmp_nerf_job_script.sh'


EXP_ROOT_DIR = '/cluster/work/cvl/specta/experiment_logs/ngp_mt/batch_experiments'
EXP_NAME = '2022_10_12_tmp30_all_all_EmiM'

N_GPU = 1

N_EXP_PER_JOB = 5

bash_script_base_gen = lambda job_name: \
f'''#!/bin/bash
#BSUB -o /cluster/work/cvl/specta/experiment_logs/_tmp  # path to output file
#BSUB -W 23:59 # HH:MM runtime
#BSUB -n 16 # number of cpu cores
#BSUB -R "rusage[mem=4096]" # MB per CPU core
#BSUB -R "rusage[ngpus_excl_p={N_GPU}]" # number of GPU cores
#BSUB -R "select[gpu_mtotal0>=10000]" # MB per GPU core
#BSUB -J "{job_name}"
#BSUB -R lca # workaround for the current wandb cluster bug

module load gcc/6.3.0 python_gpu/3.8.5 eth_proxy ninja cuda/11.3.1 pigz open3d/0.9.0
PYTHONPATH=
export PYTHONPATH
source /cluster/home/nipopovic/python_envs/ngp_pl/bin/activate

#pip install models/csrc/

rsync -aP {SCENES_METADATA_PATH} ${{TMPDIR}}/

nvidia-smi

'''

nerf_basic_call = lambda scene_n, exp_n, exp_dir: \
f'''python train_nerf.py \
--data_root_dir=${{TMPDIR}}/{scene_n} \
--log_root_dir={exp_dir} \
'''


def date_to_str():
    """
    :return: String of current date in Y-M-D format
    """
    ISOTIMEFORMAT = '%Y_%m_%d'
    string = '{}'.format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))
    return string


if __name__ == "__main__":
    # Check that experiment directory exists
    exp_dir = os.path.join(EXP_ROOT_DIR, EXP_NAME)
    assert os.path.isdir(exp_dir)

    # Load all scene names that:
    # *either dont have a folder at all
    # *there are no results in the folder
    with open(SCENES_METADATA_PATH, 'r') as f:
        scenes_metadata = json.load(f)
    created_scene_exp_dirs = [x.path for x in os.scandir(exp_dir) if os.path.isdir(x)]
    created_scene_exp_dirs.sort()
    needs_reset = []
    i = 0
    
    for j, scene_n in enumerate(sorted(list(scenes_metadata.keys()))):
        needs_restart = True
        exp_name_no_date = '_'.join(EXP_NAME.split('_')[3:])
        scene_n_dir_start = os.path.join(exp_dir, exp_name_no_date+'_'+scene_n)
        for created_scene_exp_n in created_scene_exp_dirs:
            if created_scene_exp_n.startswith(scene_n_dir_start):
                if os.path.isfile(os.path.join(EXP_ROOT_DIR, created_scene_exp_n, 'results.csv')):
                    needs_restart = False
                break
        if needs_restart:
            needs_reset.append(scene_n)

    needs_reset.sort()

    exp_i = 0
    loaded_tar_gz = []
    for scene_n in needs_reset:
        tar_gz_n = scene_n[:-3] + '.tar.gz'

        if exp_i == 0:
            new_bash_script = bash_script_base_gen('fail_reset')
        
        if tar_gz_n not in loaded_tar_gz:
            new_bash_script += f'tar -I pigz -xf {TAR_GZ_PARTITIONS_ROOT}/{tar_gz_n} -C ${{TMPDIR}}/ \n\n'
            loaded_tar_gz.append(tar_gz_n)

        job_command = nerf_basic_call(scene_n, EXP_NAME, exp_dir)
        job_command += hyperparameter_str(N_GPU) + '\n\n'
        new_bash_script += job_command
        
        exp_i += 1

        if exp_i == N_EXP_PER_JOB:
            exp_i = 0
            loaded_tar_gz = []
            with open(JOB_SCRIPT_PATH, 'w') as f:
                f.write(new_bash_script) 
            # TODO Find a better way
            os.system(f'bsub < {JOB_SCRIPT_PATH}')

    # Last job didnt run
    if exp_i != 0:
        with open(JOB_SCRIPT_PATH, 'w') as f:
            f.write(new_bash_script) 
        # TODO Find a better way
        os.system(f'bsub < {JOB_SCRIPT_PATH}')


