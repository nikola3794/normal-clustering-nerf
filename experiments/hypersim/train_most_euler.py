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
N_SCENES_LIST_PATH = os.path.join(TAR_GZ_PARTITIONS_ROOT, 'most_scenes_list.json')

JOB_SCRIPT_PATH = 'tmp_nerf_job_script.sh'


EXP_ROOT_DIR = '/cluster/work/cvl/specta/experiment_logs/ngp_mt/batch_experiments'
EXP_NAME = 'cvpr_all'

N_GPU = 1

N_EXP_PER_JOB = 24

bash_script_base_gen = lambda job_name: \
f'''#!/bin/bash
#BSUB -o /cluster/work/cvl/specta/experiment_logs/_tmp/  # path to output file
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
    # Create experiment directory
    rnd_l = 4
    exp_n = EXP_NAME
    rnd_str = ''.join(random.choice(string.ascii_letters) for i in range(rnd_l))
    exp_n += '_' + rnd_str
    exp_dir = os.path.join(EXP_ROOT_DIR, date_to_str() + '_' + exp_n)
    if os.path.isdir(exp_dir):
        raise FileExistsError
    os.mkdir(exp_dir)

    # Load scene stats
    with open(N_SCENES_LIST_PATH, 'r') as f:
        n_scenes_list = json.load(f)

    n_scenes_list.sort()

    exp_i = 0
    loaded_tar_gz = []
    for scene_n in n_scenes_list:
        tar_gz_n = scene_n[:-3] + '.tar.gz'

        if exp_i == 0:
            new_bash_script = bash_script_base_gen(EXP_NAME)
        
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


