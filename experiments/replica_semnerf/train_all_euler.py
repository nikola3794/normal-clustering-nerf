import os
import json
import time
import random
import string

import sys
sys.path.append(os.getcwd())
from experiments.replica_semnerf.hyperparameters import hyperparameter_str


TAR_GZ_PARTITIONS_ROOT = '/cluster/work/cvl/specta/data'

JOB_SCRIPT_PATH = 'tmp_nerf_job_script.sh'


EXP_ROOT_DIR = '/cluster/work/cvl/specta/experiment_logs/ngp_mt/batch_experiments'

N_GPU = 1

SLURM_header = lambda job_name: \
f'''#!/bin/bash

#SBATCH --output="/cluster/work/cvl/specta/experiment_logs/_tmp/%j.out"
#SBATCH --time=23:59:00
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4096
#SBATCH --gpus={N_GPU}  
#SBATCH --gpus=rtx_2080_ti:1
#SBATCH --gres=gpumem:10000m
#SBATCH --job-name="{job_name}"
'''

LSF_header =lambda job_name: \
f'''#!/bin/bash

#BSUB -o /cluster/work/cvl/specta/experiment_logs/_tmp/  # path to output file
#BSUB -W 23:59 # HH:MM runtime
#BSUB -n 16 # number of cpu cores
#BSUB -R "rusage[mem=4096]" # MB per CPU core
#BSUB -R "rusage[ngpus_excl_p={N_GPU}]" # number of GPU cores
#BSUB -R "select[gpu_mtotal0>=10000]" # MB per GPU core
#BSUB -J "{job_name}"
#BSUB -R lca # workaround for the current wandb cluster bug
'''

bash_script_base_gen = lambda job_name: \
f'''{SLURM_header(job_name)}

module load gcc/6.3.0 python_gpu/3.8.5 eth_proxy ninja cuda/11.3.1 pigz open3d/0.9.0
PYTHONPATH=
export PYTHONPATH
source /cluster/home/nipopovic/python_envs/ngp_pl/bin/activate

#pip install models/csrc/

tar -I pigz -xf {TAR_GZ_PARTITIONS_ROOT}/Replica_semantic_NeRF_seq1.tar.gz -C ${{TMPDIR}}/

nvidia-smi
'''

nerf_basic_call = lambda scene_n, exp_n, exp_dir: \
f'''python train_nerf.py \
--data_root_dir=${{TMPDIR}}/Replica_Semantic_NeRF_seq1/{scene_n} \
--log_root_dir={exp_dir} \
'''


def date_to_str():
    """
    :return: String of current date in Y-M-D format
    """
    ISOTIMEFORMAT = '%Y_%m_%d'
    string = '{}'.format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))
    return string



#COMMAND = './'
#COMMAND = 'bsub < '
COMMAND = 'sbatch < '
if __name__ == "__main__":

    EXP_NAME = 'replica_repeat'

    # Create experiment directory
    rnd_l = 4
    exp_n = EXP_NAME
    rnd_str = ''.join(random.choice(string.ascii_letters) for i in range(rnd_l))
    exp_n += '_' + rnd_str
    exp_dir = os.path.join(EXP_ROOT_DIR, date_to_str() + '_' + exp_n)
    if os.path.isdir(exp_dir):
        raise FileExistsError
    os.mkdir(exp_dir)

    n_scenes_list = [
        'room_0',
        'room_1',
        'room_2',
        'office_2',
        'office_3',
        #
        'office_1',
        'office_4',
        'office_0',
    ]

    N_EXP_PER_JOB = 8

    exp_i = 0
    for scene_n in n_scenes_list:
        if exp_i == 0:
            new_bash_script = bash_script_base_gen('replica')
        
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
            os.system(f'{COMMAND}{JOB_SCRIPT_PATH}')

    # Last job didnt run
    if exp_i != 0:
        with open(JOB_SCRIPT_PATH, 'w') as f:
            f.write(new_bash_script) 
            # TODO Find a better way
            os.system(f'{COMMAND}{JOB_SCRIPT_PATH}')

