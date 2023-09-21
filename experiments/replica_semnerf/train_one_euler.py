import os
import time
import random 
import string

import sys
sys.path.append(os.getcwd())
from experiments.replica_semnerf.hyperparameters import hyperparameter_str

TAR_GZ_PARTITIONS_ROOT = '/cluster/work/cvl/specta/data'

JOB_SCRIPT_PATH = 'tmp_nerf_job_script.sh'


EXP_ROOT_DIR = '/cluster/work/cvl/specta/experiment_logs/ngp_mt/individual_experiments'

N_GPU = 1

bash_script_base_gen = lambda job_name: \
f'''#!/bin/bash
#BSUB -o /cluster/work/cvl/specta/experiment_logs/_tmp/  # path to output file
#BSUB -W 03:59 # HH:MM runtime
#BSUB -n 16 # number of cpu cores
#BSUB -R "rusage[mem=4096]" # MB per CPU core
#BSUB -R "rusage[ngpus_excl_p={N_GPU}]" # number of GPU cores
#BSUB -R "select[gpu_mtotal0>=10000]" # MB per GPU core
#BSUB -J "{job_name}"
#BSUB -R lca # workaround for the current wandb cluster bug

# gcc/8.2.0
# gcc/6.3.0
module load gcc/6.3.0 python_gpu/3.8.5 eth_proxy ninja cuda/11.3.1 pigz open3d/0.9.0

PYTHONPATH=
export PYTHONPATH
source /cluster/home/nipopovic/python_envs/ngp_pl/bin/activate

#pip install models/csrc/

tar -I pigz -xf {TAR_GZ_PARTITIONS_ROOT}/Replica_semantic_NeRF_seq1.tar.gz -C ${{TMPDIR}}/



nvidia-smi

'''

rnd_str = lambda l: ''.join(random.choice(string.ascii_letters) for i in range(l)) 

nerf_basic_call = lambda scene_n, exp_n, exp_dir: \
f'''python train_nerf.py \
--data_root_dir=${{TMPDIR}}/Replica_Semantic_NeRF_seq1/{scene_n} \
--log_root_dir={exp_dir} \
--exp_name={rnd_str(6)} \
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
    assert os.path.isdir(EXP_ROOT_DIR)
    exp_dir = EXP_ROOT_DIR

    scene_n = 'room_0'
    new_bash_script = bash_script_base_gen('lst_scn')
    job_command = nerf_basic_call(scene_n, 'one_exp', exp_dir)
    job_command += hyperparameter_str(N_GPU, 3) + '\n\n'
    new_bash_script += job_command
    COMMAND = './'
    #COMMAND = 'bsub < '
    with open(JOB_SCRIPT_PATH, 'w') as f:
        f.write(new_bash_script) 
    os.system(f'chmod 770 {JOB_SCRIPT_PATH}')
    os.system(f'{COMMAND}{JOB_SCRIPT_PATH}')
        


