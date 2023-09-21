import os

path = '/home/nipopovic/MountedDirs/euler/work_specta/experiment_logs/ngp_mt/batch_experiments'
keyword = 'iccv_rebut_7M'

lst = []
for el in os.listdir(path):
    el_full = os.path.join(path, el)
    if os.path.isdir(el_full) and keyword in el:
        lst.append(el)
        print(f"        \'{el}\',")

#print(lst)