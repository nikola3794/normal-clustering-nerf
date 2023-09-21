import os

import json

# '/home/nipopovic/MountedDirs/hl_task_prediction/hl_task_pred_root/data/data_sets/Hypersim_targz_partitions'
# '/cluster/work/cvl/specta/data/Hypersim_targz_partitions'
TAR_GZ_PARTITIONS_ROOT = '/home/nipopovic/MountedDirs/hl_task_prediction/hl_task_pred_root/data/data_sets/Hypersim_targz_partitions'
SCENES_METADATA_PATH = os.path.join(TAR_GZ_PARTITIONS_ROOT, 'all_scenes_metadata.json')
SAVE_PATH = os.path.join(TAR_GZ_PARTITIONS_ROOT, 'most_scenes_list.json')

with open(SCENES_METADATA_PATH, 'r') as f:
    scene_metadata = json.load(f)

scene_list = []
for scene_name in scene_metadata:
    metadata_i = scene_metadata[scene_name]
    if scene_name == "ai_003_001":
        continue
    cam = 'cam_00' if 'cam_00' in metadata_i['cams'] else 'cam_01'
    if len(metadata_i['cams'][cam]['img_names']) > 80:
        scene_list.append(scene_name)

with open(SAVE_PATH, 'w') as f:
    json.dump(scene_list, f)
a = 1
