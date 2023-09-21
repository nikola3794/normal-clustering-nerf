
import os, sys
sys.path.append(os.getcwd())

import datasets.hypersim_src.utils as ds_utils
from datasets.hypersim_src.scene import HypersimScene

from matplotlib import pyplot as plt
# Scenes whith cutoff 
# ai_001_004
# ai_001_006
# ai_002_004
# ai_001_009 (windows)
# ai_001_010 (windows)
# ...

if __name__ == '__main__':
    which_labels = ['semantics' ,'depth', 'normals']
    which_labels = ['semantics_WF', 'normals']
    dataset = HypersimScene(
        scene_root_dir='SCENE_DIR',
        scene_metadata_path=None,
        downscale_factor=1.0,
        which_labels=which_labels,
        which_cams=['cam_00'],
        which_split='train',
    )

    n_labels = len(which_labels)
    for img_id in dataset.img_ids:
        # try:
        f, axarr = plt.subplots(1, 1 + n_labels)
        # cam_n = dataset.cams_list[0]
        # n = int(n.split('_')[1])
        # img = dataset.get_img(n, i)
        # img_id = ds_utils._get_img_id_from_num(n, i)
        cam_n, frame_name = ds_utils._split_img_id(img_id)
        n = int(cam_n.split('_')[1])
        i = int(frame_name)
        img = dataset.get_img(n, i)
        axarr[0].imshow(img)
        axarr[0].set_title(img_id)
        for lab_i, lab in enumerate(which_labels):
            label = dataset.get_label(lab, n, i)
            label = dataset._convert_single_label_to_vis_format(
                label,
                lab
            )
            axarr[lab_i+1].imshow(label)
        # else:
        #     try:
        #         axarr[cam_i,0].imshow(img)
        #         axarr[cam_i,0].set_title(img_id)
        #         for lab_i, lab in enumerate(which_labels):
        #             label = dataset.get_label(lab, n, i)
        #             label = dataset._convert_single_label_to_vis_format(
        #                 label,
        #                 lab
        #             )
        #             axarr[cam_i, lab_i+1].imshow(label)
        #     except:
        #         pass
        plt.show()
        import time
        time.sleep(2.0)
        plt.close()
        # except:
        #     pass