DOWNSAMPLE = 1.0
hyperparameter_str = lambda n_gpu, epochs=30: \
(
" --no_debug"
# --log_root_dir=
# --exp_name=
###########################
# --data_root_dir=
" --dataset_name=hypersim"
" --split=train"
" --split_factor=0.5"
" --keep_N_tr=-1"
f" --downsample={DOWNSAMPLE}"
" --load_depth_gt"
" --load_norm_gt"
#" --load_norm_depth_gt"
#" --load_sem_gt"
#" --load_sem_WF_gt"
##########################
" --model_name=NGPMT"
" --scale=0.5"
" --grid_size=128"
" --density_tresh_decay=1.0"
" --rend_max_samples=1024"
" --rend_near_dist=0.01"
# --use_exposure
#" --pred_norm_nn"
#" --pred_norm_nn_norm"
" --pred_norm_depth"
#" --pred_sem"
# --optimize_ext
##########################
" --loss_opacity_w=1e-3"
" --loss_distortion_w=0"
" --loss_depth_w=0"
#" --loss_norm_GT_depth"
" --loss_norm_depth_dot_w=0" 
" --loss_norm_depth_L1_w=0" 
################################
" --loss_reg_depth_w=0"
#
" --loss_sem_w=0"
" --loss_manhattan_nerf_w=0"
#
" --loss_norm_D_C_ort_dot_w=2e-3"
" --loss_norm_D_C_centr_dot_w=2e-3"
" --loss_norm_D_C_centr_L1_w=2e-3"
" --loss_norm_D_C_can_dot_w=0"
" --loss_norm_D_C_can_L1_w=0"
# #
" --loss_norm_can_tres=0.01"
" --loss_norm_can_start=500"
" --loss_norm_can_end=-1"
" --loss_norm_can_grow=2500"
" --loss_norm_yaw_offset_ang=0.0"
" --loss_norm_pitch_offset_ang=0.0"
" --loss_norm_roll_offset_ang=0.0"
###############################
##########################
" --lr=1e-2" 
" --lr_dR_norm_glob=0"
" --dR_norm_glob_coding=quaternion"
f" --num_epochs={epochs}"
" --batch_size=8192"
" --ray_sampling_strategy=all_images_triang_patch"
#" --random_tr_poses"
" --triang_max_expand=0"
" --anneal_strategy=none"
" --anneal_steps=0"
f" --num_gpus={n_gpu}"
" --grad_clip=0.05"
# --random_bg
###############################
# --eval_lpips
# --val_only
###########################
" --tqdm_refresh_rate=100"
" --pl_log_every_n_steps=200"
#" --save_test_vis"
" --log_gt_test_vis"
f" --downsample_vis={0.5/DOWNSAMPLE}"
#" --save_test_preds"
#" --save_train_preds"
f" --downsample_pred_save={0.5/DOWNSAMPLE}"
# --ckpt_path=
# --weight_path=
)
