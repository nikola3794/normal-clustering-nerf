import argparse


def get_opts():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--no_debug', action='store_true', default=False,
                        help='When false, the code is in debug mode.')
    parser.add_argument('--log_root_dir', type=str, default='SPECIFY_DIR',
                        help='Directory path to store experiment logs.')
    parser.add_argument('--exp_name', type=str, default='',
                        help='Experiment name (recommended to be unique).')

    # Dataset configuration
    parser.add_argument('--data_root_dir', type=str, default='SPECIFY_DIR',
                        help='Dataset root directory.')
    parser.add_argument('--dataset_name', type=str, default='hypersim',
                        choices=['hypersim', 'scannet_manhattan', 'replica_semnerf'],
                        help='Name of the dataset to be used.')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'trainval', 'trainvaltest'],
                        help='Defines which split is used during training.')
    parser.add_argument('--split_factor', type=float, default=0.5,
                        help='''Split factor for the training partition 
                                when applicable (rest is test)''')
    parser.add_argument('--keep_N_tr', type=int, default=-1,
                        help='Keep N training images (for sparse-view training)')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='Downsample factor (<=1.0) for training and testing')
    parser.add_argument('--load_depth_gt', action='store_true', default=False,
                        help='Load depth ground-truth.')
    parser.add_argument('--load_norm_gt', action='store_true', default=False,
                        help='Load normals ground-truth.')
    parser.add_argument('--load_norm_depth_gt', action='store_true', default=False,
                        help='Load normals ground-truth (extracted from depth gt) .')
    parser.add_argument('--load_sem_gt', action='store_true', default=False,
                        help='Load semantics ground-truth.')
    parser.add_argument('--load_sem_WF_gt', action='store_true', default=False,
                        help='Load semantics ground-truth (only wall/floor/background).')

    # Model parameters
    parser.add_argument('--model_name', type=str, default='NGPMT',
                        help='Name of the model to be used.')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='Scene scale (whole scene must lie in [-scale, scale]^3')
    parser.add_argument('--grid_size', type=int, default=128)
    parser.add_argument('--density_tresh_decay', type=float, default=1.0)
    parser.add_argument('--rend_max_samples', type=int, default=1024,
                        help='Maximum samples on a ray.')
    parser.add_argument('--rend_near_dist', type=float, default=0.01,
                        help='Near distance of a ray.')
    parser.add_argument('--use_exposure', action='store_true', default=False,
                        help='whether to train in HDR-NeRF setting.')
    parser.add_argument('--pred_norm_nn', action='store_true', default=False,
                        help='Predict surface normals from the network.')
    parser.add_argument('--pred_norm_nn_norm', action='store_true', default=False,
                        help='Predict normalized surface normals from the network.')
    parser.add_argument('--pred_norm_depth', action='store_true', default=False,
                        help='Extract surgface normals from rendered depth')
    parser.add_argument('--pred_sem', action='store_true', default=False,
                        help='Predict semantics from the network.')
    

    # Loss parameters
    parser.add_argument('--loss_opacity_w', type=float, default=1e-3,
                        help='''Weight of the opacity penalty loss (see losses.py),
                        0 to disable (default), to enable,''')
    parser.add_argument('--loss_distortion_w', type=float, default=0,
                        help='''Weight of distortion loss (see losses.py),
                        0 to disable (default), to enable,
                        a good value is 1e-3 for real scene and 1e-2 for synthetic scene
                        ''')
    parser.add_argument('--loss_depth_w', type=float, default=0,
                        help='Depth L2 supervision weight.')
    parser.add_argument('--loss_sem_w', type=float, default=0,
                        help='Semantic cross-entropy supervision weight.')
    # TODO Old?
    parser.add_argument('--loss_norm_GT_depth', action='store_true', default=False,
                        help='')
    parser.add_argument('--loss_norm_depth_dot_w', type=float, default=0,
                        help='Supervise normals (from endered depth) with dot product loss')
    parser.add_argument('--loss_norm_depth_L1_w', type=float, default=0,
                        help='Supervise normals (from endered depth) with L1 loss')
    #
    parser.add_argument('--loss_reg_depth_w', type=float, default=0,
                        help='Reg nerf depth regularization on unseen views')
    #
    parser.add_argument('--loss_manhattan_nerf_w', type=float, default=0,
                        help='''Manhattan nerf wall/floor regularization loss. 
                                Should be combined with loss_sem_w''')
    #
    parser.add_argument('--loss_norm_D_C_ort_dot_w', type=float, default=0,
                        help='Cluster normal orthogonality regularization loss.')
    parser.add_argument('--loss_norm_D_C_centr_dot_w', type=float, default=0,
                        help='Cluster normal centroid dot regularization loss.')
    parser.add_argument('--loss_norm_D_C_centr_L1_w', type=float, default=0,
                        help='Cluster normal centroid L1 regularization loss.')
    parser.add_argument('--loss_norm_D_C_can_dot_w', type=float, default=0,
                        help='Cluster normal centroid dot supervision w.r.t canonical normals.')
    parser.add_argument('--loss_norm_D_C_can_L1_w', type=float, default=0,
                        help='Cluster normal centroid L1 supervision w.r.t canonical normals.')
    # The below few arguments are for all normal regularization experiments
    # They are not just for the "loss_norm_D_C_can"
    # TODO remove "_can" from their names
    parser.add_argument('--loss_norm_can_tres', type=float, default=0,
                        help='''Trehsold determining are two normals close enough.
                                For normal regularization, it determines
                                whether a normal is close enough to a cluster centroid.
                                For the canonical loss it determines is the normal
                                close enough to one of the canconical normals to be supervised.''')
    parser.add_argument('--loss_norm_can_start', type=float, default=0,
                        help='Start using normal regularization loss only after n iterations.')
    parser.add_argument('--loss_norm_can_end', type=float, default=-1,
                        help='End using normal regularization loss after n iterations.')
    parser.add_argument('--loss_norm_can_grow', type=float, default=1,
                        help='''When normal regularization loss turs on, it starts 
                        increasing from 0 to w. This arguments defiens how much
                        iterations it takes to build up to w''')
    parser.add_argument('--loss_norm_yaw_offset_ang', type=float, default=0,
                        help='Yaw offset of the used coordinates')
    parser.add_argument('--loss_norm_pitch_offset_ang', type=float, default=0,
                        help='Pitch offset of the used coordinates')
    parser.add_argument('--loss_norm_roll_offset_ang', type=float, default=0,
                        help='Roll offset of the used coordinates')


    # Training options
    parser.add_argument('--optimize_ext', action='store_true', default=False,
                        help='Whether to optimize extrinsics as well.')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='Learning rate')
    parser.add_argument('--lr_dR_norm_glob', type=float, default=0,
                        help='Lr when optimizing the delta rotation.')
    parser.add_argument('--dR_norm_glob_coding', type=str, default='axis_angle',
                        help='How to code rotation matrix when optimizing it.')
    parser.add_argument('--num_epochs', type=int, default=4,
                        help='Number of training epochs (epoch=1000 iters).')
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='Number of rays in a batch.')
    parser.add_argument('--ray_sampling_strategy', type=str, default='all_images',
                        choices=['all_images', 'same_image', 
                                 'same_image_triang', 'all_images_triang', 'all_images_triang_val',
                                 'same_image_triang_patch', 'all_images_triang_patch'],
                        help='''Ray sampling strategy''')
    parser.add_argument('--random_tr_poses', action='store_true', default=False,
                        help='Whether to generate random unseen poses in addition to existing data.')
    parser.add_argument('--triang_max_expand', type=int, default=0,
                        help='''Triangle size when sampling triangle rays. 
                        (pixel gap betwee selected corners)''')
    # seems to not make much of a difference when annealing
    parser.add_argument('--anneal_strategy', type=str, default='none',
                    choices=['avoid_near', 'depth', 'none'],
                    help='''Anneals the ray in the beginning of training 
                    to avoid floaters''')
    parser.add_argument('--anneal_steps', type=int, default=0,
                        help='How many steps to apply ray annealing')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of gpus')
    parser.add_argument('--grad_clip', type=float, default=0.05,
                        help='Gradient clipping')
    # Does not seem to make a difference
    parser.add_argument('--random_bg', action='store_true', default=False,
                        help='''whether to train with random bg color (real scene only)
                        to avoid objects with black color to be predicted as transparent
                        ''')

    # validation options
    parser.add_argument('--eval_lpips', action='store_true', default=False,
                        help='evaluate lpips metric (consumes more VRAM)')
    parser.add_argument('--val_only', action='store_true', default=False,
                        help='run only validation (need to provide ckpt_path)')

    # misc
    parser.add_argument('--tqdm_refresh_rate', type=int, default=100,
                        help='Refresh tqdm output after n steps')
    parser.add_argument('--pl_log_every_n_steps', type=int, default=100,
                        help='Log torch-lighting loggers after n steps')
    parser.add_argument('--save_test_vis', action='store_true', default=False,
                        help='whether to save test image and video visualizations')
    parser.add_argument('--log_gt_test_vis', action='store_true', default=True,
                        help='whether to log GT test image to wandb')
    parser.add_argument('--downsample_vis', type=float, default=0.5,
                        help='''downsample factor (<=1.0) for saved rendered images
                                with respect to downsample''')
    parser.add_argument('--save_test_preds', action='store_true', default=False,
                        help='whether to store predictions on the test set')
    parser.add_argument('--save_train_preds', action='store_true', default=False,
                        help='whether to store predictions on the train set')
    parser.add_argument('--downsample_pred_save', type=float, default=0.5,
                        help='''downsample factor (<=1.0) for saved predictions
                                with respect to downsample''')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained checkpoint to load (excluding optimizers, etc)')
    parser.add_argument('--save_checkpoint', action='store_true', default=False)
                        

    return parser.parse_args()
