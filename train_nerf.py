import os
import csv
import glob
import string
import random
from copy import deepcopy
import tarfile
import io
import tempfile
from tkinter import TRUE

import imageio
import cv2
from imgviz import label_colormap

import math
import numpy as np
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
import torchvision


# config
from opt import get_opts

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays
from datasets.hypersim_src.utils import _extract_normals_from_depth_batch

# models
from kornia.utils.grid import create_meshgrid3d
from models import *
from models.rendering import render

# optimizer, losses, metrics
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFMTLoss
from metrics.metrics import NeRFMTMetricsPerIm
from torchmetrics import PeakSignalNoiseRatio

# pytorch-lightning
import pytorch_lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available


from pytorch3d.transforms import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    axis_angle_to_matrix,
)

from losses import _normals_clustering

from scipy.spatial.transform import Rotation as scipy_Rotation

from utils import slim_ckpt, load_ckpt

import warnings; warnings.filterwarnings("ignore")

SQRT3 = math.sqrt(3)


def depth2img(depth, min=None, max=None):
    if (min == None) or (max == None):
        min = depth.min()
        max = depth.max()
    depth = (depth-min)/(max-min)
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


def coding_to_matrix(vec, coding):
    if coding == 'quaternion':
        return quaternion_to_matrix(vec)
    elif coding == 'axis_angle':
        return axis_angle_to_matrix(vec)
    elif coding == 'axis_angle_custom':
        return axisangle_to_R(vec)
    else:
        raise NotImplementedError


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.warmup_steps = 256
        self.update_interval = 16

        #Dataset arguments
        dataset = dataset_dict[hparams.dataset_name]
        kwargs = {'root_dir': hparams.data_root_dir,
                  'split_factor': hparams.split_factor}

        # Scene rotation offset
        norm_yaw_offset_rad = hparams.loss_norm_yaw_offset_ang * (math.pi/180.0)
        norm_pitch_offset_rad = hparams.loss_norm_pitch_offset_ang * (math.pi/180.0)
        norm_roll_offset_rad = hparams.loss_norm_roll_offset_ang *(math.pi/180.0)
        if norm_yaw_offset_rad == norm_pitch_offset_rad == norm_roll_offset_rad == 0.0:
            self.R_offset = torch.eye(3, dtype=torch.float32)
            kwargs['R_offset'] = None
        else:
            self.R_offset = euler_angles_to_matrix(
                torch.tensor(
                    [norm_yaw_offset_rad, norm_pitch_offset_rad, norm_roll_offset_rad],
                    dtype=torch.float32),
                'ZYX').detach()
            kwargs['R_offset'] = self.R_offset

        # Datasets
        self.train_dataset_init = dataset(hparams=hparams, split=hparams.split, **kwargs)
        if hparams.random_tr_poses:
            self.train_dataset_init._generate_random_poses()
        # Sparse training views
        if hparams.keep_N_tr != -1:
            N_total = self.train_dataset_init.poses.shape[0]
            new_indices = torch.linspace(0, N_total-1, hparams.keep_N_tr, dtype=torch.long)
            self.train_dataset_init.img_ids = [x for i, x in enumerate(self.train_dataset_init.img_ids) if i in new_indices ]
            for k in self.train_dataset_init.labels:
                self.train_dataset_init.labels[k] = self.train_dataset_init.labels[k][new_indices]
            self.train_dataset_init.poses = self.train_dataset_init.poses[new_indices]
            self.train_dataset_init.rays = self.train_dataset_init.rays[new_indices]
        self.test_dataset_init = dataset(hparams=hparams, split='test', **kwargs)

        if hparams.pred_sem: 
            kwargs = {'n_sem_cls': self.train_dataset_init.n_classes}
        # Model
        rgb_act = 'None' if hparams.use_exposure else 'Sigmoid'
        if hparams.model_name == 'NGPMT':
            self.model = NGPMT(
                scale=hparams.scale, 
                grid_size=hparams.grid_size, 
                rgb_act=rgb_act,
                pred_sem=hparams.pred_sem,
                pred_norm=hparams.pred_norm_nn,
                **kwargs)
        else:
            raise NotImplementedError
        G = self.model.grid_size
        self.model.register_buffer('density_grid',
            torch.zeros(self.model.cascades, G**3))
        self.model.register_buffer('grid_coords',
            create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))
            
        # Losses and metrics
        self.loss = NeRFMTLoss(hparams_dict=vars(hparams))
            
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_metrics = NeRFMTMetricsPerIm(hparams_dict=vars(hparams), **kwargs)

    def forward(self, batch, split):
        if split=='train':
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
            if self.hparams.random_tr_poses:
                poses = torch.cat((poses, self.random_poses[batch['rnd_img_idxs']]), dim=0)
                # Have same pixels for random camera poses as well (for now)
                directions = torch.cat((directions, directions), dim=0)
        else:
            poses = batch['pose']
            directions = self.directions

        if self.hparams.optimize_ext:
            dR = axisangle_to_R(self.dR[batch['img_idxs']])
            poses[..., :3] = dR @ poses[..., :3]
            poses[..., 3] += self.dT[batch['img_idxs']]

        rays_o, rays_d = get_rays(directions, poses)

        kwargs = {'test_time': split!='train',
                  'random_bg': self.hparams.random_bg}
        if self.hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1/256
        if self.hparams.use_exposure:
            kwargs['exposure'] = batch['exposure']

        kwargs['ray_sampling_strategy'] = self.hparams.ray_sampling_strategy
        if self.hparams.pred_sem:
            kwargs['n_sem_cls'] = self.train_dataset.n_classes

        kwargs['max_samples'] = self.hparams.rend_max_samples
        kwargs['near_distance'] = self.hparams.rend_near_dist

        kwargs['anneal_strategy'] = self.hparams.anneal_strategy
        if kwargs['anneal_strategy'] == 'depth':
            assert 'depth' in batch
            kwargs['depth'] = batch['depth']
        kwargs['anneal_steps'] = self.hparams.anneal_steps
        kwargs['global_step'] = self.global_step

        kwargs['norm_depth'] = hparams.pred_norm_depth
        kwargs['pred_norm_nn_norm'] = hparams.pred_norm_nn_norm

        results = render(self.model, rays_o, rays_d, **kwargs)

        # Offset and correct normals if necessary
        with torch.cuda.amp.autocast(dtype=torch.float32):
            # if (self.R_offset != torch.eye(3, device=self.R_offset.device)).any():
            #     if self.R_offset.device != self.device:
            #         self.R_offset = self.R_offset.to(self.device)
            #     if 'norm_depth' in results:
            #         results['norm_depth'] = (self.R_offset @ results['norm_depth'].T).T
            #     if 'norm_nn' in results:
            #         results['norm_nn'] = (self.R_offset @ results['norm_nn'].T).T
            if self.hparams.lr_dR_norm_glob > 0:
                raise NotImplementedError
                # dR_glob_norm = coding_to_matrix(self.dR_glob_norm_coded, self.hparams.dR_norm_glob_coding)
                # if 'norm_depth' in results:
                #     results['norm_depth'] = (dR_glob_norm @ results['norm_depth'].T).T
                # if 'norm_nn' in results:
                #     results['norm_nn'] = (dR_glob_norm @ results['norm_nn'].T).T

        return results

    def setup(self, stage):
        '''This hook is called for every process when using DDP'''
        self.train_dataset = deepcopy(self.train_dataset_init)
        self.test_dataset = deepcopy(self.test_dataset_init)
        # TODO This will probably not work, since the first process will delete it
        del self.train_dataset_init
        del self.test_dataset_init

    def configure_optimizers(self):
        # define additional parameters
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))
        if self.hparams.random_tr_poses:
            self.register_buffer('random_poses', self.train_dataset.random_poses.to(self.device))

        if self.hparams.optimize_ext:
            N = len(self.train_dataset.poses)
            self.register_parameter('dR',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))
            self.register_parameter('dT',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))
        if self.hparams.lr_dR_norm_glob > 0:
            identity_mtx = torch.eye(3, device=self.device)
            if self.hparams.dR_norm_glob_coding == 'quaternion':
                identity_coded = matrix_to_quaternion(identity_mtx)
            elif self.hparams.dR_norm_glob_coding in ['axis_angle', 'axis_angle_custom']:
                identity_coded = torch.zeros((3,), device=self.device)
            else:
                raise NotImplementedError
            self.register_parameter('dR_glob_norm_coded', nn.Parameter(identity_coded))

        load_ckpt(self.model, self.hparams.weight_path)

        feat_params = []
        net_params = []
        for n, p in self.named_parameters():
            if n in ['dR', 'dT', 'dR_glob_norm_coded', 'theta_WF']:
                continue
            if 'xyz_encoder' in n:
                feat_params += [p]
            else:
                net_params += [p]

        opts = []
        opts.append({'params': feat_params, 'lr': self.hparams.lr, 'eps': 1e-15, 'weight_decay':0})
        opts.append({'params': net_params, 'lr': self.hparams.lr, 'eps': 1e-15, 'weight_decay':1e-6})
        if self.hparams.optimize_ext:
            # learning rate is hard-coded
            opts.append({'params': [self.dR, self.dT], 'lr': 1e-6, 'weight_decay':0})
        if self.hparams.lr_dR_norm_glob > 0:
            # learning rate is hard-coded
            opts.append({'params': [self.dR_glob_norm_coded], 'lr': self.hparams.lr_dR_norm_glob, 'weight_decay':0})
        if self.hparams.loss_manhattan_nerf_w > 0:
            init = torch.tensor([0.0], device=self.device)
            self.register_parameter('theta_WF', nn.Parameter(init))
            opts.append({'params': [self.theta_WF], 'lr': self.hparams.lr, 'weight_decay':0})
        self.opt = FusedAdam(opts)
        feat_sch = CosineAnnealingLR(self.opt,
                                    self.hparams.num_epochs,
                                    0)
                                    #self.hparams.lr/30)

        return [self.opt], [feat_sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=16, #16
                          persistent_workers=True, #True
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=8, #8
                          batch_size=None,
                          pin_memory=True)

    def on_train_start(self):
        self.model.mark_invisible_cells(self.train_dataset.K,
                                        self.device, # TODO Remove arg
                                        self.poses, 
                                        self.train_dataset.img_wh,
                                        near_distance=self.hparams.rend_near_dist,
                                        chunk=MARK_INVISIBLE_CHUNK)

    def training_step(self, batch, batch_nb, *args):
        if self.global_step%self.update_interval == 0:
            density_threshold = 0.01*self.hparams.rend_max_samples/3**0.5
            density_threshold *= self.hparams.density_tresh_decay
            self.model.update_density_grid(density_threshold,
                                           warmup=self.global_step<self.warmup_steps,
                                           erode=self.hparams.dataset_name=='colmap')

        results = self(batch, split='train')
        
        kwargs = {'global_step': self.global_step}
        if self.hparams.loss_manhattan_nerf_w > 0:
            kwargs['theta_WF'] = self.theta_WF
        loss_d = self.loss(results, batch, **kwargs)
        if self.hparams.use_exposure:
            raise NotImplementedError
            # To return this function, put it whichback into the self.loss()
            zero_radiance = torch.zeros(1, 3, device=self.device)
            unit_exposure_rgb = self.model.log_radiance_to_rgb(zero_radiance,
                                    **{'exposure': torch.ones(1, 1, device=self.device)})
            loss_d['unit_exposure'] = \
                0.5*(unit_exposure_rgb-self.train_dataset.unit_exposure_rgb)**2
        
        # Discard random poses for later metric calculation
        if hparams.random_tr_poses:
            gt_l = batch['rgb'].shape[0]
            for k in results:
                if k in ['rm_samples', 'vr_samples', 'ws']:
                    continue
                results[k] = results[k][:gt_l]
        if self.hparams.ray_sampling_strategy == 'all_images_triang_val' and batch['rgb'] == None:
            pass
        else:
            with torch.no_grad():
                self.train_psnr(results['rgb'], batch['rgb'])
                # ray marching samples per ray (occupied space on the ray)
                self.log('train/rm_s', results['rm_samples']/len(batch['rgb']), prog_bar=True)
                # volume rendering samples per ray (stops marching when transmittance drops below 1e-4)
                self.log('train/vr_s', results['vr_samples']/len(batch['rgb']), prog_bar=True)
                self.log('train/psnr', self.train_psnr, prog_bar=True)
        self.log(f'lr', self.opt.param_groups[0]['lr'])
        for i, _ in enumerate(self.opt.param_groups[1:]):
            self.log(f'lr_{i+1}', self.opt.param_groups[i]['lr'])
        for k in loss_d:
            self.log(f'train/loss_{k}', loss_d[k])
        if self.hparams.lr_dR_norm_glob > 0:
            dR = coding_to_matrix(self.dR_glob_norm_coded, self.hparams.dR_norm_glob_coding)
            dR_euler = matrix_to_euler_angles(dR, 'ZYX')*180.0/math.pi
            dR_euler = dR_euler.cpu().tolist()
            names = ['yaw', 'pitch', 'roll']
            for n, ang in zip(names, dR_euler):
                self.log(f'dR/{n}', ang, prog_bar=False)

        return loss_d['total']

    def on_validation_start(self):
        torch.cuda.empty_cache()
        if self.hparams.save_test_vis:
            self.val_dir = os.path.join(self.hparams.log_dir, 'results')
            assert not os.path.isdir(self.val_dir)
            os.makedirs(self.val_dir, exist_ok=True)
        
        if self.hparams.save_test_preds or self.hparams.save_train_preds:
            self.preds_save_dir = os.path.join(self.hparams.log_dir, 'preds')
            assert not os.path.isdir(self.preds_save_dir)
            os.makedirs(self.preds_save_dir, exist_ok=True)

    def validation_step(self, batch, batch_nb):
        w, h = self.test_dataset.img_wh
        img_id = batch['img_id']

        preds = self(batch, split='test')

        # Also extract normals from depth predictions
        #if self.hparams.pred_norm_depth:
        norm_depth = _extract_normals_from_depth_batch(
            depth=rearrange(preds['depth'].clone(), '(h w) -> 1 h w', h=h),
            ray_dirs_cc=batch['ray_dirs_cc'],
            poses=batch['pose'][:3, :3].unsqueeze(0)) # TODO Tmp just vie rotation amtix
        norm_depth = rearrange(norm_depth, '1 h w d-> (h w) d')
        norm_depth_np = norm_depth.detach().contiguous().cpu().numpy()
        preds['norm_depth'] = norm_depth 

        # # TODO <-------------
        # norm_depth_np = batch['normals'].detach().contiguous().cpu().numpy()
        # # TODO <-------------

        self.val_metrics.update(preds, batch, h, w)

        pred_dict = self._process_raw_pred_dict(preds, h, w)
        del preds

        pred_vis_all, pred_vis_dict = self._pack_vis_all(pred_dict)
        
        if self.hparams.save_test_vis: # save test image to disk
            # TODO Hardcoded, fix
            k_list = ['rgb', 'semantics', 'semantics_WF', 'normals', 'normals_depth', 'depth']
            batch_dict = {k: v for k, v in batch.items() if k in k_list}
            gt_dict = self._process_gt_pred_dict(batch_dict, h, w)
            if self.hparams.log_gt_test_vis:
                gt_vis_all = gt_dict.copy()
                if ('semantics' in gt_vis_all):
                    # Move void class to -1
                    gt_vis_all['semantics'] -= 1
                if ('semantics_WF' in gt_vis_all):
                    # Move void class to -1
                    gt_vis_all['semantics_WF'] -= 1
            gt_vis_all, gt_vis_dict = self._pack_vis_all(gt_vis_all)

            # imageio.imsave(os.path.join(self.val_dir, f'{img_id}_all.png'), 
            #                pred_vis_all)
            for k in pred_vis_dict:
                imageio.imsave(os.path.join(self.val_dir, f'{img_id}_{k}.png'), 
                            pred_vis_dict[k])
            for k in gt_vis_dict:
                imageio.imsave(os.path.join(self.val_dir, f'{img_id}_{k}_GT.png'), 
                            gt_vis_dict[k])

        output = {'pred_vis_all': pred_vis_all, 'img_id': img_id}

        if self.hparams.save_test_preds:
            output['pred_dict'] = self._downscale_dict_for_save(pred_dict)

        if self.hparams.save_test_preds or self.hparams.log_gt_test_vis:
            # TODO Hardcoded, fix
            k_list = ['rgb', 'semantics', 'semantics_WF', 'normals', 'normals_depth', 'depth']
            batch_dict = {k: v for k, v in batch.items() if k in k_list}
            gt_dict = self._process_gt_pred_dict(batch_dict, h, w)
            if self.hparams.log_gt_test_vis:
                gt_vis_all = gt_dict.copy()
                if ('semantics' in gt_vis_all):
                    # Move void class to -1
                    gt_vis_all['semantics'] -= 1
                if ('semantics_WF' in gt_vis_all):
                    # Move void class to -1
                    gt_vis_all['semantics_WF'] -= 1
                output['gt_vis_all'], _ = self._pack_vis_all(gt_vis_all)
            if self.hparams.save_test_preds:
                output['gt_dict'] = self._downscale_dict_for_save(gt_dict)
        
        output['norm_depth_np'] = norm_depth_np

        return output

    def validation_epoch_end(self, outputs):
        # Compute metrics
        logs = self.val_metrics.compute()
        #self.val_metrics.reset()
        # Angular error between the offset and the estimated correction
        if self.hparams.lr_dR_norm_glob > 0:
            raise NotImplementedError
            # # Offset to euler angles
            # euler_offset = matrix_to_euler_angles(self.R_offset, 'ZYX').cpu()
            # euler_offset *= (180.0/math.pi)
            # # dR to euler angles
            # dR_glob_norm = coding_to_matrix(self.dR_glob_norm_coded, self.hparams.dR_norm_glob_coding)
            # euler_glob_norm = matrix_to_euler_angles(dR_glob_norm, 'ZYX').cpu()
            # euler_glob_norm *= (180.0/math.pi)
            # # dtheta_correct = -dtheta_offset...thus add them
            # logs['ang/yaw_abs'] = abs(euler_offset[0].item() + euler_glob_norm[0].item())
            # if abs(euler_offset[0].item()) > 0:
            #     logs['ang/yaw_abs_norm'] = logs['ang/yaw_abs'] / abs(euler_offset[0].item())
            # else:
            #     logs['ang/yaw_abs_norm'] = -1.0
            # logs['ang/pitch_abs'] = abs(euler_offset[1].item() + euler_glob_norm[1].item())
            # if abs(euler_offset[1].item()) > 0:
            #     logs['ang/pitch_abs_norm'] = logs['ang/pitch_abs'] / abs(euler_offset[1].item())
            # else:
            #     logs['ang/pitch_abs_norm'] = -1.0
            # logs['ang/roll_abs'] = abs(euler_offset[2].item() + euler_glob_norm[2].item())
            # if abs(euler_offset[2].item()):
            #     logs['ang/roll_abs_norm'] = logs['ang/roll_abs'] / abs(euler_offset[2].item())
            # else:
            #     logs['ang/roll_abs_norm'] = -1.0
        
        print('Extracting rotation by clustering normals_depth....')
        # Angular error between the offset and canonical clusters
        norm_depth_np = [x['norm_depth_np'] for x in outputs]
        norm_depth_np = np.concatenate(norm_depth_np, axis=0)
        # Remove (0,0,0) invalid normals
        norm_depth_np = norm_depth_np[norm_depth_np.sum(-1)!=0.0]
        clust_ass_new, clust_ass_orig, centrs_new = _normals_clustering(
                                                        norm_depth_np, 
                                                        device=self.device,
                                                        K=30,
                                                        niter=30,
                                                        t_similar=0.99, 
                                                        merge_clusters=True, 
                                                        find_opposite=False)
        # Rotation matrix is composed of rotation vectors as columns
        centrs_orthog = centrs_new.T
        # Arbitrary direction was chosen
        centrs_orthog = torch.concat([centrs_orthog, -1.0*centrs_orthog], dim=1)
        # Pick cluster directions closest to the rotation vectors
        # (there is always ambiguity for errors > 45deg)
        sim =(self.R_offset.to(self.device).unsqueeze(1) * centrs_orthog.unsqueeze(-1)).sum(0)
        cluster_rot = centrs_orthog[:, torch.argmax(sim, dim=0)]
        # Project to so3
        cluster_rot = scipy_Rotation.from_matrix(cluster_rot.detach().cpu().numpy())
        cluster_rot = torch.from_numpy(cluster_rot.as_matrix())
        # U, S, V = torch.svd(cluster_rot)
        # cluster_rot = U @ V.T
        # Print retrieved rotation 
        print('Extracted rotation from normals_depth:')
        for row in cluster_rot.detach().cpu().tolist():
            print([float(f'{x:.4f}') for x in row])

        # Offset to euler angles
        offset_euler = matrix_to_euler_angles(self.R_offset, 'ZYX').cpu()
        offset_euler *= (180.0/math.pi)
        cluster_euler = matrix_to_euler_angles(cluster_rot, 'ZYX').cpu()
        cluster_euler *= (180.0/math.pi)
        logs['ang/clust/yaw_abs'] = abs(offset_euler[0].item() - cluster_euler[0].item())
        logs['ang/clust/pitch_abs'] = abs(offset_euler[1].item() - cluster_euler[1].item())
        logs['ang/clust/roll_abs'] = abs(offset_euler[2].item() - cluster_euler[2].item())

        # Print metric
        self._print_metrics(logs)

        # Save results in a .csv file
        self._save_results_to_csv(logs)

        # Repack outputs
        img_ids = [x['img_id'] for x in outputs]

        # Log images to wandb
        self._log_wandb_preds(outputs, img_ids)
        self._log_wandb_gt(outputs, img_ids)

        # Store all rendered test images and labels for further use
        self._save_test_preds(outputs, img_ids)

        # Store all rendered train images and labels for further use
        self._save_train_preds()

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
       
    def _process_raw_pred_dict(self, pred_dict, h, w):
        pred_dict_new = {}
        for k in pred_dict:
            if k in ['total_samples']: #, 'opacity'
                continue
            pred_dict_new[k] = NeRFSystem._vec_to_spat(pred_dict[k], h, w)
            pred_dict_new[k] = NeRFSystem._process_pred_raw(pred_dict_new[k], k)
                    
        return pred_dict_new   

    def _process_gt_pred_dict(self, gt_dict, h, w):
        for k in gt_dict:
            gt_dict[k] = NeRFSystem._vec_to_spat(gt_dict[k], h, w)
            gt_dict[k] = NeRFSystem._process_pred_raw(gt_dict[k], k)
        return gt_dict
        
    def _pack_vis_all(self, pred_dict):
        w, h = self.test_dataset.img_wh
        pred_vis_all = None
        pred_vis_dict = {}
        for k in sorted(list(pred_dict.keys())):
            pred_vis = pred_dict[k].copy()
            resize_shape = (int(w*self.hparams.downsample_vis), int(h*self.hparams.downsample_vis))
            pred_vis = NeRFSystem._downscale_pred(pred_vis, k, resize_shape)
            pred_vis = self._pred_to_vis_format(pred_vis, k)
            pred_vis_dict[k] = pred_vis
            pred_vis_all = self._update_pred_all(pred_vis_all, pred_vis)
        return pred_vis_all, pred_vis_dict

    @staticmethod
    def norm_to_unit_np(pred):
        return pred / (np.linalg.norm(pred, ord=2, axis=-1, keepdims=True) + 1e-12)

    @staticmethod
    def _vec_to_spat(pred, h, w):
        if len(pred.shape) == 1:
            rearange_code = '(h w) -> h w' 
        elif len(pred.shape) == 2:
            rearange_code = '(h w) c -> h w c'
        else:
            raise AssertionError
        pred = rearrange(pred.cpu().numpy(), rearange_code, h=h)
        return pred
        
    @staticmethod
    def _process_pred_raw(pred, which_task):
        if which_task in ['sem', 'semantics', 'sem_WF', 'semantics_WF']:
            if not (pred.dtype == np.int8 or pred.dtype == np.int16 or \
                pred.dtype == np.int32 or pred.dtype == np.int64 or \
                pred.dtype == np.uint8):
                assert len(pred.shape) == 3
                pred = np.argmax(pred, axis=-1)
        elif which_task in ['norm_nn', 'normals', 'norm_depth', 'normals_depth']:
            normalized = pred / np.linalg.norm(pred, ord=2, axis=-1, keepdims=True)
            pred = np.where((np.sum(np.abs(pred), -1) == 0.0)[...,None], pred, normalized)
        elif which_task == 'depth':
            pass
        elif which_task == 'rgb':
            pass
        elif which_task == 'opacity':
            pass
        else:
            raise NotImplementedError
        return pred    
    
    @staticmethod
    def _downscale_pred(pred, which_task, resize_shape):
        if which_task == 'rgb':
            pred = cv2.resize(pred, dsize=resize_shape, 
                                interpolation=cv2.INTER_LINEAR)
        elif which_task == 'depth':
            pred = cv2.resize(pred, dsize=resize_shape, 
                                interpolation=cv2.INTER_LINEAR)
        elif which_task in ['norm_nn', 'norm_depth', 'normals', 'normals_depth']:
            pred = cv2.resize(pred, dsize=resize_shape, 
                                interpolation=cv2.INTER_LINEAR)
            pred = NeRFSystem.norm_to_unit_np(pred)
        elif which_task in ['sem', 'semantics', 'sem_WF', 'semantics_WF']:
            pred = cv2.resize(pred, dsize=resize_shape, 
                                interpolation=cv2.INTER_NEAREST)
        elif which_task == 'opacity':
            pred = cv2.resize(pred, dsize=resize_shape, 
                                interpolation=cv2.INTER_LINEAR)
        else:
            raise NotImplementedError
        return pred

    def _downscale_dict_for_save(self, pred_dict):
        for k in pred_dict:
            h = pred_dict[k].shape[0]; w = pred_dict[k].shape[1];
            downsample = self.hparams.downsample_pred_save
            resize_shape = (int(w*downsample), int(h*downsample))
            pred_dict[k] = NeRFSystem._downscale_pred(pred_dict[k], k, resize_shape)
        return pred_dict

    def _pred_to_vis_format(self, pred, which_task):
        # # Body diagonal of a cubge
        # max_depth = hparams.scale*SQRT3
        if which_task == 'depth':
            # max is big diagonal of unit cube = sqrt(3)
            pred_vis = depth2img(pred, min=0.0, max=1.74)
        elif which_task in ['norm_nn', 'normals', 'norm_depth', 'normals_depth']:
            pred_vis = (pred + 1.0) / 2.0
            pred_vis = (pred_vis*255).astype(np.uint8)
        elif which_task in ['sem', 'semantics', 'sem_WF', 'semantics_WF']:
            sem_colormap = label_colormap(self.train_dataset.n_classes)
            pred_vis = sem_colormap[pred]
        elif which_task == 'rgb':
            pred_vis = (pred*255).astype(np.uint8)
        elif which_task == 'opacity':
            pred_vis = (pred*255).astype(np.uint8)
            pred_vis = np.repeat(pred_vis[..., np.newaxis], 3, axis=-1)
        else:
            raise NotImplementedError
        return pred_vis

    @staticmethod
    def _update_pred_all(pred_all, pred):
        if pred_all is not None:
            return np.concatenate((pred_all, pred), axis=1)
        else:
            return pred

    def _print_metrics(self, logs):
        print('Test metrics:')
        for name in logs:
            display = True if name == 'rgb/psnr' else False
            self.log(f'test/{name}', logs[name], prog_bar=display)
            print(f'{name}: {logs[name]:.4f}')
    
    def _save_results_to_csv(self, logs):
        val_csv_output = os.path.join(hparams.log_dir, 'results.csv')
        print(f'Saving results in a .csv file {val_csv_output}...')
        assert not os.path.isfile(val_csv_output)
        header = []
        data = []
        for k, v in logs.items():
            header.append('metric/'+k)
            data.append(v)
        header += ['info/epoch', 'info/step']
        data+= [self.current_epoch, self.global_step]
        for k, v in self.hparams.items():
            if not (('dir' in k) or ('path' in k) or ('eval' in k)):
                header.append('param/'+k)
                data.append(v)

        with open(val_csv_output, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(data)
    
    def _get_wandb_logger(self):
        logger = None
        loggers = getattr(self, 'loggers', None)
        if loggers is not None:
            for x in loggers:
                if isinstance(x, pytorch_lightning.loggers.wandb.WandbLogger):
                    logger = x
        return logger

    def _log_wandb_preds(self, outputs, img_ids):
        # Load wandb logger if exists
        wandb_logger = self._get_wandb_logger()
        if wandb_logger is not None:
            print('Logging predictions into wandb')
            pred_vis_all = [x['pred_vis_all'] for x in outputs]
            # Log all predictions jointly
            wandb_logger.log_image(key='vis/test/pred', images=pred_vis_all,
                                    caption=img_ids)
        
    def _log_wandb_gt(self, outputs, img_ids):
        # Load wandb logger if exists
        wandb_logger = self._get_wandb_logger()
        if (wandb_logger is not None) and self.hparams.log_gt_test_vis:
            print('Logging GT in wandb...')
            w, h = self.test_dataset.img_wh
            gt_vis_all = [x['gt_vis_all'] for x in outputs]
            # Log all predictions jointly
            wandb_logger.log_image(key='vis/test/gt', images=gt_vis_all,
                                    caption=img_ids)

    def _save_test_preds(self, outputs, img_ids):
        if hparams.save_test_preds:
            print('Saving test predictions...')
            pred_dict = {k: [x['pred_dict'][k] for x in outputs] 
                        for k in outputs[0]['pred_dict'].keys()}
            gt_dict = {k: [x['gt_dict'][k] for x in outputs] 
                        for k in outputs[0]['gt_dict'].keys()}

            self._save_preds_tar_gz(pred_dict, img_ids, "test", "pred")
            self._save_preds_tar_gz(gt_dict, img_ids, "test", "gt")
    
    def _save_train_preds(self):
        if hparams.save_train_preds:
            print('Saving train predictions (rendering them first)...')
            self.train_dataset.split = 'eval_train'
            w, h = self.train_dataset.img_wh
            # Predict for all and pack in dict
            pred_list = []
            gt_list = []
            img_ids = []
            for idx in range(self.train_dataset.__len__()):
                batch = self.train_dataset.__getitem__(idx)
                img_ids.append(batch['img_id'])
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.device)

                preds = self(batch, split='test')
                pred_dict = self._process_raw_pred_dict(preds, h, w)
                pred_dict = self._downscale_dict_for_save(pred_dict)
                
                # TODO Hardcoded, fix
                k_list = ['rgb', 'semantics', 'normals', 'depth']
                batch_dict = {k: v for k, v in batch.items() if k in k_list}
                gt_dict = self._process_gt_pred_dict(batch_dict, h, w)
                gt_dict = self._downscale_dict_for_save(gt_dict)

                pred_list.append(pred_dict)
                gt_list.append(gt_dict)
            pred_dict = {k: [x[k] for x in pred_list] for k in pred_list[0].keys()}
            gt_dict = {k: [x[k] for x in gt_list] for k in gt_list[0].keys()}

            self._save_preds_tar_gz(pred_dict, img_ids, "train", "pred")
            self._save_preds_tar_gz(gt_dict, img_ids, "train", "gt")

    def _save_preds_tar_gz(self, save_dict, img_ids, which_split, tag):
        tar_fname = f'{which_split}_{tag}'
        tar_name_k = os.path.join(self.preds_save_dir, f'{tar_fname}.tar.gz')
        assert not os.path.isfile(tar_name_k)
        with tarfile.open(tar_name_k, "w:gz") as tar_handle:
            for k in save_dict:
                if k in ['opacity']:
                    continue
                k_name = k
                k_name = 'semantics' if k_name == 'sem' else k_name
                k_name = 'normals' if k_name == 'norm' else k_name
                for pred_k_i, img_id in zip(save_dict[k], img_ids):
                    b = io.BytesIO()
                    np.save(b, pred_k_i)
                    b.seek(0)
                    fname = f'{tag}.{which_split}.{k_name}.{hparams.scene_name}.{img_id}.npy'
                    tarinfo = tarfile.TarInfo(name=fname)
                    #tarinfo.size = b.getbuffer().shape[0]
                    tarinfo.size = len(b.getvalue())
                    tar_handle.addfile(tarinfo=tarinfo, fileobj=b)
                    b.close()
        
        # Create a file to confirm that its done
        with open(os.path.join(self.preds_save_dir, f'{tar_fname}.done'), 'w') as fp:
            pass


if __name__ == '__main__':
    hparams = get_opts()

    MARK_INVISIBLE_CHUNK = 64**3
    # DEBUG mode
    if not hparams.no_debug:
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        hparams.batch_size = 256
        hparams.grid_size = 32
        MARK_INVISIBLE_CHUNK = 16**3
        hparams.downsample = 1.0 #/ 8.0
        root = '/home/nipopovic/MountedDirs/hl_task_prediction/hl_task_pred_root'
        hparams.data_root_dir = f'{root}/data/data_sets/Hypersim/ai_001_001'
        hparams.log_root_dir = f'{root}/data/nikola/experiment_logs/ngp_mt/_debug'
        hparams.num_epochs = 2
        hparams.exp_name = '_debug' + hparams.exp_name + ''.join(random.choice(string.ascii_letters) for i in range(6)) 

        hparams.load_depth_gt = True
        hparams.load_norm_gt = False
        hparams.load_norm_depth_gt = False
        hparams.load_sem_gt = False
        hparams.load_sem_WF_gt = True
        hparams.pred_norm_nn = True
        #hparams.pred_norm_nn_norm = True
        hparams.pred_norm_depth = True
        hparams.pred_sem = True

        hparams.tqdm_refresh_rate = 1
        hparams.ray_sampling_strategy = 'all_images_triang'
        hparams.random_tr_poses = True
        hparams.triang_max_expand=0

        hparams.loss_distortion_w = 0.1
        hparams.loss_norm_can_tres = 0.01
        hparams.loss_norm_can_start = 1
        hparams.loss_norm_can_end = -1
        hparams.loss_norm_can_grow = 1

        hparams.loss_depth_w = 0.1
        hparams.loss_reg_depth_w = 0.1

        hparams.loss_norm_D_C_ort_dot_w = 1e-3
        hparams.loss_norm_D_C_centr_dot_w = 1e-3
        hparams.loss_norm_D_C_centr_L1_w = 1e-3

        hparams.loss_manhattan_nerf_w = 0.1
        hparams.loss_sem_w = 0.1

        hparams.loss_norm_yaw_offset_ang = 15.0
        hparams.loss_norm_pitch_offset_ang = 10.0
        hparams.loss_norm_roll_offset_ang = 20.0
        hparams.lr_dR_norm_glob = 0.0

        hparams.pl_log_every_n_steps = 1
        hparams.model_name = 'NGPMT'
        hparams.density_tresh_decay=1.0
        hparams.save_test_vis = True
        hparams.keep_N_tr = -1

        
    # Create log dir
    if not os.path.isdir(hparams.log_root_dir):
        raise NotADirectoryError
    rnd_l = 6
    scene_name = os.path.basename(hparams.data_root_dir)
    hparams.scene_name = scene_name
    hparams.exp_name += '_' + scene_name if hparams.exp_name else scene_name
    # hparams.exp_name += '_' + ''.join(random.choice(string.ascii_letters) for i in range(rnd_l)) 
    hparams.log_dir = os.path.join(hparams.log_root_dir, hparams.exp_name)
    # while os.path.isdir(hparams.log_dir):
    #     rnd_str = ''.join(random.choice(string.ascii_letters) for i in range(rnd_l))
    #     hparams.exp_name = hparams.exp_name[:-rnd_l] + rnd_str
    #     hparams.log_dir = os.path.join(hparams.log_root_dir, hparams.exp_name)
    if not os.path.isdir(hparams.log_dir):
        os.mkdir(hparams.log_dir)
    hparams.exp_name = f'{os.path.basename(hparams.log_root_dir)}/{hparams.exp_name}'

    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    system = NeRFSystem(hparams)

    if hparams.save_checkpoint:
        ckpts_dir = os.path.join(hparams.log_dir, 'ckpts')
        ckpt_cb = ModelCheckpoint(dirpath=ckpts_dir,
                                filename='{epoch:d}',
                                save_weights_only=True,
                                every_n_epochs=hparams.num_epochs,
                                save_on_train_epoch_end=True,
                                save_top_k=-1)
        callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=hparams.tqdm_refresh_rate)]
    else:
        callbacks = [TQDMProgressBar(refresh_rate=hparams.tqdm_refresh_rate)]

    logger = []
    tb_logger = TensorBoardLogger(
        save_dir=hparams.log_dir,
        name='tb_logs',
        default_hp_metric=False
    )
    logger.append(tb_logger)

    # Euler cluster solution for not saving a lot of files:
    # Feed local wandb log into some tmpdir, 
    # which will be automatically deleted after the job ends
    if 'TMPDIR' in os.environ:
        assert os.path.isdir(os.environ['TMPDIR'])
        wandb_log_dir = os.path.join(os.environ['TMPDIR'], 'wandb_log_dir_TMP')
        if not os.path.isdir(wandb_log_dir):
            os.mkdir(wandb_log_dir)
        print(f'Storing wandb local files in a TMPDIR instead:{wandb_log_dir}')
    else:
        wandb_log_dir = hparams.log_dir
        assert os.path.isdir(wandb_log_dir)

    wandb_use = False
    if hparams.no_debug: 
        # TODO
        # Euler fix for not finding the port due to network slowness 
        # https://github.com/wandb/wandb/issues/3911#issuecomment-1204961296     
        wandb_use = True     
        wandb_logger = WandbLogger(
            entity="nikola3794",
            project='nerf-multi-task',
            name=hparams.exp_name,
            save_dir=wandb_log_dir,
            id=None,
            offline=True, # TODO <----- Disable logging while the experiment is running and sync at the end
        )

        logger.append(wandb_logger)

    # log_every_n_steps problems:
    # wandb is rate limited to about 200 requests per minute
    # https://docs.wandb.ai/guides/track/limits#rate-limits
    # With the assumption of 55 processes which do 30 it/sec (1800 it/min)
    # logging with log_every_n_steps=500 should be slightly below the rate limit.
    trainer = Trainer(max_epochs=hparams.num_epochs,
                    check_val_every_n_epoch=hparams.num_epochs,
                    callbacks=callbacks,
                    logger=logger,
                    enable_model_summary=False,
                    accelerator='gpu',
                    devices=hparams.num_gpus,
                    strategy=DDPPlugin(find_unused_parameters=False)
                            if hparams.num_gpus>1 else None,
                    num_sanity_val_steps=-1 if hparams.val_only else 0,
                    precision=16,
                    gradient_clip_val=hparams.grad_clip,
                    log_every_n_steps=hparams.pl_log_every_n_steps,
                    enable_checkpointing=hparams.save_checkpoint, # TODO <----- disabled checkpoints
                    )

    trainer.fit(system, ckpt_path=hparams.ckpt_path)

    if wandb_use:
        sync_command = f'cd {wandb_log_dir}; wandb online; wandb sync --sync-all .'
        print(f'Syncing wandb with: {sync_command}')
        os.system(sync_command)

    print("Finished.")



