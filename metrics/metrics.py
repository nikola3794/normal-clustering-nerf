import torch
from einops import rearrange

import torchmetrics

from metrics.normals_metrics import AngularErrorDegPerImg

from .rgb_metrics import PSNRPerImg, SSIMPerImg, SSIMPerImgNorm, SSIMPerImgNormSckit, LPIPSPerImg
from .depth_metrics import RMSEPerImg, AbsPerImg
from .normals_metrics import AngularErrorDegPerImg
from .semantic_metrics import SemanticMetrics


class NeRFMTMetricsPerIm(torchmetrics.Metric):
    @torch.no_grad()
    def __init__(self, hparams_dict, **kwargs):
        super().__init__()

        # RGB
        self.psnr_per_im = PSNRPerImg()
        self.ssim_per_im = SSIMPerImg()
        self.ssim_per_im_norm = SSIMPerImgNorm()
        self.ssim_per_im_norm_sckit = SSIMPerImgNormSckit()
        self.eval_lpips = hparams_dict['eval_lpips']
        if self.eval_lpips:
            self.lpips_per_im = LPIPSPerImg()

        # Depth
        self.eval_depth = hparams_dict['load_depth_gt']
        if self.eval_depth:
            self.rmse_per_im = RMSEPerImg()
            self.abs_per_img = AbsPerImg()

        # Normals
        self.norm_gt = hparams_dict['load_norm_gt']
        self.pred_norm_nn = hparams_dict['pred_norm_nn']
        if self.pred_norm_nn and self.norm_gt:
            self.angular_error_per_im_nn = AngularErrorDegPerImg(tag='nn')
        self.pred_norm_depth = True
        if self.pred_norm_depth and self.norm_gt:
            self.angular_error_per_im_depth = AngularErrorDegPerImg(tag='depth')


        self.eval_sem = hparams_dict['pred_sem'] and hparams_dict['load_sem_gt']
        if self.eval_sem:
            self.sem_metrics = SemanticMetrics(
                n_classes=kwargs['n_sem_cls'], 
                ignore_label=-1
            )

    @torch.no_grad()
    def update(self, preds: dict, target: dict, h: int, w: int):
        '''Update each metric per image.'''

        # RGB
        self.psnr_per_im.update(preds['rgb'], target['rgb'])
        rgb_pred = rearrange(preds['rgb'], '(h w) c -> 1 c h w', h=h)
        rgb_gt = rearrange(target['rgb'], '(h w) c -> 1 c h w', h=h)
        # # TODO
        # rgb_pred = torch.clip(rgb_pred, min=0.0, max=1.0)
        # rgb_gt = torch.clip(rgb_gt, min=0.0, max=1.0)
        # # TODO
        self.ssim_per_im.update(rgb_pred, rgb_gt)
        self.ssim_per_im_norm.update(rgb_pred, rgb_gt)
        self.ssim_per_im_norm_sckit.update(rgb_pred, rgb_gt)
        if self.eval_lpips:
            self.lpips_per_im.update(
                torch.clip(rgb_pred*2-1, -1, 1), 
                torch.clip(rgb_gt*2-1, -1, 1)
            )

        # Depth
        if self.eval_depth:
            self.rmse_per_im.update(preds['depth'], target['depth'])
            self.abs_per_img.update(preds['depth'], target['depth'])

        # Normals
        if self.pred_norm_nn and self.norm_gt:
            self.angular_error_per_im_nn.update(preds['norm_nn'], target['normals'])
        if self.pred_norm_depth and self.norm_gt:
            self.angular_error_per_im_depth.update(preds['norm_depth'], target['normals'])

        # Semantics
        if self.eval_sem:
            # TODO Substract 1 to move void class to -1
            self.sem_metrics.update(preds['sem'], target['semantics'] - 1)

    @torch.no_grad()    
    def compute(self):
        logs = {}

        # RGB
        logs['rgb/psnr'] = self.psnr_per_im.compute()
        logs['rbg/ssim'] = self.ssim_per_im.compute()
        logs['rbg/ssim_n'] = self.ssim_per_im_norm.compute()
        logs['rbg/ssim_n_sck'] = self.ssim_per_im_norm_sckit.compute()
        if self.eval_lpips:
            logs['rgb/lpips'] = self.lpips_per_im.compute()
        
        # Depth
        if self.eval_depth:
            logs['depth/rmse'] = self.rmse_per_im.compute()
            logs['depth/abs'] = self.abs_per_img.compute()

        # Normals
        if self.pred_norm_nn and self.norm_gt:
            logs.update(self.angular_error_per_im_nn.compute())
        if self.pred_norm_depth and self.norm_gt:
            logs.update(self.angular_error_per_im_depth.compute())

        # Semantics
        if self.eval_sem:
            sem_met = self.sem_metrics.compute()
            logs['sem/miou'] = sem_met['miou']
            logs['sem/miou_valid_cls'] = sem_met['miou_valid_cls']
            logs['sem/total_acc'] = sem_met['total_acc']
            logs['sem/cls_avg_acc'] = sem_met['cls_avg_acc']
            #logs['sem/ious'] = sem_met['ious']

        return logs

    @torch.no_grad() 
    def reset(self):
        # Needs to be impelmented in members
        raise NotImplementedError
        for child in self.children():
            if isinstance(child, torchmetrics.Metric):
                child.reset()