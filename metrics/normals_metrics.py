import math
import torch
import torch.nn.functional as F
import torchmetrics


class AngularErrorDegPerImg(torchmetrics.Metric):

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = False

    def __init__(self, tag=''):
        super().__init__()
        self.tag = tag
        self.add_state(
            "ang_err_mean", 
            default=torch.tensor(0.0), 
            dist_reduce_fx="sum")
        self.add_state(
            "ang_err_mean_min", 
            default=torch.tensor(0.0), 
            dist_reduce_fx="sum")
        self.add_state(
            "ang_err_median", 
            default=torch.tensor(0.0), 
            dist_reduce_fx="sum")
        self.add_state(
            "ang_err_median_min", 
            default=torch.tensor(0.0), 
            dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")

        self.rad_to_deg = (180.0/math.pi)

    @staticmethod
    def _to_unit(x: torch.Tensor):
        return x / (x.norm(p=2, dim=-1, keepdim=True).detach() + 1e-8)

    @staticmethod
    def _dot_prod(x: torch.Tensor, y: torch.Tensor):
        return torch.sum((x * y), dim=-1)

    def _angular_errors(self, x: torch.Tensor, y: torch.Tensor):
        x_norm = self._to_unit(x)
        y_norm = self._to_unit(y)
        dot_prod = self._dot_prod(x_norm, y_norm)
        dot_prod_clamp =  torch.clamp(dot_prod, -1, 1)
        ang_errs_rad = torch.acos(dot_prod_clamp)
        ang_errs_deg = ang_errs_rad * self.rad_to_deg

        return ang_errs_deg

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.dim() == 2
        assert targets.dim() == 2
        valid_idx = (targets.abs().sum(-1) > 0)
        if (valid_idx.numel() > 0) and (True in valid_idx):
            ang_errs_deg = self._angular_errors(preds[valid_idx], targets[valid_idx])
            ang_errs_deg_flip = self._angular_errors(-1.0*preds[valid_idx], targets[valid_idx])
            ang_errs_deg_min = torch.minimum(ang_errs_deg, ang_errs_deg_flip)
            self.ang_err_mean += torch.mean(ang_errs_deg)
            self.ang_err_median += torch.median(ang_errs_deg)
            self.ang_err_mean_min += torch.mean(ang_errs_deg_min)
            self.ang_err_median_min += torch.median(ang_errs_deg_min)
            self.n += 1
    
    def compute(self):
        return {
            f'norm_{self.tag}/ang_err_mean': (self.ang_err_mean / self.n).item(),
            f'norm_{self.tag}/ang_err_median': (self.ang_err_median / self.n).item(),
            f'norm_{self.tag}/ang_err_mean_min': (self.ang_err_mean_min / self.n).item(),
            f'norm_{self.tag}/ang_err_median_min': (self.ang_err_median_min / self.n).item(),
        }