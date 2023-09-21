import torch
from torch import nn
import torchmetrics


class RMSEPerImg(torchmetrics.Metric):

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state(
            "rmse_sum", 
            default=torch.tensor(0.0), 
            dist_reduce_fx="sum"
        )
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")

        self.mse = nn.MSELoss(reduction='mean')

    def _rmse(self, x: torch.Tensor, y: torch.Tensor):
        return torch.sqrt(self.mse(x, y))
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        valid_idx = (targets > 0)
        if (valid_idx.numel() > 0) and (True in valid_idx):
            self.rmse_sum += self._rmse(preds[valid_idx], targets[valid_idx])
            self.n += 1
    
    def compute(self):
        return (self.rmse_sum / self.n).item()


class AbsPerImg(torchmetrics.Metric):
    
    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state(
            "abs_sum", 
            default=torch.tensor(0.0), 
            dist_reduce_fx="sum"
        )
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")

        self.l1 = nn.L1Loss(reduction='mean')

    def _abs(self, x: torch.Tensor, y: torch.Tensor):
        return self.l1(x, y)
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        valid_idx = (targets > 0)
        if (valid_idx.numel() > 0) and (True in valid_idx):
            self.abs_sum += self._abs(preds[valid_idx], targets[valid_idx])
            self.n += 1
    
    def compute(self):
        return (self.abs_sum / self.n).item()