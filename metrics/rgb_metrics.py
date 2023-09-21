import torch
import torchmetrics
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


from skimage.metrics import structural_similarity as calculate_ssim


class PSNRPerImg(torchmetrics.Metric):
    
    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state(
            "psnr_sum", 
            default=torch.tensor(0.0), 
            dist_reduce_fx="sum"
        )
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")

        self.psnr = PeakSignalNoiseRatio(data_range=1)
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.dim() == 2
        assert targets.dim() == 2
        self.psnr(preds, targets)
        self.psnr_sum += self.psnr.compute()
        self.psnr.reset()
        self.n += 1
    
    def compute(self):
        return (self.psnr_sum / self.n).item()

        

class SSIMPerImg(torchmetrics.Metric):

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state(
            "ssim_sum", 
            default=torch.tensor(0.0), 
            dist_reduce_fx="sum"
        )
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1)
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.dim() == 4
        assert targets.dim() == 4
        self.ssim(preds, targets)
        self.ssim_sum += self.ssim.compute()
        self.ssim.reset()
        self.n += 1
    
    def compute(self):
        return (self.ssim_sum / self.n).item()
        
class SSIMPerImgNorm(torchmetrics.Metric):

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state(
            "ssim_sum", 
            default=torch.tensor(0.0), 
            dist_reduce_fx="sum"
        )
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")

        #self.ssim = StructuralSimilarityIndexMeasure(data_range=1)
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.dim() == 4
        assert targets.dim() == 4
        ssim = StructuralSimilarityIndexMeasure(data_range=targets.max()-targets.min())
        ssim(preds, targets)
        self.ssim_sum += ssim.compute()
        ssim.reset()
        self.n += 1
    
    def compute(self):
        return (self.ssim_sum / self.n).item()    

class SSIMPerImgNormSckit(torchmetrics.Metric):

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state(
            "ssim_sum", 
            default=torch.tensor(0.0), 
            dist_reduce_fx="sum"
        )
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.dim() == 4
        assert targets.dim() == 4
        '''image size: (h, w, 3)'''
        gt = targets[0].clone().permute(1, 2, 0).cpu().numpy()
        pred = preds[0].clone().permute(1, 2, 0).cpu().numpy()
        ssim = calculate_ssim(pred, gt, data_range=gt.max() - gt.min(), multichannel=True)
        self.ssim_sum += ssim
        self.n += 1
    
    def compute(self):
        return (self.ssim_sum / self.n).item()


class LPIPSPerImg(torchmetrics.Metric):
    
    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = False
    
    def __init__(self):
        super().__init__()
        self.add_state(
            "lpips_sum", 
            default=torch.tensor(0.0), 
            dist_reduce_fx="sum"
        )
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")

        self.lpips = LearnedPerceptualImagePatchSimilarity('vgg')
        for p in self.lpips.net.parameters():
            p.requires_grad = False
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.dim() == 4
        assert targets.dim() == 4
        self.lpips(preds, targets)
        self.lpips_sum += self.lpips.compute()
        self.lpips.reset()
        self.n += 1
    
    def compute(self):
        return (self.lpips_sum / self.n).item()