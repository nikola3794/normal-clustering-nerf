import torch
import torchmetrics
from torchmetrics.functional import confusion_matrix


class SemanticMetrics(torchmetrics.Metric):

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = False

    def __init__(self, n_classes, ignore_label):
        super().__init__()
        self.ignore_label = ignore_label
        self.n_classes = n_classes
        self.add_state(
            "conf_mat", 
            default=torch.zeros((n_classes, n_classes), dtype=torch.long), 
            dist_reduce_fx="sum"
        )

    def update(self, pred_logits: torch.Tensor, target_labels: torch.Tensor):
        if (target_labels == self.ignore_label).all():
            return

        assert target_labels.dim() == 1
        assert pred_logits.dim() == 2

        pred_labels = torch.argmax(pred_logits, dim=1)
        valid_idx = target_labels!=self.ignore_label

        if (valid_idx.numel() > 0) and (True in valid_idx):
            pred_labels = pred_labels[valid_idx] 
            target_labels = target_labels[valid_idx]
            self.conf_mat += confusion_matrix(
                preds=pred_labels,
                target=target_labels,
                num_classes=self.n_classes,
            )
    
    def compute(self):
        norm_conf_mat = (self.conf_mat.T / self.conf_mat.type(torch.float).sum(dim=1)).T

        # missing class will have NaN at corresponding class
        missing_class_mask = torch.isnan(norm_conf_mat.sum(1)) 
        exsiting_class_mask = ~ missing_class_mask

        cls_avg_acc = torch.nanmean(torch.diagonal(norm_conf_mat))
        total_acc = (torch.sum(torch.diagonal(self.conf_mat)) / torch.sum(self.conf_mat))
        ious = torch.zeros(self.n_classes)
        for class_id in range(self.n_classes):
            ious[class_id] = (self.conf_mat[class_id, class_id] / (
                    torch.sum(self.conf_mat[class_id, :]) + torch.sum(self.conf_mat[:, class_id]) -
                    self.conf_mat[class_id, class_id]))
        miou = torch.nanmean(ious)
        miou_valid_cls = torch.mean(ious[exsiting_class_mask])

        metrics = {
            'miou': miou.item(),
            'miou_valid_cls': miou_valid_cls.item(),
            'total_acc': total_acc.item(),
            'cls_avg_acc': cls_avg_acc.item(),
            'ious': ious.tolist(),
        }
        return metrics
rog_bar=True