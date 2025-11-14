import torch
import numpy as np
import torch.distributed as dist
from torchmetrics import Metric
from typing import List, Optional

class BaseIoUMetric(torch.nn.Module):
    def __init__(self, thresholds=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]):
        super().__init__()

        thresholds = np.array(thresholds, dtype=np.float32)
        self.register_buffer('thresholds', torch.from_numpy(thresholds))
        self.register_buffer('tp', torch.zeros(len(thresholds)))
        self.register_buffer('fp', torch.zeros(len(thresholds)))
        self.register_buffer('fn', torch.zeros(len(thresholds)))

    def update(self, pred, label, isLogit=True):
        if isLogit: 
            pred = pred.detach().float().sigmoid().reshape(-1)
            thresholds = self.thresholds.to(pred.device)
            pred = pred[:, None] >= thresholds[None]
        else: 
            pred = pred.detach().float().reshape(-1)
            pred = pred[:, None]

        label = label.detach().bool().reshape(-1)
        label = label[:, None]

        self.tp += (pred & label).sum(0).to(self.tp.device)
        self.fp += (pred & ~label).sum(0).to(self.fp.device)
        self.fn += (~pred & label).sum(0).to(self.fn.device)

    def compute(self):
        # All-reduce for DDP: sum across all ranks
        if dist.is_initialized():
            assert self.tp.is_cuda, "Tensors must be on CUDA before all_reduce"
            dist.all_reduce(self.tp, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.fp, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.fn, op=dist.ReduceOp.SUM)

        ious = self.tp / (self.tp + self.fp + self.fn + 1e-7)

        return {f'@{t.item():.2f}': i.item() for t, i in zip(self.thresholds, ious)}


class IoUMetric(BaseIoUMetric):
    def __init__(self, label_indices: List[List[int]],
                 min_visibility: Optional[int] = None,
                 max_visibility: Optional[int] = None,
                 target_class: Optional[str] = None):
        """
        label_indices:
            transforms labels (c, h, w) to (len(labels), h, w)
            see config/experiment/* for examples

        min_visibility:
            passing "None" will ignore the visibility mask
            otherwise uses visibility values to ignore certain labels
            visibility mask is in order of "increasingly visible" {1, 2, 3, 4, 255 (default)}
            see https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md#visibility
        """
        super().__init__()

        self.label_indices = label_indices
        self.min_visibility = min_visibility
        self.max_visibility = max_visibility
        self.target_class = target_class

    def update(self, pred, batch, label=None):
        if label is None :
            label = batch['bev']                                                               # b 12 h w
            label = [label[:, idx].max(1, keepdim=True).values for idx in self.label_indices]  
            label = torch.cat(label, 1)    # b c h w
            image_visibility = False
        else:                                                    
            image_visibility = True

        if self.min_visibility is not None or self.max_visibility is not None:
            if (self.target_class == 'vehicle'): 
                if image_visibility:
                    visibility = batch['img_visibility'][:, :1]
                else:
                    #visibility = batch['visibility'][:, [0]]
                    visibility = batch['visibility'][:, :1] # update 250104
            elif (self.target_class == 'pedestrian'): 
                if image_visibility:
                    visibility = batch['img_visibility'][:, 1:]
                else:
                    #visibility = batch['visibility'][:, [1]]
                    visibility = batch['visibility'][:, 1:] # update 250104
                    
            if self.min_visibility is not None:
                mask = visibility >= self.min_visibility
            
            elif self.max_visibility is not None:
                mask = (visibility < self.max_visibility) | (visibility == 255)
                
            mask = mask.expand_as(pred) # b c h w
            indices = mask.nonzero(as_tuple=True)

            pred = pred[indices]   # m
            label = label[indices]  # m

        return super().update(pred, label)