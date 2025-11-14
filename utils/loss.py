import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
# from utils.functions import clamped_sigmoid
from fvcore.nn import sigmoid_focal_loss
import sys
import copy
from scipy.ndimage import distance_transform_edt
import random
from typing import Optional
# --------------------------------
# Common
class Optimizers(nn.Module):
    def __init__(self, model, optimizer_type, learning_rate, weight_decay, config=None):
        super(Optimizers, self).__init__()

        if (optimizer_type == 'adam'):
            self.opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)
        elif (optimizer_type == 'adamw'):
            self.opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay) 
        else:
            sys.exit(f">> Optimizer {optimizer_type} is not supported !!")


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha 
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none') 
        pt = torch.exp(-ce_loss) 

        if self.alpha is not None:
            at = self.alpha[targets]  
            ce_loss = at * ce_loss

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# --------------------------------
# BEV
class SpatialRegressionLoss(nn.Module):
    def __init__(self, norm, ignore_index=255, future_discount=1.0):
        super(SpatialRegressionLoss, self).__init__()
        self.norm = norm
        self.ignore_index = ignore_index
        self.future_discount = future_discount

        if norm == 1: 
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'Expected norm 1 or 2, but got norm={norm}')

    def forward(self, prediction, target):
        assert len(prediction.shape) == 5, 'Must be a 5D tensor'
        # ignore_index is the same across all channels
        mask = target[:, :, :1] != self.ignore_index
        if mask.sum() == 0:
            return prediction.new_zeros(1)[0].float()

        loss = self.loss_fn(prediction, target, reduction='none')

        # Sum channel dimension
        loss = torch.sum(loss, dim=-3, keepdims=True)

        seq_len = loss.shape[1]
        future_discounts = self.future_discount ** torch.arange(seq_len, device=loss.device, dtype=loss.dtype)
        future_discounts = future_discounts.view(1, seq_len, 1, 1, 1)
        loss = loss * future_discounts

        return loss[mask].mean()

class SegmentationLoss(nn.Module):
    def __init__(self, class_weights, ignore_index=255, use_top_k=False, top_k_ratio=1.0, future_discount=1.0):
        super().__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.future_discount = future_discount

    def forward(self, prediction, target):
        if target.shape[-3] != 1:
            raise ValueError('segmentation label must be an index-label with channel dimension = 1.')
        b, s, c, h, w = prediction.shape

        prediction = prediction.view(b * s, c, h, w)
        target = target.view(b * s, h, w)
        loss = F.cross_entropy(
            prediction,
            target,
            ignore_index=self.ignore_index,
            reduction='none',
            weight=self.class_weights.to(target.device),
        )

        loss = loss.view(b, s, h, w)

        future_discounts = self.future_discount ** torch.arange(s, device=loss.device, dtype=loss.dtype)
        future_discounts = future_discounts.view(1, s, 1, 1)
        loss = loss * future_discounts

        loss = loss.view(b, s, -1)
        if self.use_top_k:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[2])
            loss, _ = torch.sort(loss, dim=2, descending=True)
            loss = loss[:, :, :k]

        return torch.mean(loss)

class ProbabilisticLoss(nn.Module):
    def forward(self, output):
        present_mu = output['present_mu']
        present_log_sigma = output['present_log_sigma']
        future_mu = output['future_mu']
        future_log_sigma = output['future_log_sigma']

        var_future = torch.exp(2 * future_log_sigma)
        var_present = torch.exp(2 * present_log_sigma)
        kl_div = (
                present_log_sigma - future_log_sigma - 0.5 + (var_future + (future_mu - present_mu) ** 2) / (
                    2 * var_present)
        )

        kl_loss = torch.mean(torch.sum(kl_div, dim=-1))

        return kl_loss

class SigmoidFocalLoss(torch.nn.Module):
    def __init__(self, alpha=-1.0, gamma=2.0, reduction='mean'):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, label):
        return sigmoid_focal_loss(pred, label, self.alpha, self.gamma, self.reduction)

class TopKBinaryCrossEntropyLoss(nn.Module):
    def __init__(self, label_indices, min_visibility, use_top_k=False, top_k_ratio=1.0):
        super().__init__()

        self.label_indices = label_indices
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.min_visibility = min_visibility

    def calc_cross_entropy(self, pred, label, visibility=None, ignore_label_indices=False):

        if ignore_label_indices is False:
            if self.label_indices is not None:
                label = [label[:, idx].max(dim=1, keepdim=True).values for idx in self.label_indices]
                label = torch.cat(label, dim=1)

        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")

        if self.min_visibility is not None:
            mask = visibility >= self.min_visibility
            loss = loss[mask]

        if self.use_top_k is not None:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[0])
            loss, _ = torch.sort(loss, descending=True)
            loss = loss[:k]
        return loss.mean()

    def forward(self, pred, batch, alpha, beta, target):

        loss_bev = self.calc_cross_entropy(pred['bev'], batch['bev'].to(pred['bev']), batch['visibility'])
        loss_center = self.calc_cross_entropy(pred['center'], batch['center'].to(pred['bev']),
                                              batch['visibility'], ignore_label_indices=True)

        return alpha*loss_bev + beta*loss_center

class DiceLoss(nn.Module):
    def __init__(self, label_indices, min_visibility, smooth=1.0):
        super().__init__()

        self.label_indices = label_indices
        self.min_visibility = min_visibility
        self.smooth = smooth

    def calc_diceloss(self, pred, label, visibility=None, ignore_label_indices=False):

        if ignore_label_indices is False:
            if self.label_indices is not None:
                label = [label[:, idx].max(dim=1, keepdim=True).values for idx in self.label_indices]
                label = torch.cat(label, dim=1)

        if self.min_visibility is not None:
            mask = visibility >= self.min_visibility
            pred = torch.sigmoid(pred)[mask]
            label = label[mask]
        else:
            pred = torch.sigmoid(pred).view(-1)
            label = label.view(-1)

        intersection = (pred * label).sum()
        union = pred.sum() + label.sum()

        # dice coefficient
        dice = 2.0 * (intersection + self.smooth) / (union + 1e-10)

        # dice loss
        dice_loss = 1.0 - dice

        return dice_loss

    def forward(self, pred, batch, alpha, beta, target):

        loss_bev = self.calc_diceloss(pred['bev'], batch['bev'].to(pred['bev']), batch['visibility'])
        loss_center = self.calc_diceloss(pred['center'], batch['center'].to(pred['bev']),
                                              batch['visibility'], ignore_label_indices=True)

        return alpha*loss_bev + beta*loss_center

class LossScratch(torch.nn.Module):
    def __init__(self, cfg, args, min_visibility=None, reduction='none'):
        super().__init__()

        self.cfg = cfg
        self.args = args
        self.target = args.targets
        self.label_indices = cfg['label_indices']
        self.min_visibility = min_visibility
        self.bce = SigmoidFocalLoss(alpha=cfg['Loss']['bce']['alpha'], gamma=cfg['Loss']['bce']['gamma'], reduction=reduction)
        self.focal = SigmoidFocalLoss(alpha=cfg['Loss']['focal']['alpha'], gamma=cfg['Loss']['focal']['gamma'], reduction=reduction)
        self.l1 = nn.L1Loss(size_average=None, reduce=None, reduction='none')
        self.crossEntropy = torch.nn.CrossEntropyLoss()

    def bce_loss(self, pred, label, label_indices=None, visibility=None, ignore_label_indices=False):

        if ignore_label_indices is False:
            if label_indices is not None:
                label = [label[:, idx].max(dim=1, keepdim=True).values for idx in label_indices]
                label = torch.cat(label, dim=1)

        _, _, hp, wp = pred.size() # height_pred, width_pred
        _, _, hl, wl = label.size()
        if (hp < hl):
            scale = hp / hl 
            label = F.interpolate(label, scale_factor=scale, mode='nearest') 
            if (visibility is not None):
                visibility = F.interpolate(visibility.to(pred), scale_factor=scale, mode='nearest')

        loss = self.bce(pred, label)

        if self.min_visibility is not None and visibility is not None:
            mask = visibility >= self.min_visibility
            loss = loss[mask]

        return loss
            

    def focal_loss(self, pred, label, label_indices=None, visibility=None, ignore_label_indices=False):
        '''
        pred : b x 1 x h x w
        label : b x 1 x h x w
        visibility : b x 1 x h x w
        '''
        
        if ignore_label_indices is False:
            if label_indices is not None:
                label = [label[:, idx].max(dim=1, keepdim=True).values for idx in label_indices]
                label = torch.cat(label, dim=1)
                '''
                  "label_indices": {"vehicle": [[4, 5, 6, 7, 8, 10, 11]],
                                    "lane":  [[2, 3]],
                                    "road":  [[0, 1]],
                                    "pedestrian":  [[9]]},
                '''

        _, _, hp, wp = pred.size()
        _, _, hl, wl = label.size()
        if (hp < hl):  # visibility 
            scale = hp / hl
            label = F.interpolate(label, scale_factor=scale, mode='nearest')
            if (visibility is not None):  
                visibility = F.interpolate(visibility.to(pred), scale_factor=scale, mode='nearest')

        loss = self.focal(pred, label)

        if self.min_visibility is not None and visibility is not None:
            mask = visibility >= self.min_visibility 
            loss = loss[mask]

        return loss
    
    def label_to_one_hot_label(
        self,
        labels: torch.Tensor,
        num_classes: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        eps: float = 1e-6,
        ignore_index=255,
    ) -> torch.Tensor:
        r"""Convert an integer label x-D tensor to a one-hot (x+1)-D tensor.

        Args:
            labels: tensor with labels of shape :math:`(N, *)`, where N is batch size.
            Each value is an integer representing correct classification.
            num_classes: number of classes in labels.
            device: the desired device of returned tensor.
            dtype: the desired data type of returned tensor.

        Returns:
            the labels in one hot tensor of shape :math:`(N, C, *)`,

        Examples:
            >>> labels = torch.LongTensor([
                    [[0, 1], 
                    [2, 0]]
                ])
            >>> one_hot(labels, num_classes=3)
            tensor([[[[1.0000e+00, 1.0000e-06],
                    [1.0000e-06, 1.0000e+00]],
            
                    [[1.0000e-06, 1.0000e+00],
                    [1.0000e-06, 1.0000e-06]],
            
                    [[1.0000e-06, 1.0000e-06],
                    [1.0000e+00, 1.0000e-06]]]])

        """
        shape = labels.shape
        # one hot : (B, C=ignore_index+1, H, W)
        one_hot = torch.zeros((shape[0], ignore_index+1) + shape[1:], device=device, dtype=dtype)
        
        # labels : (B, H, W)
        # labels.unsqueeze(1) : (B, C=1, H, W)
        # one_hot : (B, C=ignore_index+1, H, W)
        one_hot = one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps
        
        # ret : (B, C=num_classes, H, W)
        ret = torch.split(one_hot, [num_classes, ignore_index+1-num_classes], dim=1)[0]
        
        return ret


    def l1_loss(self, pred, label, visibility=None, instance=None):
        '''
        pred : b x 2 x h x w
        label : b x 2 x h x w
        visibility : b x 1 x h x w
        instance : b x 1 x h x w
        '''

        if (pred is None):
            return torch.zeros(1).to(label)

        loss = self.l1(pred, label)
        if (visibility is not None and instance is not None):
            mask_visibility = visibility >= self.min_visibility
            mask_instance = instance > 0
            mask = torch.logical_and(mask_visibility, mask_instance)
            loss = loss[mask.repeat(1, pred.size(1), 1, 1)]

        return loss

    def main(self, pred, batch):  # CVT_solver.py - def train()

        # labels
        bev_gt = batch['bev'].to(pred[self.target[0]][0]) # b 12 h w  # tensor([[[[0,0,0,...]]]])
        bev_hgt = batch['bev_height'].to(pred[self.target[0]][0])
        bev_multi = batch['bev_multi'].to(pred[self.target[0]][0])

        center_gt, visibility = None, None
        if ('center' in batch.keys()):
            center_gt = {'vehicle': batch['center'][:, [0]].to(pred[self.target[0]][0]),  # loader_typeApro.py center_score_veh
                         'pedestrian': batch['center'][:, [1]].to(pred[self.target[0]][0])} 

        if ('visibility' in batch.keys()):
            visibility = {'vehicle': batch['visibility'][:, [0]],  
                          'pedestrian': batch['visibility'][:, [1]]} 

        
        # losses
        losses = {}
        for i, target in enumerate(self.target):
            if (target == 'road' or target == 'drivable'): # 250211 Jeongbin
                bce = self.bce_loss(pred[target][0], bev_gt, self.label_indices[target])
                focal = self.focal_loss(pred[target][0], bev_gt, self.label_indices[target])
                losses.update({target: {'loss': bce.mean() + focal.mean(), 'weight': 1.0}})

            elif (target == 'lane'):
                focal = self.focal_loss(pred[target][0], bev_gt, self.label_indices[target])
                losses.update({target: {'loss': focal.mean(), 'weight': 1.0}})

            elif (target == 'vehicle' or target == 'pedestrian'):
                focal = self.focal_loss(pred[target][0], bev_gt, self.label_indices[target], visibility[target]) 
                if len(pred[target]) > 1:
                    center = self.focal_loss(pred[target][1], center_gt[target], label_indices=None, visibility=visibility[target],
                                            ignore_label_indices=True).mean() # 유사도(챠량중심 - 그리드 상의 차량 위치)에 대한 loss
                else:
                    center = 0.0
                    
                losses.update({target: {'loss': focal.mean() + center, 'weight': 1.0}})

            else:
                sys.exit(f'>> {target} is not supported for loss calculation!!')
                
        if 'height' in pred:
            h_mse = F.mse_loss(pred['height'][0], bev_hgt, reduction='none')
            if bev_multi.ndim==4 and bev_multi.size(1)==1:
                bev_multi = bev_multi.squeeze(1).long()
            alpha = torch.tensor(self.args.multi_class_height_weights, device=pred['height'][0].device)
            alpha_class = alpha[bev_multi]                        # [B, H, W]
            h_mse = (h_mse * alpha_class).mean()
            
            losses.update({'height': {'loss': h_mse, 'weight': 1.0}})

        
        return losses

    def intermediate(self, pred, batch):
        '''
        ** pred
            pred['road'][0] : b 1 h w
            pred['lane'][0] : b 1 h w
            pred['vehicle'][0] : b 1 h w
            pred['vehicle'][1] : b 1 h w  -> center
            pred['pedestrian'][0] : b 1 h w
            pred['pedestrian'][1] : b 1 h w  -> center
            pred['intp'] :  a list of tensors of shape 'b 1 h w'

        ** batch
            batch['bev'] : b 12 h w
            batch['center'] : b 1 h w
            batch['visibility'] : b 1 h w

        ** batch['bev']
            load : 0, 1
            lane : 2, 3
            vehicle : 4, 5, 6, 7, 8, 10, 11
            pedestrian : 9
        '''

        # update 231006
        tdix = {}
        for i, key in enumerate(self.target):
            tdix[key] = [i, i+1]

        # labels
        bev_gt = batch['bev'].to(pred[self.target[0]][0]) # b 12 h w

        # update 231006
        visibility = {'vehicle': batch['visibility'][:, [0]],
                      'pedestrian': batch['visibility'][:, [1]]}

        # losses
        losses = {}
        intm_logits = pred['intm']
        for _, target in enumerate(self.target):
            if (target == 'road'):
                loss = torch.zeros(1).to(bev_gt)
                for intp_logit in intm_logits:
                    loss += self.focal_loss(intp_logit[:, tdix[target][0]:tdix[target][1]],
                                           bev_gt, self.label_indices[target]).mean()
                losses.update({target: {'loss': loss, 'weight': 1.0}})

            elif (target == 'lane'):
                loss = torch.zeros(1).to(bev_gt)
                for intp_logit in intm_logits:
                    loss += self.focal_loss(intp_logit[:, tdix[target][0]:tdix[target][1]],
                                           bev_gt, self.label_indices[target]).mean()
                losses.update({target: {'loss': loss, 'weight': 1.0}})

            elif (target == 'vehicle' or target == 'pedestrian'):
                loss = torch.zeros(1).to(bev_gt)
                for intp_logit in intm_logits:
                    loss += self.focal_loss(intp_logit[:, tdix[target][0]:tdix[target][1]],
                                           bev_gt, self.label_indices[target], visibility[target]).mean() # update 231006
                losses.update({target: {'loss': loss, 'weight': 1.0}})

            else:
                sys.exit(f'>> {target} is not supported for loss calculation!!')

        return losses

    def offset(self, pred, batch):
        '''
        ** pred
        pred['vehicle'][0] : b 1 h w
        pred['vehicle'][1] : b 1 h w
        pred['road'][0] : b 1 h w
        pred['lane'][0] : b 1 h w
        pred['intp'] :  a list of tensors of shape 'b 1 h w'

        ** batch
        batch['bev'] : b 12 h w
        batch['center'] : b 1 h w
        batch['visibility'] : b 1 h w
        batch['offsets'] : b 2 h w
        '''

        # labels, update 231006
        bev_gt = {}
        bev_gt['vehicle'] = batch['offsets'][:, :2].to(pred[self.target[0]][0]) # b 2 h w
        bev_gt['pedestrian'] = batch['offsets'][:, 2:].to(pred[self.target[0]][0]) # b 2 h w

        visibility = {'vehicle': None, 'pedestrian': None}
        instance = {'vehicle': None, 'pedestrian': None}

        if (self.cfg['bool_use_vis_offset']):
            visibility['vehicle'] = batch['visibility'][:, [0]]
            visibility['pedestrian'] = batch['visibility'][:, [1]]
            instance['vehicle'] = batch['instance'][:, [0]]
            instance['pedestrian'] = batch['instance'][:, [1]]

        # losses
        losses = {}
        for _, target in enumerate(self.target):
            # update 231006
            if (target == 'vehicle'):
                loss = self.l1_loss(pred['offsets'][target], bev_gt[target], visibility[target], instance[target]).mean()
                losses.update({target: {'loss': loss, 'weight': 1.0}})
            elif (target == 'pedestrian'):
                loss = self.l1_loss(pred['offsets'][target], bev_gt[target], visibility[target], instance[target]).mean()
                losses.update({target: {'loss': loss, 'weight': 1.0}})

        return losses


def calc_ED_error(o, k, best_k, pred_trajs, future_traj, ADE_k, FDE_k):
    if (k == 1):
        error_ADE = np.sqrt(np.sum((pred_trajs[0, :, o, :2] - future_traj[:, o, :2]) ** 2, axis=1))
        error_FDE = np.sqrt(np.sum((pred_trajs[0, :, o, :2] - future_traj[:, o, :2]) ** 2, axis=1))
        return error_ADE, error_FDE[-1]

    elif (k <= best_k):
        minADE_idx = np.argmin(np.array(ADE_k[:k]))
        minFDE_idx = np.argmin(np.array(FDE_k[:k]))
        error_ADE = np.sqrt(np.sum((pred_trajs[minADE_idx, :, o, :2] - future_traj[:, o, :2]) ** 2, axis=1))
        error_FDE = np.sqrt(np.sum((pred_trajs[minFDE_idx, :, o, :2] - future_traj[:, o, :2]) ** 2, axis=1))
        return error_ADE, error_FDE[-1]

    else:
        return 0, 0


    def __call__(self, pred, gt):
        """
        pred : best_k x seq_len x batch x 2
        gt : seq_len x batch x 2
        """
        best_k, seq_len, batch, dim = pred.size()
        squared_error = (pred - gt.unsqueeze(0)).pow(2).sum(dim=-1)
        return squared_error / seq_len / batch




class AvgL2Loss(nn.Module):

    def __init__(self):
        super(AvgL2Loss, self).__init__()

    def __call__(self, pred, gt):
        """
        pred : best_k x seq_len x batch x 2
        gt : seq_len x batch x 2
        """
        best_k, seq_len, batch, dim = pred.size()
        squared_error = (pred - gt.unsqueeze(0)).pow(2).sum(dim=-1)
        return squared_error / seq_len / batch


class BOSL2Loss(nn.Module):

    def __init__(self):
        super(BOSL2Loss, self).__init__()

    def __call__(self, pred, gt):
        """
        pred : best_k x seq_len x batch x 2
        gt : seq_len x batch x 2
        """
        best_k, seq_len, batch, dim = pred.size()
        squared_error = (pred - gt.unsqueeze(0)).pow(2).sum(dim=-1)  # best_k x seq_len x batch
        k_losses = squared_error.sum(-1).sum(-1)  # best_k
        return k_losses.min() / seq_len / batch


class L2Loss(nn.Module):

    def __init__(self, loss_type='avg'):
        super(L2Loss, self).__init__()

        if (loss_type == 'avg'):
            self.loss_func = AvgL2Loss()
        elif (loss_type == 'bos'):
            self.loss_func = BOSL2Loss()
        else:
            sys.exit(f'[Error] {loss_type} is not supported loss function!!')

    def __call__(self, pred, gt):
        """
        pred : best_k x seq_len x batch x 2
        gt : seq_len x batch x 2
        """
        return self.loss_func(pred, gt)


class LRScheduler(nn.Module):
    def __init__(self, optimizer, type='OnecycleLR', config=None):
        super(LRScheduler, self).__init__()
        """ Required config keys
        1) StepLR : step_size, gamma
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html

        2) ExponentialLR : gamma
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR

        3) OneCycleLR : max_lr, div_factor, final_div_factor, pct_start, steps_per_epoch, epochs, cycle_momentum
        https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
        """

        if (type == 'OnecycleLR' and config is not None):
            self.scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                                            max_lr=config['max_lr'],
                                                            div_factor=config['div_factor'], # starts at max_lr / 10
                                                            final_div_factor=config['final_div_factor'], # ends at lr / 10 / 10
                                                            pct_start=config['pct_start'], # reaches max_lr at 30% of total steps
                                                            steps_per_epoch=config['steps_per_epoch'],
                                                            epochs=config['epochs'],
                                                            cycle_momentum=False)
        elif (type == 'CosineAnnealingLR' and config is not None):
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-2*1e-3)
        else:
            sys.exit(f'[Error] LR scheduler named {type} is not implemented!!')

    def __call__(self):
        self.scheduler.step()

def softmax_focal_loss(
        logits,
        targets,
        gamma=2.0,
        alpha=[0.02, 0.05, 0.2, 1.0],
        reduction='none',
    ):
        """
        logits: [B, C, H, W]
        targets: [B, H, W] 
        """

        B, C, H, W = logits.shape

        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # [B, H, W]

        probs = F.softmax(logits, dim=1)                            # [B, C, H, W]
        one_hot = F.one_hot(targets, num_classes=C)                 # [B, H, W, C]
        one_hot = one_hot.permute(0, 3, 1, 2).float()               # [B, C, H, W]
        pt = (probs * one_hot).sum(dim=1)                           # [B, H, W]
        focal_term = (1 - pt) ** gamma
        loss = focal_term * ce_loss                                 # [B, H, W]

        if alpha is not None:
            if isinstance(alpha, (list, torch.Tensor)):
                alpha = torch.tensor(alpha, device=logits.device)
                alpha_class = alpha[targets]                        # [B, H, W]
                loss = loss * alpha_class
            else:
                loss = loss * alpha  # scalar

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss