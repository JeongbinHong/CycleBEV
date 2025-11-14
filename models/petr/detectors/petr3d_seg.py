# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import os
import torch
import torch.nn as nn
import numpy as np
from os import path as osp
from PIL import Image

from models.backbone import VoVNetCP
from models.extractors import ResNet_Extractor
from ..dense_heads.petr_head_seg import PETRHead_seg
from ..neck.cp_fpn import CPFPN
from einops import rearrange

def IOU (intputs, targets, eps=1e-6):
    intputs = intputs.bool()
    targets = targets.bool()
    inter = (intputs & targets).sum(-1)
    union = (intputs | targets).sum(-1)
    # iou = (numerator + eps) / (denominator + eps - numerator)
    return inter.cpu(),union.cpu()

class Petr3D_seg(nn.Module):
    """Detr3D."""

    def load_pretrained_weights(self, model, checkpoint_path, device):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        if 'state_dict' in checkpoint:  # Handle checkpoints saved with model wrappers
            checkpoint = checkpoint['state_dict']
        model.to("cpu")
        model.load_state_dict(checkpoint, strict=False)
        model.to(device)
        return model

    def __init__(self, cfg, petr_cfg, args):
        super(Petr3D_seg, self).__init__()

        self.use_grid_mask = petr_cfg['use_grid_mask']
        img_backbone = petr_cfg['img_backbone']
        out_features = petr_cfg['img_backbone']['out_features']
        img_neck = petr_cfg['img_neck']
        pts_bbox_head = petr_cfg['pts_bbox_head']
        self.with_time = pts_bbox_head['with_time']
        self.with_se = pts_bbox_head['with_se']
        
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)


        img_backbone_args = img_backbone.copy()
        img_backbone_type = img_backbone_args.pop('type')
        pretrained_file = img_backbone_args.pop('pretrained_file')

        if img_backbone_type == 'VoVNetCP':
            self.img_backbone = VoVNetCP(**img_backbone_args)
            if pretrained_file != None:
                if args.ddp:
                    local_rank = int(os.environ["LOCAL_RANK"])
                    device = torch.device(f"cuda:{local_rank}")
                    torch.cuda.set_device(device)
                else:
                    device = torch.device(f"cuda:0")
                self.img_backbone = self.load_pretrained_weights(self.img_backbone, pretrained_file, device)
                # img_neck['in_channels'] = [768, 1024]
                img_neck['in_channels'] = []
                for s in out_features:
                    if s == 'stage2': img_neck['in_channels'].append(256)
                    elif s == 'stage3': img_neck['in_channels'].append(512)
                    elif s == 'stage4': img_neck['in_channels'].append(768)
                    elif s == 'stage5': img_neck['in_channels'].append(1024) 
                img_neck['num_outs'] = len(img_neck['in_channels'])
                
        elif img_backbone_type == 'resnet101':
            self.img_backbone = ResNet_Extractor(num_classes=3, type='resnet101', layer_nums=[2,3])
            img_neck['in_channels'] = [1024, 2048]
            img_neck['num_outs'] = len(img_neck['in_channels'])
        else :
            raise NotImplementedError("Petr3D_seg.py supports only ['VoVNetCP', 'resnet101'] as the img_backbone.")

        img_neck_args = img_neck.copy()
        img_neck_type = img_neck_args.pop('type')
        if img_neck_type == 'CPFPN':
            self.img_neck = CPFPN(**img_neck_args)
        else :
            raise NotImplementedError("Petr3D_seg.py supports only ['CPFPN'] as the img_neck.")

        pts_bbox_head_args = pts_bbox_head.copy()
        pts_bbox_head_type = pts_bbox_head_args.pop('type')
        if pts_bbox_head_type == 'PETRHead_seg':
            assert self.with_time == self.with_se == cfg['use_temporal'], f"'with_time', 'with_se', 'use_temporal' are should be same."
            if self.with_time:
                self.pts_bbox_head = PETRHead_seg_temp(cfg=cfg, args=args, **pts_bbox_head_args)
            else:
                self.pts_bbox_head = PETRHead_seg(cfg=cfg, args=args, **pts_bbox_head_args)
        else :
            raise NotImplementedError("Petr3D_seg.py supports only ['PETRHead_seg'] as the pts_bbox_head.")
    

    def extract_img_feat(self, img, B, N, T=None):
        """Extract features of images."""
        # img shape : (b*n, c, h, w)

        if self.use_grid_mask:
            img = self.grid_mask(img)
        img_feats = self.img_backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        if self.img_neck != None:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            if self.with_time:
                BTN, C, H, W = img_feat.size()
                img_feat = img_feat.contiguous().view(B, T, N, C, H, W) # 먼저 (B,T,N,...)로 복구. 순서가 중요하기 때문
                img_feats_reshaped.append(img_feat.view(B, T*N, C, H, W))
            else:
                BN, C, H, W = img_feat.size()
                img_feats_reshaped.append(img_feat.view(B, N, C, H, W))
        return img_feats, img_feats_reshaped


    def forward(self, image, I, E, e2w=None, l2e=None, isTrain=True):
        if self.with_time:
            B, T, N, _, _, _ = image.shape
            image = rearrange(image, 'b t n c h w -> (b t n) c h w')
            I = rearrange(I, 'b t n p q -> b (t n) p q')
            E = rearrange(E, 'b t n p q -> b (t n) p q')
        else:
            B, N = I.size(0), I.size(1)
            T = None

        img_feats, img_feats_reshaped = self.extract_img_feat(image, B, N, T)
        img_size = (image.size(-2), image.size(-1))
        output, x = self.pts_bbox_head(img_feats_reshaped, I, E, img_size, e2w=e2w, l2e=l2e, isTrain=isTrain) # petr_head_seg.py - PETRHead_seg
        return output, x, img_feats #output['road'] =[(b, 1, 200, 200)]


class GridMask(nn.Module):
    def __init__(self, use_h, use_w, rotate = 1, offset=False, ratio = 0.5, mode=0, prob = 1.):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch #+ 1.#0.5

    def forward(self, x):
        if np.random.rand() > self.prob or not self.training:
            return x
        n,c,h,w = x.size()
        x = x.view(-1,h,w)
        hh = int(1.5*h)
        ww = int(1.5*w)
        d = np.random.randint(2, h)
        self.l = min(max(int(d*self.ratio+0.5),1),d-1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh//d):
                s = d*i + st_h
                t = min(s+self.l, hh)
                mask[s:t,:] *= 0
        if self.use_w:
            for i in range(ww//d):
                s = d*i + st_w
                t = min(s+self.l, ww)
                mask[:,s:t] *= 0
       
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh-h)//2:(hh-h)//2+h, (ww-w)//2:(ww-w)//2+w]

        mask = torch.from_numpy(mask).float().cuda()
        if self.mode == 1:
            mask = 1-mask
        mask = mask.expand_as(x)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h,w) - 0.5)).float().cuda()
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask 

        return x.view(n,c,h,w)