"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from .tools import gen_dx_bx, cumsum_trick, QuickCumsum

import torch.nn.functional as F
from einops import rearrange
from utils.functions import Normalize
import numpy as np
from models.cvt.decoder import Decoder

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, cfg, D, C, img_feat_shapes):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C
        
        model_name = cfg['LSS']['backbone']['model_name']
        self.layer_nums = cfg['LSS']['backbone']['layer_nums']
        self.trunk = EfficientNet.from_pretrained(model_name)

        self.trunk._conv_head = nn.Identity()
        self.trunk._bn1 = nn.Identity()
        self.trunk._avg_pooling = nn.Identity()
        self.trunk._dropout = nn.Identity()
        self.trunk._fc = nn.Identity()
        self.trunk._swish = nn.Identity()
        
        input_size = img_feat_shapes[0][1] + img_feat_shapes[1][1]
        if input_size < 512: # eff-b0: 112+320=432
            output_size = 512
        else:                # eff-b4: 160+448=608
            output_size = 640
        self.up1 = Up(input_size, output_size)
        self.depthnet = nn.Conv2d(output_size, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x, img_feats = self.get_eff_depth(x)
        # Depth
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        return new_x, img_feats

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints[f'reduction_{self.layer_nums[1]}'], endpoints[f'reduction_{self.layer_nums[0]}'])
        img_feats = [endpoints[f'reduction_{self.layer_nums[0]}'], endpoints[f'reduction_{self.layer_nums[1]}']]
        return x, img_feats

    def forward(self, x):
        x, img_feats = self.get_depth_feat(x)

        return x, img_feats


class BevEncode(nn.Module):
    def __init__(self, inC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        return x3, x1
    
class BevDecode(nn.Module):
    def __init__(self, outC):
        super(BevDecode, self).__init__()

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x3, x1):
        x = self.up1(x3, x1)
        x = self.up2(x)

        return x


class LiftSplatShoot(nn.Module):
    def __init__(self, cfg, args, img_feat_shapes):
        super(LiftSplatShoot, self).__init__()
        #self.grid_conf = grid_conf
        # self.data_aug_conf = data_aug_conf
        
        self.cfg = cfg
        self.args = args
        self.grid_conf = {'xbound': cfg['LSS']['xbound'],
                          'ybound': cfg['LSS']['ybound'],
                          'zbound': cfg['LSS']['zbound'],
                          'dbound': cfg['LSS']['dbound']}
        self.data_aug_conf = {'final_dim': [args.img_h, args.img_w]}
        outC = len(args.targets)

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                self.grid_conf['ybound'],
                                self.grid_conf['zbound'],
                                )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)
        # self.nx = nx.tolist()

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(cfg, self.D, self.camC, img_feat_shapes)
        self.bevencode = BevEncode(inC=self.camC)
        
        if cfg['LSS']['decoder']['type'] == 'LSS_original':
            self.bevdecode = BevDecode(outC=outC)
            if self.args.get_height:
                self.heightdecode = BevDecode(outC=1)
                
        elif cfg['LSS']['decoder']['type'] == 'CVT_conv':
            dim_last = cfg['CVT']['dim_last']
            self.decoder = Decoder(dim=256,
                                blocks=cfg['CVT']['decoder']['blocks'],
                                residual=cfg['CVT']['decoder']['residual'],
                                factor=cfg['CVT']['decoder']['factor'])
            self.to_logits = nn.Sequential(
                nn.Conv2d(self.decoder.out_channels, dim_last, 3, padding=1, bias=False),
                nn.BatchNorm2d(dim_last),
                nn.ReLU(inplace=False), 
                nn.Conv2d(dim_last, len(args.targets), 1)) 
            
            if self.args.get_height:
                self.decoder_height = Decoder(dim=256,
                                            blocks=cfg['CVT']['decoder']['blocks'],
                                            residual=cfg['CVT']['decoder']['residual'],
                                            factor=cfg['CVT']['decoder']['factor'])
                self.to_logits_height = nn.Sequential(
                    nn.Conv2d(self.decoder.out_channels, dim_last, 3, padding=1, bias=False),
                    nn.BatchNorm2d(dim_last),
                    nn.ReLU(inplace=False), 
                    nn.Conv2d(dim_last, 1, 1)) 
        

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
        
        self.norm = Normalize('imagenet')
    
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape
        post_rots = post_rots.expand(B, N, 3, 3).to(rots.device)
        post_trans = post_trans.expand(B, N, 3).to(trans.device)

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points


    def extrinsic_transform(self, resize, crop, flip, rotate):
        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)
        
        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate/180*np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
        
        post_tran2 = torch.zeros(3)
        post_rot2 = torch.eye(3)
        post_tran2[:2] = post_tran
        post_rot2[:2, :2] = post_rot

        return post_rot2, post_tran2
    
    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])


    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        
        BN, C, imH, imW = x.shape
        N = 6
        B = BN // N

        # B, N, C, imH, imW = x.shape
        # x = x.view(B*N, C, imH, imW)
        x, img_feats = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x, img_feats

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        #geom_feats = torch.floor((geom_feats - (self.bx - self.dx/2.)) / self.dx).to(torch.long)
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2].to(torch.int), self.nx[0].to(torch.int), self.nx[1].to(torch.int)), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x, img_feats = self.get_cam_feats(x)
        x = self.voxel_pooling(geom, x)
        return x, img_feats

    def forward(self, batch, isTrain=True):
        x = self.norm(batch['image'])
        I = batch['intrinsics']
        E = batch['c2e'].inverse() # EV2C
        
        resize = self.args.img_w / 1600
        crop = (0, self.args.img_top_crop, self.args.img_w, self.args.img_h+self.args.img_top_crop)
        post_rots, post_trans = self.extrinsic_transform(resize, crop, False, 0.0)
        
        rots = E[..., :3, :3]
        trans = E[..., :3, 3]
        x, img_feats = self.get_voxels(x, rots, trans, I, post_rots, post_trans)
        
        x3, x1 = self.bevencode(x)
        
        if self.cfg['LSS']['decoder']['type'] == 'LSS_original':
            z = self.bevdecode(x3, x1)
        elif self.cfg['LSS']['decoder']['type'] == 'CVT_conv':
            z = self.to_logits(self.decoder(x3))
        
        output = {}
        for idx, key in enumerate(self.args.targets):
            output[key] = [z[:, idx:idx+1]]
            
        if isTrain and self.args.get_height:
            if self.cfg['LSS']['decoder']['type'] == 'LSS_original':
                h = self.heightdecode(x3, x1)
            elif self.cfg['LSS']['decoder']['type'] == 'CVT_conv':
                h = self.to_logits_height(self.decoder_height(x3))
            output['height'] = [h]
            
        if self.args.model_name == 'Baseline_PYVA':
            return output, x3, img_feats
        else:
            return output, x3


def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShoot(grid_conf, data_aug_conf, outC)