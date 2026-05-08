# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

from models.petr.transformers.petr_transformer import PETRTransformer
from models.petr.utils.positional_encoding import SinePositionalEncoding, SinePositionalEncoding3D
from models.petr.utils.functions import inverse_sigmoid

def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb

def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    # pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

class RegLayer(nn.Module):
    def __init__(self,  embed_dims=256, 
                        shared_reg_fcs=2, 
                        group_reg_dims=(2, 1, 3, 2, 2),  # xy, z, size, rot, velo
                        act_layer=nn.ReLU, 
                        drop=0.0):
        super().__init__()

        reg_branch = []
        for _ in range(shared_reg_fcs):
            reg_branch.append(nn.Linear(embed_dims, embed_dims))
            reg_branch.append(act_layer())
            reg_branch.append(nn.Dropout(drop))
        self.reg_branch = nn.Sequential(*reg_branch)

        self.task_heads = nn.ModuleList()
        for reg_dim in group_reg_dims:
            task_head = nn.Sequential(
                nn.Linear(embed_dims, embed_dims),
                act_layer(),
                nn.Linear(embed_dims, reg_dim)
            )
            self.task_heads.append(task_head)

    def forward(self, x):
        reg_feat = self.reg_branch(x)
        outs = []
        for task_head in self.task_heads:
            out = task_head(reg_feat.clone())
            outs.append(out)
        outs = torch.cat(outs, -1)
        return outs


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_dim, residual, factor):
        super().__init__()

        dim = out_channels // factor
        
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
            # nn.BatchNorm2d(dim),
            # nn.LayerNorm(32),
            nn.ReLU(),
            nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
            # nn.BatchNorm2d(out_channels),
            # nn.LayerNorm(out_channels)
            )

        if residual:
            self.up = nn.Conv2d(skip_dim, out_channels, 1)
        else:
            self.up = None

        self.relu = nn.ReLU()

    def forward(self, x, skip):
        x = self.conv(x)

        if self.up is not None:
            up = self.up(skip)
            up = F.interpolate(up, x.shape[-2:])

            x = x + up

        return self.relu(x)


class Decoder(nn.Module):
    def __init__(self, dim, blocks, out_dim, residual=True, factor=2):
        super().__init__()

        layers = list()
        channels = dim

        for out_channels in blocks:
            layer = DecoderBlock(channels, out_channels, dim, residual, factor)
            layers.append(layer)

            channels = out_channels

        self.layers = nn.Sequential(*layers)
        
        self.to_logits = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, out_dim, 1))
        
    def forward(self, x):
        y = x

        for layer in self.layers:
            y = layer(y, x)
        y = self.to_logits(y)
        return y


class PETRHead_seg(nn.Module):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    _version = 2
    def __init__(self,
                 cfg,
                 args,
                 num_classes,
                 in_channels,
                 num_query=100,
                 num_lane=100,
                 num_reg_fcs=2,
                 transformer=None,
                 transformer_lane=None,
                 sync_cls_avg_factor=False,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 code_weights=None,
                 bbox_coder=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_dri=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.5,
                     loss_weight=2.0),
                 loss_lan=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.5,
                     loss_weight=2.0),
                 loss_veh=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.5,
                     loss_weight=2.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 loss_lane_mask=None,
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0))),
                 test_cfg=dict(max_per_img=100),
                 with_position=True,
                 with_multiview=False,
                 depth_step=0.8,
                 depth_num=64,
                 blocks=[128,128,64],
                 LID=False,
                 depth_start = 1,
                 position_level = 0,
                 position_range=[-65, -65, -8.0, 65, 65, 8.0],
                 init_cfg=None,
                 normedlinear=False,
                 with_se=False,
                 with_time=False,
                 with_detach=False,
                 with_multi=False,
                 group_reg_dims=(2, 1, 3, 2, 2),
                 print_log_of_all_decoders=True,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        
        # if train_cfg:
        #     sampler_cfg = dict(type='PseudoSampler')
        #     self.sampler = build_sampler(sampler_cfg, context=self)

        self.cfg = cfg
        self.args = args
        self.num_query = num_query
        self.blocks=blocks
        self.num_lane=num_lane
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.embed_dims = 256
        self.depth_step = depth_step
        self.depth_num = depth_num
        self.position_dim = 3 * self.depth_num
        self.position_range = position_range
        self.LID = LID
        self.depth_start = depth_start
        self.position_level = position_level
        self.with_position = with_position
        self.with_multiview = with_multiview
        self.with_multi = with_multi
        self.group_reg_dims = group_reg_dims
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.num_pred = self.cfg['PETR']['model']['pts_bbox_head']['transformer']['decoder']['num_pred'] # num decoders
        self.normedlinear = normedlinear
        self.with_se = with_se
        self.with_time = with_time
        self.with_detach = with_detach
        super(PETRHead_seg, self).__init__()
        
        positional_encoding_args = positional_encoding.copy()
        positional_encoding_type = positional_encoding_args.pop('type')
        if positional_encoding_type=='SinePositionalEncoding':
            self.positional_encoding = SinePositionalEncoding(**positional_encoding_args)
        elif positional_encoding_type=='SinePositionalEncoding3D':
            self.positional_encoding = SinePositionalEncoding3D(**positional_encoding_args)
        else:
            raise NotImplementedError("PETRHead_seg.py supports only ['SinePositionalEncoding', 'SinePositionalEncoding3D'] as the img_neck.")
        
        self.print_log_of_all_decoders = print_log_of_all_decoders

        transformer_lane_args = transformer_lane.copy()
        transformer_lane_type = transformer_lane_args.pop('type')
        if transformer_lane_type :
            self.transformer_lane = PETRTransformer(**transformer_lane_args)
        else :
            raise NotImplementedError("PETRHead_seg.py supports only ['PETRTransformer'] as the transformer_lane.")


        # 세그멘테이션의 경우 get_bboxes()에서만 사용해서 딱히 필요 없음.
        # bbox_coder_args = bbox_coder.copy()
        # bbox_coder_type = bbox_coder_args.pop('type')

        # if bbox_coder_type == 'NMSFreeCoder':
        #     self.bbox_coder = NMSFreeCoder(**args)
        # elif bbox_coder_type == 'NMSFreeClsCoder':
        #     self.bbox_coder = NMSFreeClsCoder(**args)
        # else : 
        #     raise NotImplementedError("Only ['NMSFreeCoder', 'NMSFreeClsCoder'] are supported as the bbox_coder.type.")

        self.single_decoder = self.cfg['PETR']['decoder']['single_decoder_on_multiclass']

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        if self.with_position:
            self.input_proj = nn.Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)
        else:
            self.input_proj = nn.Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)

        if self.single_decoder:
            decoder = Decoder(self.embed_dims, self.blocks, len(self.args.targets))
            # self.branches = nn.ModuleList([decoder for _ in range(self.num_pred)])
            self.branches = nn.ModuleList([decoder])
            if self.args.get_height:
                height_decoder = Decoder(self.embed_dims, self.blocks, 1)
                # self.branches_height = nn.ModuleList([height_decoder for _ in range(self.num_pred)])
                self.branches_height = nn.ModuleList([height_decoder])
        else:
            lane_branch_dri = Decoder(self.embed_dims,self.blocks,1)
            #lane_branch_lan = Decoder(self.embed_dims,self.blocks,1)
            lane_branch_veh = Decoder(self.embed_dims,self.blocks,1)
            lane_branch_ped = Decoder(self.embed_dims,self.blocks,1)

            self.lane_branches_dri = nn.ModuleList(
                [lane_branch_dri for _ in range(self.num_pred)])
            # self.lane_branches_lan = nn.ModuleList(
            #     [lane_branch_lan for _ in range(self.num_pred)])
            self.lane_branches_vie = nn.ModuleList(
                [lane_branch_veh for _ in range(self.num_pred)])
            self.lane_branches_ped = nn.ModuleList(
                [lane_branch_ped for _ in range(self.num_pred)])

            if self.args.get_height:
                lane_branch_hei = Decoder(self.embed_dims,self.blocks,1)
                self.lane_branches_hei = nn.ModuleList(
                    [lane_branch_hei for _ in range(self.num_pred)])

            # self.branches = nn.ModuleDict()
            # for idx, key in enumerate(self.args.targets):
            #     if key == 'drivable':
            #         self.branches[key] = nn.ModuleList([lane_branch_dri for _ in range(self.num_pred)])
            #     elif key == 'vehicle':
            #         self.branches[key] = nn.ModuleList([lane_branch_veh for _ in range(self.num_pred)])
            #     elif key == 'pedestrian':
            #         self.branches[key] = nn.ModuleList([lane_branch_ped for _ in range(self.num_pred)])
            
        
        if self.with_multiview:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims*3//2, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )

        if self.with_position:
            self.position_encoder = nn.Sequential(
                nn.Conv2d(self.position_dim, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )

        if self.with_se:
            self.se = SELayer(self.embed_dims)

        nx=ny=round(math.sqrt(self.num_lane))
        x = (torch.arange(nx) + 0.5) / nx
        y = (torch.arange(ny) + 0.5) / ny
        xy=torch.meshgrid(x,y)
        
        self.reference_points_lane = torch.cat([xy[0].reshape(-1)[...,None],xy[1].reshape(-1)[...,None]],-1)#.cuda()
        
        self.query_embedding_lane = nn.Sequential(
            nn.Linear(self.embed_dims*2//2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        
        self.transformer_lane.init_weights()
        
    def build_img2lidars(self, intrinsics, extrinsics, ref):
        """
        intrinsics: (..., N, 3,3) 또는 (..., N, 4,4)  # K
        extrinsics: (..., N, 4,4)                      # EL2C
        ref: dtype/device 참조용 텐서 (예: coords)
        return: (..., N, 4, 4)  # img2lidars
        """
        device, dtype = ref.device, ref.dtype
        intrinsics = intrinsics.to(device=device, dtype=dtype)
        extrinsics = extrinsics.to(device=device, dtype=dtype)

        if intrinsics.shape[-2:] == (3, 3):
            K4 = torch.zeros(*intrinsics.shape[:-2], 4, 4, device=device, dtype=dtype)
            K4[..., :3, :3] = intrinsics
            K4[..., 3, 3] = 1
        elif intrinsics.shape[-2:] == (4, 4):
            K4 = intrinsics
        else:
            raise ValueError(f"invalid intrinsics shape: {intrinsics.shape}")

        # lidar->img, then invert to get img->lidar
        lidar2img  = K4 @ extrinsics          # EL2C
        img2lidars = torch.linalg.inv(lidar2img)
        return img2lidars

    def position_embeding(self, img_feats, I, E, pad_size, masks=None):
        eps = 1e-5
        # pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        pad_h, pad_w = pad_size
        
        B, N, C, H, W = img_feats[self.position_level].shape
        coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H
        coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W

        if self.LID:
            index  = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index  = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        D = coords_d.shape[0]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0) # W, H, D, 3
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        # img2lidars = []
        # for img_meta in img_metas:
        #     img2lidar = []
        #     for i in range(len(img_meta['lidar2img'])): # n=6*t  
        #         img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i]))
        #     img2lidars.append(np.asarray(img2lidar))
        # img2lidars = np.asarray(img2lidars)
        # img2lidars = coords.new_tensor(img2lidars) # (B, N, 4, 4)
        
        img2lidars = self.build_img2lidars(I, E, coords)
        
        # P = torch.zeros(B, N, 4, 4, device=K.device, dtype=K.dtype)
        # P[..., :3, :3] = torch.einsum('bnij,bnjk->bnik', K, E[..., :3, :3])   # K R # E=EL2C
        # P[..., :3,  3] = torch.einsum('bnij,bnj->bni',   K, E[..., :3,  3])   # K t
        # P[..., 3, 3] = 1.0
        # lidar2img = P                                  # (B,N,4,4)
        # img2lidars = torch.inverse(lidar2img)


        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)  # K^-1
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3] # P^3d = K^(-1) * P^m 
        # position_range = [-65, -65, -8.0, 65, 65, 8.0]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (self.position_range[3] - self.position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (self.position_range[4] - self.position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (self.position_range[5] - self.position_range[2])

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0) 
        coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)
        coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B*N, -1, H, W)
        coords3d = torch.clamp(coords3d, 1e-6, 1 - 1e-6) # 2025-08-13
        coords3d = inverse_sigmoid(coords3d) # 3D Coodrinates(Frustum)
        coords_position_embeding = self.position_encoder(coords3d) # PE_K
        
        return coords_position_embeding.view(B, N, self.embed_dims, H, W), coords_mask

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is PETRHead_seg:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                # '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(PETRHead_seg, self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)
    
    def forward(self, img_feats, I, E, img_size, e2w=None, l2e=None, isTrain=True):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        pad_size = img_size # non_padding

        
        x = img_feats[self.position_level]
        batch_size, num_cams = x.size(0), x.size(1)

        if self.with_detach:
            current_frame = x[:, :6]
            past_frame = x[:, 6:]
            x = torch.cat([current_frame, past_frame.detach()], 1)
        
        # input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]
        input_img_h, input_img_w = pad_size
        

        if pad_size != img_size:
            masks = x.new_ones(
                (batch_size, num_cams, input_img_h, input_img_w))
            for img_id in range(batch_size):
                for cam_id in range(num_cams):
                    # img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]
                    img_h, img_w = img_size
                    masks[img_id, cam_id, :img_h, :img_w] = 0
        else:
            masks = x.new_zeros(
                (batch_size, num_cams, input_img_h, input_img_w))


        x = self.input_proj(x.flatten(0,1))
        x = x.view(batch_size, num_cams, *x.shape[-3:])

        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks, size=x.shape[-2:], mode='nearest').to(torch.bool)

        key_padding_mask = masks
        if self.with_position:
            coords_position_embeding, coords_mask = self.position_embeding(img_feats, I, E, pad_size, masks)
            key_padding_mask = masks | coords_mask # 2025-08-13
            
            if self.with_se:
                coords_position_embeding = self.se(coords_position_embeding.flatten(0,1), x.flatten(0,1)).view(x.size())

            pos_embed = coords_position_embeding

            if self.with_multiview:
                sin_embed = self.positional_encoding(masks)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed  # PE_Key + Key
            else:
                pos_embeds = []
                for i in range(num_cams):
                    xy_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(xy_embed.unsqueeze(1))
                sin_embed = torch.cat(pos_embeds, 1)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
        else:
            if self.with_multiview:
                pos_embed = self.positional_encoding(masks)
                pos_embed = self.adapt_pos3d(pos_embed.flatten(0, 1)).view(x.size())
            else:
                pos_embeds = []
                for i in range(num_cams):
                    pos_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(pos_embed.unsqueeze(1))
                pos_embed = torch.cat(pos_embeds, 1)

        query=self.query_embedding_lane(pos2posemb2d(self.reference_points_lane.to(pos_embed))) # PE_Query b dim 25 25

        tf_outputs, _ = self.transformer_lane(x, key_padding_mask, query, pos_embed)
        tf_outputs = torch.nan_to_num(tf_outputs) # [6, b, 625(25*25), 256]
        queries = tf_outputs

        if self.single_decoder:
            all_outputs = []
            queries_lvl = queries[-1].contiguous().view(x.size(0),25,25,-1).permute(0,3,1,2) # [b, 256, 25, 25]
            decoder_output = self.branches[-1](queries_lvl) # [b, class, 200, 200]
                
            output = {}
            for idx, key in enumerate(self.args.targets):
                output[key] = [decoder_output[:, idx:idx+1]] # [b, 1, 200, 200]
                
            if isTrain and self.args.get_height:
                height_output = self.branches_height[-1](queries_lvl)
                output['height'] = [height_output]

        else:
            for lvl in range(self.num_pred): # 6 Decoders
                queries_lvl = queries[lvl].contiguous().view(x.size(0),25,25,-1).permute(0,3,1,2) # [b, dim, 25, 25]
                output = {}
                output['drivable'] = [self.lane_branches_dri[lvl](queries_lvl)] # [b, 1, 200, 200]
                output['vehicle'] = [self.lane_branches_veh[lvl](queries_lvl)]
                output['pedestrian'] = [self.lane_branches_ped[lvl](queries_lvl)]
                if isTrain and self.args.get_height:
                    output['height'] = [self.lane_branches_hei[lvl](queries_lvl)]

                all_outputs.append(output)
            
        return output, queries_lvl
