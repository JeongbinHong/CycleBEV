import numpy as np
import torch
import torch.nn as nn
from models.icvt.encoder import Encoder
from models.extractors import *

class InverseCrossViewTransformer(nn.Module):
    def __init__(self, cfg, args, ivt_feat_shapes, bev_layer_nums, rank, from_pretrain=True):
        super().__init__()

        if from_pretrain:
            num_classes = cfg['IVT']['pretrained_target_num'] + 1#args.get_height
        else:
            num_classes = len(args.targets) + args.get_height
        if 'resnet' in cfg['IVT']['ivt_backbone']:
            ivt_backbone = ResNet_Extractor(num_classes=num_classes, 
                                            type=cfg['IVT']['ivt_backbone'], 
                                            layer_nums=bev_layer_nums,
                                            pretrained=True,
                                            channel_adapter=True)
        elif 'efficientnet' in cfg['IVT']['ivt_backbone']:
            ivt_backbone = EffNet_Extractor(num_classes=num_classes, 
                                            type=cfg['IVT']['ivt_backbone'], 
                                            layer_nums=bev_layer_nums,
                                            pretrained=True,
                                            channel_adapter=True)
        
        ivt_backbone = ivt_backbone.to(rank)
        sample = torch.randn(1, num_classes, args.bev_h, args.bev_w, device=rank)
        bev_feats = ivt_backbone(sample)
        bev_feat_shapes = [bev_feat.shape for bev_feat in bev_feats]
        
        self.encoder = Encoder(cfg=cfg,
                               args=args,
                               ivt_backbone=ivt_backbone,
                               bev_feat_shapes=bev_feat_shapes,
                               img_feat_shapes=ivt_feat_shapes,
                                )
    
    def forward(self, batch):
        x, bev_features = self.encoder(batch)
        return x, bev_features
    
    
class InverseCrossViewTransformer_without_backbone(nn.Module):
    def __init__(self, cfg, args, bev_feat_shapes, ivt_feat_shapes):
        super().__init__()

        self.encoder = Encoder(cfg=cfg,
                               args=args,
                               ivt_backbone=None,
                               bev_feat_shapes=bev_feat_shapes,
                               img_feat_shapes=ivt_feat_shapes,
                                )
    
    def forward(self, batch):
        x, bev_features = self.encoder(batch)
        return x, bev_features