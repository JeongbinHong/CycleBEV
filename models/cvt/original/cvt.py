import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.cvt.decoder import Decoder
from models.cvt.original.encoder import Encoder
from models.cvt.efficientnet import EfficientNetExtractor

class CrossViewTransformer(nn.Module):
    def __init__(self, cfg, args):
        super().__init__()
        target = self.args.targets
        self.output, num_dec_Q = {}, 0
        for _, key in enumerate(target):
            if (key == 'vehicle' or key == 'pedestrian'):
                self.output[key] = [[num_dec_Q, num_dec_Q + 1], [num_dec_Q + 1, num_dec_Q + 2]]
                num_dec_Q += 2
            else:
                self.output[key] = [[num_dec_Q, num_dec_Q + 1]]
                num_dec_Q += 1

        backbone = EfficientNetExtractor(layer_names=cfg['CVT']['backbone']['layer_names'],
                                        image_height=cfg['image']['h'],
                                        image_width=cfg['image']['w'],
                                        model_name=cfg['CVT']['backbone']['model_name'])

        self.encoder = Encoder(backbone=backbone,
                               cfg=cfg)

        self.decoder = Decoder(dim=cfg['CVT']['decoder']['dim'],
                               blocks=cfg['CVT']['decoder']['blocks'],
                               residual=cfg['CVT']['decoder']['residual'],
                               factor=cfg['CVT']['decoder']['factor'])

        dim_last = cfg['CVT']['dim_last']
        self.to_logits = nn.Sequential(
            nn.Conv2d(self.decoder.out_channels, dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),
            nn.ReLU(inplace=False),
            nn.Conv2d(dim_last, num_dec_Q, 1))

    def forward(self, batch, dtype, isTrain=True):

        inputs = {}
        for key, value in batch.items():
            if (key == 'intrinsics' or key == 'extrinsics'):
                inputs[key+'_inv'] = batch[key].type(dtype).inverse().cuda()
            else :
                inputs[key] = batch[key].type(dtype).cuda()

        
        x = self.encoder(inputs) 
        y = self.decoder(x) 
        z = self.to_logits(y)

        output = {}
        for key, item in self.output.items():
            output[key] = []
            for idx in item: output[key].append(z[:, idx[0]:idx[1]])
            
        return output