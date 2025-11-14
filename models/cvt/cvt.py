import torch
import torch.nn as nn
from models.cvt.decoder import Decoder
from models.cvt.encoder import Encoder

class CrossViewTransformer(nn.Module):
    def __init__(self, cfg, args, img_feat_shapes):
        super().__init__()
        self.args = args
        self.output, num_dec_Q= {}, 0
        for _, key in enumerate(args.targets):
            if (key == 'vehicle' or key == 'pedestrian'): # To predict the center
                self.output[key] = [[num_dec_Q, num_dec_Q + 1], [num_dec_Q + 1, num_dec_Q + 2]]
                num_dec_Q += 2
            else:
                self.output[key] = [[num_dec_Q, num_dec_Q + 1]]
                num_dec_Q += 1
            
        self.encoder = Encoder(img_feat_shapes=img_feat_shapes,
                               cfg=cfg,
                               args=args)

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
        
        if self.args.get_height:
            self.decoder_height = Decoder(dim=cfg['CVT']['decoder']['dim'],
                                        blocks=cfg['CVT']['decoder']['blocks'],
                                        residual=cfg['CVT']['decoder']['residual'],
                                        factor=cfg['CVT']['decoder']['factor'])
            self.to_logits_height = nn.Sequential(
                nn.Conv2d(self.decoder.out_channels, dim_last, 3, padding=1, bias=False),
                nn.BatchNorm2d(dim_last),
                nn.ReLU(inplace=False), 
                nn.Conv2d(dim_last, 1, 1)
                )

    def forward(self, batch, img_feats, isTrain=True):

        batch['intrinsics_inv'] = batch['intrinsics'].inverse()
        batch['extrinsics_inv'] = batch['extrinsics'].inverse()
        
        x = self.encoder(batch, img_feats)  # x: [b, 128, 25, 25]
        y = self.decoder(x)                 # y: [b, 64, 200, 200]
        z = self.to_logits(y)               # z: [b, num_dec_Q, 200, 200]

        output = {}
        for key, item in self.output.items():
            output[key] = []
            for idx in item: output[key].append(z[:, idx[0]:idx[1]])
            
        if isTrain and self.args.get_height:
            h = self.to_logits_height(self.decoder_height(x))
            output['height'] = [h]


        return output, x