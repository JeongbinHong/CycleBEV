"""
Unofficial implementation of BEVFormer (ECCV22)
https://arxiv.org/abs/2203.17270
"""

import torch.nn as nn
import torch.nn.functional as F
from models.bevformer.encoder import BEVFormerEncoder
from models.bevformer.decoder import Decoder


class FPN(nn.Module):

    def __init__(self, dim, sizes, channels, IsReturnWithList=False):
        '''
        dim : target dimension
        sizes = [57, 113, 225, 450]
        channels = [1024, 512, 256, 64]
        '''
        super(FPN, self).__init__()

        self.sizes = sizes
        self.channels = channels
        self.dim_reduce, self.merge = nn.ModuleDict(), nn.ModuleDict()
        for idx, size in enumerate(sizes):
            self.dim_reduce[str(size)] = nn.Conv2d(channels[idx], dim, kernel_size=1, stride=1, padding=0)
            if (idx > 0):
                self.merge[str(size)] = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)

        self.flag = IsReturnWithList

    def upsample_add(self, up, bottom):
        _, _, H, W = bottom.size()
        return F.upsample(up, size=(H, W), mode='bilinear') + bottom

    def forward(self, feats):
        '''
        feats : Dicts
        '''

        outputs, outputs_list  = {}, []
        for idx, size in enumerate(self.sizes):
            if (idx == 0):
                top = feats[str(size)]
                top = self.dim_reduce[str(size)](top)
                outputs[str(size)] = top
                outputs_list.append(top)
            else:
                bottom = feats[str(size)]
                bottom = self.dim_reduce[str(size)](bottom)
                bottom = self.upsample_add(up=outputs[str(self.sizes[idx-1])], bottom=bottom)
                bottom = self.merge[str(size)](bottom)
                outputs[str(size)] = bottom
                outputs_list.append(bottom)

        if (self.flag):
            return outputs_list
        else:
            return outputs

class BEVformer(nn.Module):
    def __init__(self, cfg, args, img_feat_shapes):
        super().__init__()

        self.cfg = cfg
        self.args = args
        self.h_dim = cfg['BEVFormer']['encoder']['dim']
        self.n_cam = cfg['image']['n_cam']
        self.feat_levels = cfg['BEVFormer']['encoder']['feat_levels']
        self.n_lvl = len(self.feat_levels)
        self.depth_candi = cfg['BEVFormer']['encoder']['z_candi']

        h, w = args.bev_h, args.bev_w
        if cfg['BEVFormer']['type'] == 'base': # 200x200
            query_map_size = (200, 200)
            decoding_sizes = [(h, w), (h, w), (h, w)]
        elif cfg['BEVFormer']['type'] == 'small': # 150x150
            query_map_size = (150, 150)
            decoding_sizes = [(150, 150), (h, w), (h, w)]
        elif cfg['BEVFormer']['type'] == 'tiny': # 50x50
            query_map_size = (50, 50)
            decoding_sizes = [(50, 50), (100, 100), (h, w)]
        elif cfg['BEVFormer']['type'] == 'nano': # 25x25
            query_map_size = (25, 25)
            decoding_sizes = [(50, 50), (100, 100), (h, w)]
        else:
            raise ValueError(f"{cfg['BEVFormer']['type']} is not supported for the type of BEVFormer. Please use base, small, or tiny.")

        self.encoder = BEVFormerEncoder(cfg=cfg, query_map_size=query_map_size)
        
        out_channels = cfg['BEVFormer']['encoder']['dim']

        decoder_type = cfg['BEVFormer']['decoder']['type']
        embed_dims = cfg['BEVFormer']['decoder']['dim']
        
        self.heads = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_channels, out_channels, 1))
        
        if decoder_type == 'CVT_conv':
            dim_last = 32
            blocks = [embed_dims, 128, 64]
            decoder = Decoder(dim=embed_dims,
                            blocks=blocks, #[embed_dims for _ in range(num_decoder_blocks)], 
                            sizes=decoding_sizes, 
                            residual=True, 
                            factor=2)
            to_logits = nn.Sequential(
                nn.Conv2d(decoder.out_channels, dim_last, 3, padding=1, bias=False),
                nn.BatchNorm2d(dim_last),
                nn.ReLU(inplace=False), 
                nn.Conv2d(dim_last, len(args.targets), 1)) 
            self.decoder = nn.Sequential(decoder, to_logits)

            if self.args.get_height:
                decoder_height = Decoder(dim=embed_dims,
                                        blocks=blocks, #[embed_dims for _ in range(num_decoder_blocks)],
                                        sizes=decoding_sizes, 
                                        residual=True,
                                        factor=2)
                to_logits_height = nn.Sequential(
                    nn.Conv2d(decoder_height.out_channels, dim_last, 3, padding=1, bias=False),
                    nn.BatchNorm2d(dim_last),
                    nn.ReLU(inplace=False), 
                    nn.Conv2d(dim_last, 1, 1))
                self.decoder_height = nn.Sequential(decoder_height, to_logits_height)

        sizes, channels = [], []
        for shapes in reversed(img_feat_shapes):
            sizes.append(shapes[-1])
            channels.append(shapes[1])
            
        self.fpn = FPN(dim=self.h_dim, sizes=sizes, channels=channels, IsReturnWithList=True)
        


    def forward(self, batch, features, isTrain=True):
        '''
        images : (b n) c h w
        I : b n 3 3
        E : b n 4 4
        '''
        # feature maps
        feat_dict = {}
        for feat in features:
            feat_dict[str(int(feat.size(-1)))] = feat
        x = self.fpn(feat_dict)

        # bevformer encoding
        I = batch['intrinsics']
        E = batch['extrinsics']
        queries = self.encoder(x, I, E)  # b d h w

        output = {}
        x = self.heads(queries) # [b, 256, 25, 25]
        z = self.decoder(x)
            
        for idx, key in enumerate(self.args.targets):
            output[key] = [z[:, idx:idx+1]]
            
        if isTrain and self.args.get_height:
            h = self.decoder_height(x)
            output['height'] = [h]

        return output, x
