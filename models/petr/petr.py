import torch.nn as nn
from models.petr.detectors.petr3d_seg import Petr3D_seg
import time
class PETR(nn.Module):
    def __init__(self, cfg, args):
        super(PETR, self).__init__() 

        self.cfg = cfg
        self.args = args
        self.petr = Petr3D_seg(cfg=cfg, petr_cfg=cfg['PETR']['model'], args=args)
            
    def forward(self, batch, isTrain=True):
        image = batch['image']
        I = batch['intrinsics']
        E = batch['extrinsics']
        e2w, l2e = None, None
        if 'e2w' in batch:
            e2w = {'cur': batch['e2w'][:, -1], 'prev': batch['e2w'][:, 0]}
        if 'l2e' in batch:    
            l2e = {'cur': batch['l2e'][:, -1], 'prev': batch['l2e'][:, 0]}
        
        output, x, img_feats = self.petr(image, I, E, e2w=e2w, l2e=l2e, isTrain=isTrain)
            
        if self.args.model_name == 'Baseline_PYVA':
            return output, x, img_feats
        else:
            return output, x