import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
# from torchvision import transforms 
import copy
import time
from torch.cuda.amp import autocast
from utils.functions import check_invalid
from models.fpn import FPN

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)

def generate_grid(height: int, width: int, pillar=None):
    """
    Generates a grid for BEV (x, y) or voxel (x, y, z) with homogeneous coordinate (1).
    Returns shape: (3, H, W) or (4, H, W, Z)
    """
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    if pillar is None:
        indices = torch.stack(torch.meshgrid(xs, ys, indexing='xy'), 0)  # (2, h, w)
        homogeneous = torch.ones(1, height, width)  # Homogeneous coordinate
        indices = torch.cat([indices, homogeneous], dim=0)  # (3, h, w)
    else:
        zs = torch.linspace(0, 1, pillar)
        indices = torch.stack(torch.meshgrid(xs, ys, zs, indexing='ij'), 0)
        indices = indices.permute(0, 2, 1, 3)  # (3, h, w, p)
        homogeneous = torch.ones(1, height, width, pillar)
        indices = torch.cat([indices, homogeneous], dim=0)

    return indices


def inverse_get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """
    Inverse of the view matrix for transforming 'BEV coordinates' back to 'camera coordinates'.
    """

    sh = h / h_meters 
    sw = w / w_meters 

    return [
        [0., sh, -h*offset-h/2.],  # Inverted Y axis (back to camera view)
        [sw, 0., -w / 2.],         # Inverted X axis (back to camera view)
        [0., 0., 1.]               # Homogeneous coordinate
    ]


class ImageEmbedding(nn.Module):
    def __init__(self, 
                 args,
                 dim, 
                 img_feat_height, 
                 img_feat_width,
                 sigma, 
                 n_cam
                 ):

        super().__init__()

        # image coordinate grid 
        self.img_feat_grid = generate_grid(img_feat_height, img_feat_width, None)  # 3 feat_h feat_w
        self.img_feat_grid[0] *= (args.img_w / img_feat_width)
        self.img_feat_grid[1] *= (args.img_h / img_feat_height)

        V = inverse_get_view_matrix(args.bev_h, args.bev_w, h_meters=100.0, w_meters=100.0, offset=0.0)
        V = torch.FloatTensor(V)
        self.img_feat_grid = V @ rearrange(self.img_feat_grid, 'd h w -> d (h w)')
        self.img_feat_grid = rearrange(self.img_feat_grid, 'd (h w) -> 1 1 d h w', h=img_feat_height, w=img_feat_width) 
        self.register_buffer('_img_feat_grid', self.img_feat_grid, persistent=False)

        # learnable image feature query
        self.learned_image_emb = nn.Parameter(sigma * torch.randn(n_cam, dim, img_feat_height, img_feat_width))

    def get_prior(self):
        return self.learned_image_emb

class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        # Define the query (from image), and key/value (from BEV) projection layers
        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

        # Final projection after attention
        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def forward(self, q, k, v, skip=None):
        """
        q: (b n d H W)
        k: (b n d h w)
        v: (b n d h w)
        """
        _, _, _, H, W = q.shape

        # Move feature dim to last for multi-head proj
        q = rearrange(q, 'b n d H W -> b n (H W) d')
        k = rearrange(k, 'b n d h w -> b n (h w) d')
        v = rearrange(v, 'b n d h w -> b n (h w) d')


        # Project with multiple heads -> Multi-Head Attention
        q = self.to_q(q)                                # b (n H W) (heads dim_head)
        k = self.to_k(k)                                # b (n h w) (heads dim_head)
        v = self.to_v(v)                                # b (n h w) (heads dim_head)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        dot = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q, k) # scaled dot product
        att = dot.softmax(dim=-1) # softmax-cross-attention

        # Combine values (image level features).
        attn = torch.einsum('b n Q K, b n K d -> b n Q d', att, v)
        attn = rearrange(attn, '(b m) n Q d -> b n Q (m d)', m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(attn)

        # Optional skip connection
        if skip is not None:
            z = z + rearrange(skip, 'b n d H W -> b n (H W) d')

        z = self.prenorm(z)  # Add & Norm
        z = z + self.mlp(z)  # z + Feed Forward
        z = self.postnorm(z) # Add & Norm
        z = rearrange(z, 'b n (H W) d -> (b n) d H W', H=H, W=W)
        return z


class InverseCrossViewAttention(nn.Module):
    def __init__(self, 
                 args,
                 dim,
                 bev_feat_dim, 
                 bev_feat_height, 
                 bev_feat_width, 
                 pillar,
                 heads,
                 dim_head,
                 qkv_bias,
                 skip,
                 no_image_features):
        super().__init__()

        self.args = args

        # BEV coordinate grid
        bev_feat_grid = generate_grid(bev_feat_height, bev_feat_width, pillar)  # 4 bf_h, bf_w
        bev_feat_grid = bev_feat_grid.unsqueeze(0)                            # 1 4 bf_h, bf_w
        bev_feat_grid[:,0] *= bev_feat_width
        bev_feat_grid[:,1] *= bev_feat_height
        if pillar is not None:
            bev_feat_grid[:,2] *= pillar
        self.register_buffer('_bev_feat_grid', bev_feat_grid, persistent=False)

        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(bev_feat_dim),
            nn.ReLU(),
            nn.Conv2d(bev_feat_dim, dim, 1, bias=False))


        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm2d(bev_feat_dim),
                nn.ReLU(),
                nn.Conv2d(bev_feat_dim, dim, 1, bias=False))

            
        # self.bev_feat_conv = torch.nn.Conv2d(bev_feat_dim, bev_feat_dim, kernel_size=1, device='cuda')
        # self.bev_feat_conv.requires_grad_(True)
        if pillar is not None:
            self.bev_embed = nn.Conv2d(2*pillar, dim, 1, bias=False)
        else : 
            self.bev_embed = nn.Conv2d(2, dim, 1, bias=False)
        self.img_embed = nn.Conv2d(3, dim, 1)  
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        self.cross_attention = CrossAttention(dim, heads, dim_head, qkv_bias)
        self.skip = skip

    def forward(self, 
                x,
                img_embedding, 
                bev_feature, 
                intrinsic, 
                extrinsic):

        b, n, _, _ = intrinsic.shape

        # -----------------
        # BEV GT coordinate
        bev_feat_grid = self._bev_feat_grid                                        
        if bev_feat_grid.ndim == 4:
            _, _, bf_h, bf_w = bev_feat_grid.shape 
        elif bev_feat_grid.ndim == 5:
             _, _, bf_h, bf_w, bf_p = bev_feat_grid.shape

        intrinsic_homog = torch.eye(4).repeat(b, n, 1, 1).to(bev_feat_grid.device) # b n 3 3
        intrinsic_homog[:, :, :3, :3] = intrinsic                                  # b n 4 4

        extrinsic = extrinsic.to(bev_feat_grid.device)
        
        if bev_feat_grid.ndim == 4:
            Xw = rearrange(bev_feat_grid, '... h w -> ... (h w)')              # 1 3 (bf_h, bf_w)
            Xw = F.pad(Xw, (0, 0, 0, 1), value=1)                              # 1 4 (bf_h, bf_w)
        elif bev_feat_grid.ndim == 5:
            Xw = rearrange(bev_feat_grid, '... h w p -> ... (h w p)')          # 1 4 (bf_h, bf_w, bf_p)

        #with autocast(enabled=False):
        Xi = intrinsic_homog @ extrinsic @ Xw              # d*Xi = K@R@Xw     # b n 4 (bf_h, bf_w, bf_p)
        Xi = check_invalid(Xi, 'Xi')

        valid_mask = Xi[:, :, 2, :] > 0.0
        z = Xi[:, :, 2, :].clamp(min=0.0) 

        Xi[:, :, 0, :] = torch.where(valid_mask, Xi[:, :, 0, :] / z, torch.zeros(1, device=Xi.device, dtype=Xi.dtype))
        Xi[:, :, 1, :] = torch.where(valid_mask, Xi[:, :, 1, :] / z, torch.zeros(1, device=Xi.device, dtype=Xi.dtype))

        Xi[:, :, 0, :] = Xi[:, :, 0, :].clamp(0, self.args.img_w - 1) 
        Xi[:, :, 1, :] = Xi[:, :, 1, :].clamp(0, self.args.img_h - 1) 


        Xi_2D = Xi[:, :, :2, :]                                                  # b n 2 (bf_h, bf_w, bf_p)

        if bev_feat_grid.ndim == 4:
            Xi_flat = rearrange(Xi_2D, 'b n d (h w) -> (b n) d h w', h=bf_h, w=bf_w)
        elif bev_feat_grid.ndim == 5:
            Xi_flat = rearrange(Xi_2D, 'b n d (h w p) -> (b n) (d p) h w', h=bf_h, w=bf_w, p=bf_p) # (b n) (2 bf_p) bf_h, bf_w
        else:
            raise ValueError(f"Unexpected bev_feat_grid ndim: {bev_feat_grid.ndim}")
        bev_embed = self.bev_embed(Xi_flat)                                    # (b n) d bf_h, bf_w 

        t = extrinsic[..., -1:]                                                # b n 4 1
        t_flat = rearrange(t, 'b n ... -> (b n) ...')[..., None]               # (b n) 4 1 1
        t_embed = self.cam_embed(t_flat)                                       # (b n) d 1 1   
        bev_embed -= t_embed
        bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)

        # -----------------
        # Image coordinate
        img_feat_grid = img_embedding._img_feat_grid                              # 1 1 3 if_h if_w
        img_feat_grid = img_feat_grid.expand(b, n, -1, -1, -1)                    # b n 3 if_h if_w
        img_feat_grid = rearrange(img_feat_grid, 'b n ... -> (b n) ...')          # (b n) 3 if_h, if_w
        img_feat_embed = self.img_embed(img_feat_grid)                            # (b n) d if_h, if_w
        img_feat_embed -= t_embed
        img_feat_embed = img_feat_embed / (img_feat_embed.norm(dim=1, keepdim=True) + 1e-7)

        query_pos = rearrange(img_feat_embed, '(b n) ... -> b n ...', b=b, n=n) 

        # -------------------------
        # image coord. (delta) and image feat. (phi) are merged
        # for key
        # -------------------------
        if self.feature_proj is not None:
            if bev_feature.ndim==4 and bev_feature.size(0)!=b*n: 
                bev_embed = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n) # b n d h w
                bev_feat_proj = self.feature_proj(bev_feature).unsqueeze(1)        # b 1 d h w
                key_flat = bev_embed + bev_feat_proj                               # b n d h w
                key_flat = rearrange(key_flat, 'b n ... -> (b n) ...', b=b, n=n)   # (b n) d h w
            else:
                key_flat = bev_embed + self.feature_proj(bev_feature)            # (b n) d h w
        else:
            key_flat = bev_embed                                                 # (b n) d h w

        # -------------------------
        # image feat. is used for value
        # x : learned query embedding(= query grid) 
        # -------------------------
        query = query_pos + x                                                     # b n d H W   
        key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)               # b n d h w   
        
        val_flat = self.feature_linear(bev_feature) # pi                          # (b n) d h w  
        
        if val_flat.ndim==4 and val_flat.size(0)!=b*n:
            val = val_flat.unsqueeze(1).expand(-1, n, -1, -1, -1)  # b n d h w
        else:
            val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)  # b n d h w  
        return self.cross_attention(query, key, val, skip=x if self.skip else None)
    


class Encoder(nn.Module):
    # BEV2Image
    def __init__(
            self,
            cfg,
            args,
            ivt_backbone,
            bev_feat_shapes,
            img_feat_shapes,  
    ):
        super().__init__()

        self.args = args
        self.cfg = cfg

        cross_view = {"heads": cfg['CVT']['encoder']["heads"],  # key
                      "dim_head": cfg['CVT']['encoder']["dim_head"], # value
                      "qkv_bias": cfg['CVT']['encoder']["qkv_bias"],
                      "skip": cfg['CVT']['encoder']["skip"],
                      "no_image_features": cfg['CVT']['encoder']["no_image_features"],
                      }

        sigma = cfg['CVT']['encoder']['sigma']
        dim = cfg['IVT']['encoder']['dim'] 
        nbl = cfg['IVT']['encoder']['num_bottleneck_layers']
        pillar = cfg['IVT']['encoder']['pillar'] 
        if pillar == 0 : pillar = None

        n_cam = cfg['image']['n_cam']
        self.ivt_backbone = ivt_backbone
        
        self.fpn_on = cfg['IVT']['encoder']['fpn']['use']
        reverse_outputs = cfg['IVT']['encoder']['fpn']['reverse_outputs'] #bool(args.reverse_outputs)
        fpn_outC = cfg['IVT']['encoder']['fpn']['out_channels']
        if self.fpn_on:
            in_channels_list = [feat[1] for feat in bev_feat_shapes]
            self.fpn = FPN(in_channels_list, out_channels=fpn_outC, reverse_outputs=reverse_outputs)
            
        afn = cfg['IVT']['encoder']['align_feature_num']
        bln = cfg['IVT']['encoder']['bev_layer_nums']
        assert afn in bln, f"'align_feature_num' {afn} is not in 'bev_layer_nums'."
        if reverse_outputs:
            self.idx = bln[::-1].index(afn)
        else:
            self.idx = bln.index(afn) # - len(bln)
            
        self.resizers = {}
        self.non_zero_indices = None
        self.valid_masks = None
        self.boundaries = None
        self.stored_boundaries = []


        img_embeddings = nn.ModuleList()
        for ifs in img_feat_shapes: 
            _, feat_dim, feat_height, feat_width = ifs
            img_emb = ImageEmbedding(args,
                                    feat_dim, 
                                    feat_height, 
                                    feat_width,
                                    sigma, 
                                    n_cam)
            img_embeddings.append(img_emb)


        _cross_views = nn.ModuleList()
        _layers = nn.ModuleList()

        if reverse_outputs:
            bev_feat_shapes = bev_feat_shapes[::-1]
        for bfs in bev_feat_shapes:
            _, feat_dim, feat_height, feat_width = bfs
            if self.fpn_on:
                feat_dim = fpn_outC
            icva = InverseCrossViewAttention(args,
                                             dim,
                                             feat_dim, 
                                             feat_height, 
                                             feat_width, 
                                             pillar,
                                             **cross_view)
            _cross_views.append(icva)
            
            layer = nn.Sequential(*[ResNetBottleNeck(dim) for _ in range(nbl)])
            _layers.append(layer)

        assert len(bev_feat_shapes) % len(img_feat_shapes) == 0, f"bev_feat_shapes:{len(bev_feat_shapes)}, img_feat_shapes:{len(img_feat_shapes)}"
        block_num = len(bev_feat_shapes)//len(img_feat_shapes)
        cross_views = nn.ModuleList([_cross_views[i:i+block_num] for i in range(0, len(_cross_views), block_num)])
        layers = nn.ModuleList([_layers[i:i+block_num] for i in range(0, len(_layers), block_num)])

        self.img_embeddings = nn.ModuleList(img_embeddings)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)

    def forward(self, batch):

        intrinsic = batch['intrinsics2']  # b n 3 3
        b, n, _, _ = intrinsic.shape
        if batch['extrinsics'].ndim == 5:
            extrinsic = batch['extrinsics'][-1]  # b n 4 4
        elif batch['extrinsics'].ndim == 4:
            extrinsic = batch['extrinsics']
        else:
            raise ValueError(f"Unexpected extrinsics ndim: {batch['extrinsics'].ndim}")
            
        if self.ivt_backbone is not None:
            bev_gt = batch['bev']           # b 12 bev_h bev_w
            if bev_gt.size(1) == 12 and not self.args.get_height:
                bev_target = self.get_traget_bev(bev_gt, self.args.targets) # b len(targets) bev_h bev_w
            elif bev_gt.size(1) == 13 and self.args.get_height:          
                bev_target = self.get_traget_bev(bev_gt[:,:-1], self.args.targets) # b len(targets)+height bev_h bev_w
                bev_target = torch.concat([bev_target, bev_gt[:,-1:]], dim=1)
            else:
                bev_target = bev_gt

            device = next(self.ivt_backbone.parameters()).device
            _bev_features = self.ivt_backbone(bev_target.to(device))
            
            if self.fpn_on:
                _bev_features = self.fpn(_bev_features)
            backbone_feature = _bev_features[self.idx]
        else:
            _bev_features = [batch['vt_output']]
            backbone_feature = None

        block_num = len(_bev_features)//len(self.img_embeddings) # [a,b,c,d]→[[a,b],[c,d]] or [a,b]→[[a,b]]
        self.bev_features = [_bev_features[i:i+block_num] for i in range(0, len(_bev_features), block_num)]

        ouputs = []
        for i, img_embedding in enumerate(self.img_embeddings):
            x = img_embedding.get_prior()  # query  # n d H W
            x = repeat(x, '... -> b ...', b=b)           # b n d H W
            for cross_view, layer, bev_feature in zip(self.cross_views[i], 
                                                        self.layers[i],
                                                        self.bev_features[i]
                                                        ):
                
                x = cross_view(x, img_embedding, bev_feature, intrinsic, extrinsic) # Cross-Attention
                x = layer(x)  # (b n) d H W
                x = rearrange(x, '(b n) ... -> b n ...', b=b, n=n)

            x = rearrange(x, 'b n ... -> (b n) ...', b=b, n=n)
            ouputs.append(x)

        return ouputs, backbone_feature

            

    def get_traget_bev(self, bev_all_classes, targets):
        bev_images = [self.return_bev_target(bev_all_classes, target) for target in targets]
        target_bev = torch.cat(bev_images, dim=1) # [b, c, h, w]
        return target_bev

    def return_bev_target(self, label, target):
        '''
        label : b x 12 x h x w
        '''
        label_indices = self.cfg['label_indices'][target]
        
        label = [label[:, idx].max(dim=1, keepdim=True).values for idx in label_indices]
        return torch.cat(label, dim=1)