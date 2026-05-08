import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
import math
from utils.functions import inverse_sigmoid

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)


def generate_grid(height: int, width: int):  
    '''
    F.pad : to pad the last 3 dimensions, use (left, right, top, bottom, front, back)
    For example,
       x = torch.zeros(size=(2, 3, 4))
       x = F.pad(x, (a, b, c, d, e, f), value=1)
       x.size() # 2+e+f x 3+c+d x 4+a+b
    '''

    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)       # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                   # 3 h w
    indices = indices[None]                                                 # 1 3 h w

    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """
    sh = h / h_meters
    sw = w / w_meters

    return [
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ]


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()

        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class RandomCos(nn.Module):
    def __init__(self, *args, stride=1, padding=0, **kwargs):
        super().__init__()

        linear = nn.Conv2d(*args, **kwargs)

        self.register_buffer('weight', linear.weight)
        self.register_buffer('bias', linear.bias)
        self.kwargs = {
            'stride': stride,
            'padding': padding,
        }

    def forward(self, x):  # cosine 값 계산
        return torch.cos(F.conv2d(x, self.weight, self.bias, **self.kwargs))


class BEVEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        sigma: int,
        bev_height: int,
        bev_width: int,
        h_meters: int,
        w_meters: int,
        offset: int,
        decoder_blocks: list,
    ):
        """
        Only real arguments are:

        dim: embedding size
        sigma: scale for initializing embedding

        The rest of the arguments are used for constructing the view matrix.

        In hindsight we should have just specified the view matrix in config
        and passed in the view matrix...
        """
        super().__init__()
        # each decoder block upsamples the bev embedding by a factor of 2
        h = bev_height // (2 ** len(decoder_blocks))
        w = bev_width // (2 ** len(decoder_blocks))

        # bev coordinates
        grid = generate_grid(h, w).squeeze(0)
        grid[0] = bev_width * grid[0]
        grid[1] = bev_height * grid[1]

        # map from bev coordinates to ego frame
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)  # 3 3
        V_inv = torch.FloatTensor(V).inverse()                                  # 3 3
        grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')                      # 3 (h w)
        grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)                    # 3 h w

        # egocentric frame
        self.register_buffer('grid', grid, persistent=False)                    # 3 h w
        self.learned_features = nn.Parameter(sigma * torch.randn(dim, h, w))    # d h w

    def get_prior(self):
        return self.learned_features


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head, bias=qkv_bias))

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
        v = rearrange(v, 'b n d h w -> b (n h w) d')

        # Project with multiple heads
        q = self.to_q(q)                                # b (n H W) (heads dim_head)
        k = self.to_k(k)                                # b (n h w) (heads dim_head)
        v = self.to_v(v)                                # b (n h w) (heads dim_head)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads, d=self.dim_head)

        # Dot product attention along cameras
        dot = self.scale * torch.einsum('b n Q d, b n K d -> b n Q K', q, k) # scaled dot product
        dot = rearrange(dot, 'b n Q K -> b Q (n K)')
        att = dot.softmax(dim=-1) # softmax-cross-attention

        # Combine values (image level features).
        a = torch.einsum('b Q K, b K d -> b Q d', att, v)
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads, d=self.dim_head)

        # Combine multiple heads
        z = self.proj(a)

        # Optional skip connection
        if skip is not None:
            z = z + rearrange(skip, 'b d H W -> b (H W) d')

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, 'b (H W) d -> b d H W', H=H, W=W)

        return z

class CrossViewAttention(nn.Module):
    def __init__(
        self,
        feat_height: int,
        feat_width: int,
        feat_dim: int,
        dim: int,
        cfg : dict,
        image_height: int,
        image_width: int,
        qkv_bias: bool,
        heads: int = 4,
        dim_head: int = 32,
        no_image_features: bool = False,
        skip: bool = True,

    ):
        super().__init__()

        # 1 1 3 h w
        image_plane = generate_grid(feat_height, feat_width)[None]
        image_plane[:, :, 0] *= image_width
        image_plane[:, :, 1] *= image_height

        self.register_buffer('image_plane', image_plane, persistent=False)

        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(),
            nn.Conv2d(feat_dim, dim, 1, bias=False))

        if no_image_features:
            self.feature_proj = None
        else:
            self.feature_proj = nn.Sequential(
                nn.BatchNorm2d(feat_dim),
                nn.ReLU(),
                nn.Conv2d(feat_dim, dim, 1, bias=False))

        self.bev_embed = nn.Conv2d(2, dim, 1)
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)

        self.cross_attend = CrossAttention(dim, heads, dim_head, qkv_bias)
        self.skip = skip

        
        self.cfg = cfg
        if self.cfg['geo_mode'] == 'petr':
            self.D = 64
            self.anchor_size = 100
            self.in_channels = 256
            self.embed_dims = 256
            self.position_dim = 4*self.D

            self.input_proj = nn.Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)
            
            self.position_encoder = nn.Sequential(
                nn.Conv2d(self.position_dim, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )

            nx=ny=round(math.sqrt(self.anchor_size))
            x_ = (torch.arange(nx) + 0.5) / nx
            y_ = (torch.arange(ny) + 0.5) / ny
            xy=torch.meshgrid(x_,y_)
            self.reference_points_lane =torch.cat([xy[0].reshape(-1)[...,None],xy[1].reshape(-1)[...,None]],-1).cuda()
            
            self.query_embedding_lane = nn.Sequential(
                nn.Linear(self.embed_dims*2//2, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )


    def forward(
        self,
        x: torch.FloatTensor,
        bev: BEVEmbedding,
        feature: torch.FloatTensor,
        I_inv: torch.FloatTensor,
        E_inv: torch.FloatTensor,
        ):
        """
        x: (b, c, H, W)
        feature: (b, n, dim_in, h, w)
        I_inv: (b, n, 3, 3)
        E_inv: (b, n, 4, 4)

        Returns: (b, d, H, W)
        """
        b, n, _, _, _ = feature.shape

        # -------------------------
        # image coordinate tensor, x_{i}^{(I)}
        # -------------------------
        # pixel : Image coordinate
        # ?? : World coordinate embedding (x_W)
        pixel = self.image_plane                                                    # 1 1 3 h w
        if self.cfg['geo_mode'] == 'CVT':
            _, _, _, h, w = pixel.shape

            c = E_inv[..., -1:]                                                     # b n 4 1
            c_flat = rearrange(c, 'b n ... -> (b n) ...')[..., None]                # (b n) 4 1 1
            c_embed = self.cam_embed(c_flat)                                        # (b n) d 1 1   (24) 128 1 1
            # -------------------------
            # translation embedding, tau_{k}
            # ------------bev_embednsic rotation (R_{k}^{-1})
            # d_embed : R_{k}^{-1} X K_{k}^{-1} X x_{i}^{(I)} from eq2
            pixel_flat = rearrange(pixel, '... h w -> ... (h w)')                   # 1 1 3 (h w)
            cam = I_inv @ pixel_flat                                                # b n 3 (h w)
            cam = F.pad(cam, (0, 0, 0, 1, 0, 0, 0, 0), value=1)                     # b n 4 (h w)
            d = E_inv @ cam                                                         # b n 4 (h w)
            d_flat = rearrange(d, 'b n d (h w) -> (b n) d h w', h=h, w=w)           # (b n) 4 h w
            d_embed = self.img_embed(d_flat)                                        # (b n) d h w   (24) 128 56 120
            # "d" is changed into "delta" using an MLP

            # -------------------------
            # Normalization for attention
            # -------------------------
            # c_embed : Camera location embedding (tau_k)
            # img_embed : Camera-aware positional embedding vector (delta)
            img_embed = d_embed - c_embed                                           # (b n) d h w
            img_embed = img_embed / (img_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d h w


            # -------------------------
            # map view coordinate, c^{t}
            # -------------------------
            # w_embed : map-view embedding (c)
            # c_embed : Camera location embedding (tau_k)
            world = bev.grid[:2]                                                    # 2 H W
            w_embed = self.bev_embed(world[None])                                   # 1 d H W
            bev_embed = w_embed - c_embed                                           # (b n) d H W
            bev_embed = bev_embed / (bev_embed.norm(dim=1, keepdim=True) + 1e-7)    # (b n) d H W
            query_pos = rearrange(bev_embed, '(b n) ... -> b n ...', b=b, n=n)      # b n d H W
            feature_flat = rearrange(feature, 'b n ... -> (b n) ...')               # (b n) d h w

            # -------------------------
            # image coord. (delta) and image feat. (phi) are merged
            # for key
            # -------------------------
            if self.feature_proj is not None:
                key_flat = img_embed + self.feature_proj(feature_flat) # [delta, pi] # (b n) d h w 
            else:
                key_flat = img_embed                                                 # (b n) d h w

            # -------------------------
            # image feat. is used for value
            # -------------------------
            val_flat = self.feature_linear(feature_flat) # pi                         # (b n) d h w

            # Expand + refine the BEV embedding

                            # x : 'learned_features(= query grid)' from BEVEmbedding  # lab meeting ppt : CVT reproducing, center_gt, CVT&PETR
            query = query_pos + x[:, None]                                            # b n d H W   4, 6, 128, 25, 25
            key = rearrange(key_flat, '(b n) ... -> b n ...', b=b, n=n)               # b n d h w   4, 6, 128, 56, 120
            val = rearrange(val_flat, '(b n) ... -> b n ...', b=b, n=n)               # b n d h w   4, 6, 128, 56, 120

        else:
            raise ValueError(f"Unsupported geo_mode: {self.cfg['geo_mode']}")

        return self.cross_attend(query, key, val, skip=x if self.skip else None)

    def petr_embedding(self, feature, pixel, E_inv, I_inv, masks):
        B, N, _, _, _ = feature.shape
        _, _, C, H, W = pixel.shape
        D = self.D
        coords_h = torch.arange(H, device=pixel[0].device).float()
        coords_w = torch.arange(W, device=pixel[0].device).float() 
        coords_d = torch.arange(start=1, end=D+1, step=1, device=pixel[0].device).float() # depth
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_d])) #.permute(1, 2, 3, 0) # C, H, W, D
        coords = coords.view(1, 1, C, H, W, D) #.repeat(B, N, 1, 1, 1, 1)
        coords = rearrange(coords, 'b n c h w d -> b n c (h w d)')

        coords_i = torch.matmul(I_inv, coords)
        coords_i = F.pad(coords, (0, 0, 0, 1, 0, 0, 0, 0), value=1)
        coords3d = torch.matmul(E_inv, coords_i)    # b, n, 4, (h, w, d)

        position_range = [-65, -65, -8.0, 65, 65, 8.0]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - position_range[0]) / (position_range[3] - position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - position_range[1]) / (position_range[4] - position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - position_range[2]) / (position_range[5] - position_range[2])

        coords_mask = None
        if masks:
            coords_mask = (coords3d > 1.0) | (coords3d < 0.0) 
            coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)
            coords_mask = masks | coords_mask.permute(0, 1, 3, 2)
        coords3d = coords3d.view(B*N, -1, H, W)
        coords3d = inverse_sigmoid(coords3d) # 3D Coodrinates(Frustum)
        pos_emb_key = self.position_encoder(coords3d) # PE_K

        return pos_emb_key, coords_mask

    def pos2posemb2d(self, pos, num_pos_feats=128, temperature=10000):
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


class Encoder(nn.Module):
    def __init__(
            self,
            backbone,
            cfg,
    ):
        super().__init__()


        cross_view = {"heads": cfg['CVT']['encoder']["heads"],  # key
                      "dim_head": cfg['CVT']['encoder']["dim_head"], # value
                      "qkv_bias": cfg['CVT']['encoder']["qkv_bias"],
                      "skip": cfg['CVT']['encoder']["skip"],
                      "no_image_features": cfg['CVT']['encoder']["no_image_features"],
                      "image_height": cfg["image"]["h"],
                      "image_width": cfg["image"]["w"]}

        bev_embedding = {"sigma": cfg['CVT']['encoder']["sigma"],
                         "bev_height": cfg["bev"]["h"],
                         "bev_width": cfg["bev"]["w"],
                         "h_meters": cfg["bev"]["h_meters"],
                         "w_meters": cfg["bev"]["w_meters"],
                         "offset": cfg["bev"]["offset"],
                         "decoder_blocks": cfg['CVT']['decoder']["blocks"]}
        
        dim = cfg['CVT']['encoder']['dim']
        middle = cfg['CVT']['encoder']['middle']
        scale = cfg['CVT']['encoder']['scale']

        self.norm = Normalize()
        self.backbone = backbone

        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        assert len(self.backbone.output_shapes) == len(middle)

        cross_views = list()
        layers = list()

        for feat_shape, num_layers in zip(self.backbone.output_shapes, middle):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape

            cva = CrossViewAttention(feat_height, feat_width, feat_dim, dim, cfg, **cross_view)
            cross_views.append(cva)

            layer = nn.Sequential(*[ResNetBottleNeck(dim) for _ in range(num_layers)])
            layers.append(layer)

        self.bev_embedding = BEVEmbedding(dim, **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)

    def forward(self, batch):

        image = batch['image'][:, 0]
        b, n, _, _, _ = image.shape
        image = image.flatten(0, 1)  # b n c h w

        I_inv = batch['intrinsics_inv'][:, 0]
        E_inv = batch['extrinsics_inv'][:, 0]

        features = [self.down(y) for y in self.backbone(self.norm(image))]

        x = self.bev_embedding.get_prior()              # d H W
        x = repeat(x, '... -> b ...', b=b)              # b d H W

        for cross_view, feature, layer in zip(self.cross_views, features, self.layers):
            feature = rearrange(feature, '(b n) ... -> b n ...', b=b, n=n)

            x = cross_view(x, self.bev_embedding, feature, I_inv, E_inv) # Cross-Attention
            x = layer(x) # Self-Attention

        return x
