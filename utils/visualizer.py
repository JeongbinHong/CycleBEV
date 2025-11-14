import os
import torch
from PIL import Image
import numpy as np
from einops import rearrange
import cv2
from matplotlib.pyplot import get_cmap
from einops import rearrange

COLORS = {
    'drivable': (200, 200, 200),
    'vehicle': (0, 100, 255),
    'pedestrian': (255, 0, 0),

    'dividers': (0, 158, 255),
    'nothing': (0, 0, 0)
}

# BEV part -------------------------------
def colorize(x, colormap=None):
    """
    x: (h w) np.uint8 0-255
    colormap
    """
    try:
        return (255 * get_cmap(colormap)(x)[..., :3]).astype(np.uint8)
    except:
        pass

    if x.dtype == np.float32:
        x = (255 * x).astype(np.uint8)

    if colormap is None:
        return x[..., None].repeat(3, 2)

    return cv2.applyColorMap(x, getattr(cv2, f'COLORMAP_{colormap.upper()}'))


def get_colors(semantics):
    return np.array([COLORS[s] for s in semantics], dtype=np.uint8)


def to_image(x):
    return (255 * x).byte().cpu().numpy().transpose(1, 2, 0)


def greyscale(x):
    return (255 * x.repeat(3, 2)).astype(np.uint8)


def resize(src, dst=None, shape=None, idx=0):
    if dst is not None:
        ratio = dst.shape[idx] / src.shape[idx]
    elif shape is not None:
        ratio = shape[idx] / src.shape[idx]

    width = int(ratio * src.shape[1])
    height = int(ratio * src.shape[0])

    return cv2.resize(src, (width, height), interpolation=cv2.INTER_CUBIC)


class BaseViz:
    '''
    BEV Visualization
    '''

    def __init__(self, cfg, args, targets, label_indices, Thresholds):
        self.cfg = cfg
        self.args = args
        self.targets = targets
        self.label_indices = label_indices
        self.Thresholds = Thresholds
        self.colors = get_colors(['drivable', 'vehicle', 'pedestrian', 'nothing']) #'dividers', 

    def return_bev_map(self, label, target='vehicle'):
        '''
        label : b x 12 x h x w
        '''

        label = [label[:, idx].max(dim=1, keepdim=True).values for idx in self.label_indices[target]]
        return torch.cat(label, dim=1)

    def __call__(self, batch, pred, min_visibility=0):

        # visualize GT BEV
        bev_gt = self.vis_gt(batch, min_visibility) # list of [h x w x 3, ....]

        # visualize Pred
        bev_pred = self.vis_ped(pred) # list of [h x w x 3, ....]
        if self.cfg['use_temporal']:
            if self.args.vt_model == 'BEVFormer':
                images = batch['image'][-1]
            elif self.args.vt_model == 'PETR':
                images = batch['image'][:,-1]
                images = rearrange(images, 'b n c h w -> (b n) c h w')
        else:
            images = batch['image']
        bn, c, h, w = images.shape
        images = images.view(bn//6, 6, c, h, w)

        # visualize images
        output = []
        batch_size = images.size(0)
        for b in range(batch_size):
            # images = batch['image'][b] # num_cam x c x h x w
            imgs = [to_image(images[b][n]) for n in range(images.size(1))]
            imgs = np.vstack((np.hstack(imgs[:3]), np.hstack(imgs[3:])))
            # imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
            imgs = cv2.resize(imgs, (0, 0), fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)

            h, w, c = imgs.shape
            gt = cv2.resize(bev_gt[b], dsize=(h, h), interpolation=cv2.INTER_NEAREST)
            pr = cv2.resize(bev_pred[b], dsize=(h, h), interpolation=cv2.INTER_NEAREST)

            output.append(np.hstack((imgs, gt, pr)))

        return output

    def vis_gt(self, batch, min_visibility=0):
        '''
        bev : b x 12 x h x w
        output : [h x w x 3, h x w x 3, ...]
        '''
        bev = batch['bev']
        min_visibility = 2
        veh_vis = batch['visibility'][:, :1]
        ped_vis = batch['visibility'][:, 1:]
        veh_mask_visT = veh_vis >= min_visibility
        veh_mask_visF = veh_vis < min_visibility
        ped_mask_visT = ped_vis >= min_visibility
        ped_mask_visF = ped_vis < min_visibility

        # b x 1 x h x w, tensor
        veh = self.return_bev_map(bev, target='vehicle').permute(0, 2, 3, 1)
        ped = self.return_bev_map(bev, target='pedestrian').permute(0, 2, 3, 1)
        dri = self.return_bev_map(bev, target='drivable').permute(0, 2, 3, 1)
        # div = self.return_bev_map(bev, target='divider').permute(0, 2, 3, 1)

        # veh_mask_visT = veh_mask_visT.expand_as(veh)
        # veh_mask_visF = veh_mask_visF.expand_as(veh)
        # ped_mask_visT = ped_mask_visT.expand_as(ped)
        # ped_mask_visF = ped_mask_visF.expand_as(ped)

        # veh_visT_indices = veh_mask_visT.nonzero(as_tuple=True)
        # veh_visF_indices = veh_mask_visF.nonzero(as_tuple=True)
        # ped_visT_indices = ped_mask_visT.nonzero(as_tuple=True)
        # ped_visF_indices = ped_mask_visF.nonzero(as_tuple=True)

        # veh_visT = veh[veh_visT_indices]
        # veh_visF = veh[veh_visF_indices]
        # ped_visT = veh[ped_visT_indices]
        # ped_visF = veh[ped_visF_indices]

        output = []
        # if 'divider' in self.targets:
        #     bev_all = torch.cat((dri, div, veh, ped), dim=-1).numpy() # b x h x w x 4
        # else:
        #     bev_all = torch.cat((dri, veh, ped), dim=-1).numpy() 
        bev_all = torch.cat((dri, veh, ped), dim=-1).cpu().numpy()
        b, h, w, c = bev_all.shape
        for i in range(b):

            bev_cur = bev_all[i] # h w c

            # Prioritize higher class labels
            eps = (1e-5 * np.arange(c))[None, None]  # 1 1 c
            idx = (bev_cur + eps).argmax(axis=-1)  # h w
            val = np.take_along_axis(bev_cur, idx[..., None], -1)

            # Spots with no labels are light grey
            empty = np.uint8(COLORS['nothing'])[None, None]  # 1 1 3
            result = (val * self.colors[idx]) + ((1 - val) * empty)
            output.append(np.uint8(result))

        return output

    def vis_ped(self, pred):
        '''
        bev : b x 12 x h x w
        output : [h x w x 3, h x w x 3, ...]
        '''

        batch_size, _, h, w = pred[self.targets[0]][0].size()

        veh = np.zeros(shape=(batch_size, h, w, 1))
        if ('vehicle' in self.targets):
            thr = pred['vehicle'][0].permute(0, 2, 3, 1) > self.Thresholds['vehicle']
            veh[thr.detach().to('cpu').numpy()] = 1

        ped = np.zeros(shape=(batch_size, h, w, 1))
        if ('pedestrian' in self.targets):
            thr = pred['pedestrian'][0].permute(0, 2, 3, 1) > self.Thresholds['pedestrian']
            ped[thr.detach().to('cpu').numpy()] = 1

        div = np.zeros(shape=(batch_size, h, w, 1))
        if ('divider' in self.targets):
            thr = pred['div'][0].permute(0, 2, 3, 1) > self.Thresholds['divider']
            div[thr.detach().to('cpu').numpy()] = 1

        dri = np.zeros(shape=(batch_size, h, w, 1))
        if ('drivable' in self.targets):
            thr = pred['drivable'][0].permute(0, 2, 3, 1) > self.Thresholds['drivable']
            dri[thr.detach().to('cpu').numpy()] = 1

        output = []
        bev_all = np.concatenate((dri, veh, ped), axis=-1)
        batch, h, w, c = bev_all.shape
        for b in range(batch):

            bev_cur = bev_all[b] # h w c

            # Prioritize higher class labels
            eps = (1e-5 * np.arange(c))[None, None]  # 1 1 c
            idx = (bev_cur + eps).argmax(axis=-1)  # h w
            val = np.take_along_axis(bev_cur, idx[..., None], -1)

            # Spots with no labels are light grey
            empty = np.uint8(COLORS['nothing'])[None, None]  # 1 1 3
            result = (val * self.colors[idx]) + ((1 - val) * empty)
            output.append(np.uint8(result))

        return output
    
    
# -------------------------------------------------------

def label_to_color(args, maps, background=True):
    """
    multi_label : [C, H, W]
    binary_label: [H, W]
    """
    if len(maps.shape) == 3:
        h, w = maps.shape[-2:]
        color_map = np.zeros((h, w, 3), dtype=np.uint8)

        if background:
            # background(0)
            color_map[maps[0] == 1] = [0, 0, 0]  # black
            # driveable(1)
            color_map[maps[1] == 1] = [200, 200, 200]  # G
            # vehicle(2)
            color_map[maps[2] == 1] = [0, 100, 255]  # B
            # pedestrian(3)
            color_map[maps[3] == 1] = [255, 0, 0]  # R
        else:
            # driveable(0)
            color_map[maps[0] == 1] = [200, 200, 200]  # G
            # vehicle(1)
            color_map[maps[1] == 1] = [0, 100, 255]  # B
            # pedestrian(2)
            color_map[maps[2] == 1] = [255, 0, 0]  # R
        return color_map
    
    else:
        h, w = maps.shape
        color_map = np.zeros((h, w, 3), dtype=np.uint8)

        for i, target in enumerate(args.targets, start=1):
            color_map[maps==i] = COLORS[target]

        return color_map
    
def save_images(args, idx, output_dir, images=None, preds=None, labels=None, filepath=None, norm_type='imagenet', rank=0, png_mode='RGB'):
    output_dir_idx = os.path.join(output_dir, str(idx))
    os.makedirs(output_dir_idx, exist_ok=True)

    n = 6
    b = images.size(0) // n
    if images is not None:
        images = rearrange(images, '(b n) ... -> b n ...', n=n)
    if preds is not None:
        preds = rearrange(preds, '(b n) ... -> b n ...', n=n)
    if labels is not None:
        labels = rearrange(labels, '(b n) ... -> b n ...', n=n)

    for i in range(b):
        for j in range(n):

            if images is not None:
                # 원본 이미지 복원
                original_image = images[i][j].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]

                # processor의 정규화 복원
                if norm_type == 'imagenet':
                    mean = [0.485, 0.456, 0.406]
                    std = [0.229, 0.224, 0.225]

                elif norm_type == 'nuscenes':
                    mean = [0.381, 0.386, 0.378]
                    std = [0.187, 0.184, 0.190]

                elif norm_type == 'nuscenes_topcrop':
                    mean = [0.363, 0.367, 0.355]
                    std = [0.173, 0.169, 0.172]

                elif norm_type is None or norm_type.lower() == 'none':
                    mean = np.array([0.0, 0.0, 0.0])
                    std = np.array([1.0, 1.0, 1.0])
                
                else:
                    raise ValueError(f"Unsupported norm_type: {norm_type}")
                
                # 정규화 복원: x = (normalized * std) + mean
                original_image = (original_image * std + mean) * 255.0
                original_image = original_image.clip(0, 255).astype(np.uint8)
                
                if filepath is not None:
                    filename = filepath[j][i].split('/')[-1][:-4]
                else:
                    filename = ''
                original_path = os.path.join(output_dir_idx, f"mb{i}r{rank}n{j}_{filename}_image.png")
                Image.fromarray(original_image).save(original_path)

            if preds is not None:
                # 예측 마스크 저장
                pred_map = preds[i][j].cpu().numpy()
                pred_color_map = label_to_color(args, pred_map)
                if png_mode=='RGBA':
                    pred_color_map = get_rgba_image(pred_color_map)
                pred_path = os.path.join(output_dir_idx, f"mb{i}r{rank}n{j}_{filename}_pred.png") # "b{idx}m{i}r{rank}n{j}_pred.png"
                Image.fromarray(pred_color_map, mode=png_mode).save(pred_path)

            if labels is not None:
                # 레이블 마스크 저장
                label_map = labels[i][j].cpu().numpy()
                label_color_map = label_to_color(args, label_map)
                if png_mode=='RGBA':
                    label_color_map = get_rgba_image(label_color_map)
                label_path = os.path.join(output_dir_idx, f"mb{i}r{rank}n{j}_{filename}_label.png")
                Image.fromarray(label_color_map, mode=png_mode).save(label_path)

def get_rgba_image(inputs):
    alpha = np.ones((inputs.shape[0], inputs.shape[1]), dtype=np.uint8) * 255  # 기본값: 불투명(255)
    mask = (inputs[:, :, 0] == 0) & (inputs[:, :, 1] == 0) & (inputs[:, :, 2] == 0)  # RGB == (0,0,0)
    alpha[mask] = 0

    rgba_image = np.dstack((inputs, alpha))
    return rgba_image