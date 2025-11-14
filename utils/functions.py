import json
import os
import glob
import sys
import numpy as np
import shutil
import pickle
from pathlib import Path
import cv2
import time
from tqdm import tqdm
import logging
import traceback
import argparse
from PIL import Image
import cv2
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math


def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

# update, 240131
def read_config(path=None):

    if (path is None):
        cfg = read_json(path='./config/config.json')
        cfg.update(read_json(path=f'./config/data.json'))
        cfg.update(read_json(path=f'./config/model.json'))
        cfg.update(read_json(path=f'./config/loss.json'))

    else:
        file_path = os.path.join(path, 'config.json')
        cfg = read_json(path=file_path)

        file_path = os.path.join(path, f'data.json')
        cfg.update(read_json(path=file_path))

        file_path = os.path.join(path, f'model.json')
        cfg.update(read_json(path=file_path))

        file_path = os.path.join(path, f'loss.json')
        cfg.update(read_json(path=file_path))

    cfg['nuscenes']['dataset_dir'] = check_dataset_path(cfg['nuscenes']['dataset_dir'])

    return cfg

# update, 240131
def config_update(cfg, args):
    '''
    copy data from args to cfg
    '''


    cfg['model_name'] = args.model_name

    # -----------------------------
    # DATA
    # cfg['target'] = args.targets

    # BEV
    cfg['bev']['h'] = args.bev_h
    cfg['bev']['w'] = args.bev_w
    cfg['bev']['h_meters'] = args.bev_h_meters
    cfg['bev']['w_meters'] = args.bev_w_meters
    cfg['bev']['offset'] = args.bev_offset

    cfg['image']['top_crop'] = args.img_top_crop
    cfg['image']['h'] = args.img_h
    cfg['image']['w'] = args.img_w

    # Common
    if args.vt_model in {'BEVFormer', 'PETR'} and cfg['use_temporal']:
        cfg['obs_len'] = max(int(args.target_sample_period // args.past_horizon_seconds)+1, 1)
    else:
        cfg['obs_len'] = 1 #int(args.past_horizon_seconds * args.target_sample_period)
    cfg['pred_len'] = int(args.future_horizon_seconds * args.target_sample_period)
    cfg['target_frame_indices'] = [i for i in range(cfg['obs_len']+cfg['pred_len'])]

    return cfg

def check_dataset_path(path, servers=['dooseop', 'etri', 'ubuntu']):

    if (os.path.exists(path)):
        return path
    else:
        current_machine = None
        for server in servers:
            if (path.find(server) > 0):
                current_machine = server
                break

        for server in servers:
            if (os.path.exists(path.replace(current_machine, server))):
                return path.replace(current_machine, server)

    sys.exit('>> Unable to locate dataset path..')

def get_dtypes(useGPU=True):
    return torch.LongTensor, torch.FloatTensor

def init_weights(m):

    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)

def toNP(x):

    return x.detach().to('cpu').numpy()

def toTS(x, dtype):

    return torch.from_numpy(x).to(dtype)

def save_read_latest_checkpoint_num(path, val, isSave):
    file_name = path + '/checkpoint.txt'
    index = 0
    if (isSave):
        file = open(file_name, "w")
        file.write(str(int(val)) + '\n')
        file.close()
    else:
        if (os.path.exists(file_name) == False):
            print('[Error] there is no such file in the directory')
            sys.exit()
        else:
            f = open(file_name, 'r')
            line = f.readline()
            index = int(line[:line.find('\n')])
            f.close()

    return index

def read_all_saved_param_idx(path):
    ckp_idx_list = []
    files = sorted(glob.glob(os.path.join(path, '*.pt')))
    for i, file_name in enumerate(files):
        start_idx = 0
        for j in range(-3, -10, -1):
            if (file_name[j] == '_'):
                start_idx = j+1
                break
        ckp_idx = int(file_name[start_idx:-3])
        ckp_idx_list.append(ckp_idx)
    return ckp_idx_list[::-1]

def copy_chkpt_every_N_epoch(args):

    def get_file_number(fname):

        # read checkpoint index
        for i in range(len(fname) - 3, 0, -1):
            if (fname[i] != '_'):
                continue
            index = int(fname[i + 1:len(fname) - 3])
            return index

    root_path = args.model_dir + str(args.exp_id)
    save_directory =  root_path + '/copies'
    if save_directory != '' and not os.path.exists(save_directory):
        os.makedirs(save_directory)

    fname_list = []
    fnum_list = []
    all_file_names = os.listdir(root_path)
    for fname in all_file_names:
        if "saved" in fname:
            chk_index = get_file_number(fname)
            fname_list.append(fname)
            fnum_list.append(chk_index)

    max_idx = np.argmax(np.array(fnum_list))
    target_file = fname_list[max_idx]

    src = root_path + '/' + target_file
    dst = save_directory + '/' + target_file
    shutil.copy2(src, dst)

    print(">> {%s} is copied to {%s}" % (target_file, save_directory))

def remove_past_checkpoint(path, num_remain=5, name=None):

    def get_file_number(fname):

        # read checkpoint index
        for i in range(len(fname) - 3, 0, -1):
            if (fname[i] != '_'):
                continue
            index = int(fname[i + 1:len(fname) - 3])
            return index

    fname_list = []
    fnum_list = []

    all_file_names = os.listdir(path)
    for fname in all_file_names:
        if name == None:
            if "saved" in fname:
                chk_index = get_file_number(fname)
                fname_list.append(fname)
                fnum_list.append(chk_index)
        else:
            if f"saved_chk_point_{name}" in fname:
                chk_index = get_file_number(fname)
                fname_list.append(fname)
                fnum_list.append(chk_index)

    if (len(fname_list)>num_remain):
        sort_results = np.argsort(np.array(fnum_list))
        for i in range(len(fname_list)-num_remain):
            del_file_name = fname_list[sort_results[i]]
            os.remove('./' + path + '/' + del_file_name)

def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return np.pad(L, pad_width=(0, int(L.shape[0]*0.1)), mode='edge')

def print_current_train_progress(e, b, num_batchs, time_spent, total_loss):

    if b >= num_batchs-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\r [Epoch %02d] %d / %d (%.4f sec/sample), total loss : %.4f' % (e, b, num_batchs, time_spent, total_loss)),

    sys.stdout.flush()

def print_current_valid_progress(b, num_batchs):

    if b >= num_batchs-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\r >> validation process (%d / %d) ' % (b, num_batchs)),

    sys.stdout.flush()

def print_current_test_progress(b, num_batchs):

    if b >= num_batchs-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\r >> test process (%d / %d) ' % (b, num_batchs)),

    sys.stdout.flush()

def image_printer(idx, bevs, save_path, img_dir, img_name, rank, target_idx=[]) :

    output_dir = os.path.join(save_path, img_dir, str(idx))
    os.makedirs(output_dir, exist_ok=True)
    for i, bev in enumerate(bevs):
        if type(bev) == torch.Tensor :
            if bev.requires_grad : 
                bev = bev.detach()
            bev = bev.permute(1,2,0)
            bev = bev.cpu().numpy()
        
        if bev.shape[-1] in [1, 3]:  # If the last dimension is 1, treat it as a grayscale image
            bev = ((bev - bev.min()) / (bev.max() - bev.min()) * 255).astype(np.uint8)
            if bev.shape[-1] == 1:
                bev = bev.squeeze()  # Remove the last dimension
                image = Image.fromarray(bev, mode='L')  # 'L' mode is for grayscale images
            else:
                image = Image.fromarray(bev) 
        else :
            if len(target_idx)>0 :
                bev = bev[:,:,target_idx]
                bev = np.mean(bev, axis=2, keepdims=True)
                bev = ((bev - bev.min()) / (bev.max() - bev.min()) * 255).astype(np.uint8)
                bev = bev.squeeze()
                image = Image.fromarray(bev, mode='L')

        image.save(os.path.join(output_dir, f'{img_name}_mb{i}r{rank}.png'), 'PNG')
    
def run_wandb(args, wandb):
    if args.load_pretrained:
        run = wandb.init(project=f"{args.wandb_project}", name=args.training_name, save_code=False, resume='must')
    else:
        run = wandb.init(project=f"{args.wandb_project}", name=args.training_name, save_code=False)
    return run

def seed_fixer(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def get_grad_norm(model, log, mode='mean'):
        is_break=False
        max_grad = 0.0
        mean_grad = 0.0
        mean_grad_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad.norm().item()
                if mode == 'mean':
                    mean_grad += grad
                    mean_grad_count += 1
                elif mode == 'max':
                    if max_grad < grad:
                        max_grad = grad
            elif param.requires_grad and param.grad is None:
                #raise ValueError(f"Model has None grad_norm!: {name}")
                log.info(f"Model has None grad_norm!: {name}")
                is_break=True

            if mode=='first':
                break
        if is_break:
            raise ValueError(f"Model has None grad_norm!")

        if mode == 'mean':
            return mean_grad/mean_grad_count
        elif mode == 'max':
            return max_grad

def load_pretrained_models(model, ckp_path, mode='ivt', logger=None, rank=0, strict=True):
    # if mode == 'vt':
    #     strict = False

    ckp_idx = save_read_latest_checkpoint_num(path=ckp_path, val=0, isSave=False)
    file_name = ckp_path + f'/saved_chk_point_{mode}_{ckp_idx}.pt'
    checkpoint = torch.load(file_name, map_location=torch.device('cpu'))
    # if args.ddp:
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    # else:
    #     from collections import OrderedDict
    #     new_state_dict = OrderedDict()
    #     for k, v in checkpoint['model_state_dict'].items():
    #         name = k.replace("module.", "") # removing ‘module.’ from key
    #         new_state_dict[name] = v

    #     pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model.state_dict()}
    #     model.load_state_dict(pretrained_dict, strict=strict)

    if mode in {'ivt', 'vt'} and rank==0:
        logger.info('>> trained parameters are loaded from {%s}' % file_name)
        if 'prev_IoU' in checkpoint:
            logger.info(">> Pretrained Model status : %.4f IoU" % checkpoint['prev_IoU']) 
        elif 'prev_dist_loss' in checkpoint:
            logger.info(">> Pretrained Model status : %.4f MSE" % checkpoint['prev_dist_loss'])

    return model


def cosine_decay(start, end, steps):
    t = np.linspace(0, 1, steps) 
    cosine_values = 0.5 * (1 + np.cos(np.pi * t)) 
    return start + (end - start) * cosine_values


def print_training_info(cfg, args, logger):

    cfg_vt = cfg[f'{args.vt_model}']['training']
    cfg_ivt = cfg['IVT']['training']

    logger.info("--------- %s / %s ----------" % (args.dataset_type, args.model_name))
    #logger.info(" Exp id : %d" % args.exp_id)
    logger.info(" DDP : %d" % args.ddp)
    logger.info(" Num epoch : %d" % cfg_vt['num_epochs'])
    if args.ddp:
        logger.info(" Gpu idx : %s" % args.gpu_idx_ddp)
        logger.info(f" Batch size * GPU num : {args.batch_size} * {len(args.gpu_idx_ddp.split(','))}" )
    else:
        logger.info(" Gpu idx : %d" % args.gpu_idx)
        logger.info(f" Batch size : {args.batch_size}" )

    logger.info(f" Training Name : {args.training_name}")
    # logger.info(" Past horizon seconds (Sec) : %.1f" % args.past_horizon_seconds)
    # logger.info(" Future horizon seconds (Sec) : %.1f" % args.future_horizon_seconds)
    # logger.info(" Target sample period (Hz) : %.1f" % args.target_sample_period)
    logger.info(" Num workers : %d" % args.num_workers)
    logger.info(" Random seed : %d" % args.random_seed)
    logger.info("----------------------------------")
    logger.info(f" Image size : {args.img_h}x{args.img_w}")
    logger.info(f" VT Model : {args.vt_model}")
    logger.info(f" IVT Backbone : {args.ivt_backbone}, (pretrained: {args.ivt_backbone_pretrain})")
    logger.info(f" weights for dri/veh/ped : {args.w_vt_dri}/{args.w_vt_veh}/{args.w_vt_ped}")
    logger.info(f" weights for bev_loss/pvcc_loss/pvgg_loss : {args.w_bev_loss}/{args.w_pvcc_loss}/{args.w_pvgg_loss}/{args.w_feat_loss}")
    logger.info(f" Target class : {args.targets}")
    logger.info(f" Visibility : {args.visibility}")
    logger.info(f" Val ratio : {args.val_ratio}")
    logger.info(f" Augmentation : {args.augmentation}")
    logger.info("----------------VT---------------")
    logger.info(f" Optimizer type : {args.optimizer_type}")
    logger.info(f" Learning rate : {cfg_vt['learning_rate']:.0e}")
    logger.info(f" Weight decay : {cfg_vt['weight_decay']:.0e}")
    logger.info(f" LR scheduling type : {args.lr_schd_type}")
    logger.info(f" div_factor : {cfg_vt['div_factor']}")
    logger.info(f" pct_start : {cfg_vt['pct_start']}" )
    logger.info(f" final_div_factor : {cfg_vt['final_div_factor']}")
    logger.info("---------------IVT----------------")
    logger.info(f" Optimizer type : {args.optimizer_type}")
    logger.info(f" Learning rate : {cfg_ivt['learning_rate']:.0e}")
    logger.info(f" Weight decay : {cfg_ivt['weight_decay']:.0e}")
    logger.info(f" LR scheduling type : {args.lr_schd_type}")
    logger.info(f" div_factor : {cfg_ivt['div_factor']}")
    logger.info(f" pct_start : {cfg_ivt['pct_start']}" )
    logger.info(f" final_div_factor : {cfg_ivt['final_div_factor']}")
    logger.info("---------------------------------")

    if args.load_pretrained:
        logger.info(f" Retrain model: {args.start_epoch-1}.pt, Start epoch: {args.start_epoch}")
        logger.info("----------------------------------")
    
    
class Normalize(nn.Module):
    def __init__(self, norm_type='imagenet'):
        super().__init__()

        if norm_type == 'imagenet':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        
        elif norm_type == 'nuscenes':
            mean = [0.381, 0.386, 0.378]
            std = [0.187, 0.184, 0.190]

        elif norm_type == 'nuscenes_topcrop':
            mean = [0.363, 0.367, 0.355]
            std = [0.173, 0.169, 0.172]

        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        self.mean = self.mean.to(x.device)
        self.std = self.std.to(x.device)
        return (x - self.mean) / self.std

class BinaryThresholdSTE(torch.autograd.Function):
    # Straight-Through Estimator (STE)
    @staticmethod
    def forward(ctx, input, threshold):
        # Forward pass: threshold 적용
        return (input.sigmoid() >= threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass: gradient를 그대로 전달
        return grad_output, None


def gumbel_sigmoid(logits, tau=0.1, eps=1e-10):
    noise = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(noise + eps) + eps)
    y = torch.sigmoid((logits + gumbel_noise) / tau)
    return y


def add_noise_to_bev(bev, eps_min=1e-12, eps_max=1e-7):
    # binary_tensor: 0 또는 1만 포함된 텐서 (float 타입)
    noise = torch.rand_like(bev) * (eps_max - eps_min) + eps_min
    
    # 0이면 +noise, 1이면 -noise
    noisy = bev + (1 - 2 * bev) * noise  # clever trick!
    
    # Clamp to [0, 1]
    noisy = torch.clamp(noisy, min=0.0, max=1.0)
    return noisy


def check_invalid(tensor, name=''):
    if torch.isinf(tensor).any() or torch.isnan(tensor).any():
        isfin = torch.isfinite(tensor)
        _max = tensor[isfin].max()
        _min = tensor[isfin].min()
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=_max, neginf=_min)
        print()
        print(f"⚠️ inf or NaN 발생! : {name}")
    return tensor