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
import matplotlib.pyplot as plt
import cv2
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.functions import toTS, toNP

# update, 240131
def draw_traj_on_topview(img, traj, obs_len, x_range, y_range, map_size, in_color):
    '''
    traj : seq_len x 2
    '''

    # for displaying images
    axis_range_y = y_range[1] - y_range[0]
    axis_range_x = x_range[1] - x_range[0]
    scale_y = float(map_size - 1) / axis_range_y
    scale_x = float(map_size - 1) / axis_range_x

    # GT trajs --------------------------------------
    col_pels = -(traj[:, 1] * scale_y).astype(np.int32)
    row_pels = -(traj[:, 0] * scale_x).astype(np.int32)

    col_pels += int(np.trunc(y_range[1] * scale_y))
    row_pels += int(np.trunc(x_range[1] * scale_x))

    for j in range(0, traj.shape[0]):
        if (traj[j, 0] != -1000):
            color = in_color
            if (j < obs_len):
                color = (64, 64, 64)
            cv2.circle(img, (col_pels[j], row_pels[j]), 2, color, -1)

    return img

def create_ROI(grid_size, num_grid):
    '''
    traj[i, 0] : y-axis (forward direction)
    traj[i, 1] : x-axis (lateral direction)
    '''
    # grid_size = 0.2
    # num_grid = 100

    r = np.ones(shape=(1, num_grid))
    co = -4
    r_y = co * np.copy(r)
    for i in range(1, num_grid):
        co += grid_size
        r_y = np.concatenate([co * r, r_y], axis=0)
    r_y = np.expand_dims(r_y, axis=2)

    r = np.ones(shape=(num_grid, 1))
    co = grid_size * (num_grid / 2)
    r_x = co * np.copy(r)
    for i in range(1, num_grid):
        co -= grid_size
        r_x = np.concatenate([r_x, co * r], axis=1)
    r_x = np.expand_dims(r_x, axis=2)

    return np.concatenate([r_y, r_x], axis=2)

def pooling_operation(img, x, x_range, y_range, map_size):
    '''
    ROI-pooling feature vectors from feat map

    Inputs
    x : num_agents x 2
    feat_map : 8 x 200 x 200

    Outputs
    pooled_vecs.permute(1, 0) : num_agents x 8
    '''

    x_range_max = x_range[1]
    y_range_max = y_range[1]

    axis_range_y = y_range[1] - y_range[0]
    axis_range_x = x_range[1] - x_range[0]
    scale_y = float(map_size - 1) / axis_range_y
    scale_x = float(map_size - 1) / axis_range_x

    # from global coordinate system (ego-vehicle coordinate system) to feat_map index
    shift_c = np.trunc(y_range_max * scale_y)
    shift_r = np.trunc(x_range_max * scale_x)

    c_pels_f, c_oob = clip(-(x[:, 1] * scale_y) + shift_c, map_size)
    r_pels_f, r_oob = clip(-(x[:, 0] * scale_x) + shift_r, map_size)
    oob_pels = np.logical_or(c_oob, r_oob)

    # 4 neighboring positions
    '''
    -------|------|
    | lu   | ru   |
    |(cur.)|      |
    |------|------|
    | ld   | rd   |
    |      |      |
    |------|------|
    '''

    c_pels = c_pels_f.astype('int')
    r_pels = r_pels_f.astype('int')

    c_pels_lu = np.copy(c_pels)
    r_pels_lu = np.copy(r_pels)

    c_pels_ru, _ = clip(np.copy(c_pels + 1), map_size)
    r_pels_ru, _ = clip(np.copy(r_pels), map_size)

    c_pels_ld, _ = clip(np.copy(c_pels), map_size)
    r_pels_ld, _ = clip(np.copy(r_pels + 1), map_size)

    c_pels_rd, _ = clip(np.copy(c_pels + 1), map_size)
    r_pels_rd, _ = clip(np.copy(r_pels + 1), map_size)

    # feats (ch x r x c)
    feat_rd = img[r_pels_rd.astype('int'), c_pels_rd.astype('int'), :]
    feat_lu = img[r_pels_lu.astype('int'), c_pels_lu.astype('int'), :]
    feat_ru = img[r_pels_ru.astype('int'), c_pels_ru.astype('int'), :]
    feat_ld = img[r_pels_ld.astype('int'), c_pels_ld.astype('int'), :]

    # calc weights, debug 210409
    alpha = r_pels_f - r_pels_lu.astype('float')
    beta = c_pels_f - c_pels_lu.astype('float')

    dist_lu = (1 - alpha) * (1 - beta) + 1e-10
    dist_ru = (1 - alpha) * beta + 1e-10
    dist_ld = alpha * (1 - beta) + 1e-10
    dist_rd = alpha * beta

    # weighted sum of features, debug 210409
    ch_dim = 3
    w_lu = toTS(dist_lu, dtype=torch.float).view(1, -1).repeat_interleave(ch_dim, dim=0)
    w_ru = toTS(dist_ru, dtype=torch.float).view(1, -1).repeat_interleave(ch_dim, dim=0)
    w_ld = toTS(dist_ld, dtype=torch.float).view(1, -1).repeat_interleave(ch_dim, dim=0)
    w_rd = toTS(dist_rd, dtype=torch.float).view(1, -1).repeat_interleave(ch_dim, dim=0)

    w_lu = toNP(w_lu).T
    w_ru = toNP(w_ru).T
    w_ld = toNP(w_ld).T
    w_rd = toNP(w_rd).T

    pooled_vecs = (w_lu * feat_lu) + (w_ru * feat_ru) + (w_ld * feat_ld) + (w_rd * feat_rd)
    pooled_vecs[oob_pels] = 0

    return pooled_vecs

def clip(array, map_size):
    OOB = np.logical_or((array < 0), (array > map_size - 1))

    array[array < 0] = 0
    array[array > map_size - 1] = map_size - 1

    return array, OOB
