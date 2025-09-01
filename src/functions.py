from typing import Dict, Any
import sys
import json
import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import GPUtil
import lpips

from render import render_psf, psf2meas

sys.path.append("../")
import diffoptics as do

# from render import render_psf, psf2meas, render_ff, render_geo_diff, ray_tracing_diff_sphere, ray_tracing_2d_scene

def show_psf(config: Dict[str, Any], lens: do.Lensgroup) -> None:
    # Display PSFs
    for i, wv in enumerate([532]):
        _p, _c, h_stack, _r = render_psf(
            config, lens, wv, plot=False, use_wave=True)
    print(h_stack.shape)
    
    
def display(config: Dict[str, Any], lens: do.Lensgroup) -> None:
    # Display PSFs
    fn_list = config["filelist"]
    meas_stack = np.zeros((config["dim"] + 1, config["dim"] + 1, 3))
    full_stack = np.zeros((config["dim"], config["dim"], 3))
    gt_stack = np.zeros((config["dim"] + 1, config["dim"] + 1, 3))
    # print(lens.d_sensor)
    # save_lens(lens, "/home/ubuntu/DiffWaveOptics/lens/singlet_f56.pkl")
    for i, wv in enumerate([440, 510, 650]):
        torch.cuda.synchronize()
        tic = time.perf_counter()
        proj, chief_pos, h_stack, rms_i = render_psf(
            config, lens, wv, plot=False, use_wave=True)
        print("==== Finish rendering PSFs ====")
        meas, full, gt = psf2meas(
            config, proj, chief_pos, h_stack, wv, i, fn_list)
        print("==== Finish rendering measurement ====")
        torch.cuda.synchronize()
        toc = time.perf_counter()
        print("Time: ", toc - tic)
        meas_stack += meas.detach().cpu().numpy()[..., 0]
        full_stack += full.detach().cpu().numpy()[..., 0]
        gt_stack += gt.cpu().numpy()[..., 0]
    np.save(config["display_folder"] + "/full_stack.npy", full_stack)
    cv2.imwrite(
        config["display_folder"] +
        "rgb_meas_{}.png".format(
            config["physic"]),
        255 *
        meas_stack)
    cv2.imwrite(
        config["display_folder"] +
        "rgb_full_{}.png".format(
            config["physic"]),
        255 *
        full_stack)
    cv2.imwrite(
        config["display_folder"] +
        "rgb_gt_{}.png".format(
            config["physic"]),
        255 *
        gt_stack)