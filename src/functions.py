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

from render import render_psf

sys.path.append("../")
import diffoptics as do

# from render import render_psf, psf2meas, render_ff, render_geo_diff, ray_tracing_diff_sphere, ray_tracing_2d_scene

def show_psf(config: Dict[str, Any], lens: do.Lensgroup) -> None:
    # Display PSFs
    for i, wv in enumerate([532]):
        _p, _c, h_stack, _r = render_psf(
            config, lens, wv, plot=False, use_wave=True)
    print(h_stack.shape)