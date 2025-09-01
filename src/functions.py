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
from unet_model import UNet
from utils import get_data_list, save_lens
from plotter import plot_loss_curve
from e2e import train_recon, valid_recon

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
    
    
def e2e_recon(config: Dict[str, Any], lens: do.Lensgroup) -> None:
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if config["dtype"] == "float64":
        datatype = torch.float64
    elif config["dtype"] == "float32":
        datatype = torch.float32
    mse_weight = config["mse_weight"]
    perc_weight = config["perc_weight"]
    batch_size = config["batch_size"]
    batch_num = int(160 // batch_size)
    valid_batch_num = int(config["valid_num"] // batch_size)
    wv_sample = torch.tensor(
        config["wv_sample"],
        dtype=datatype,
        device=device)

    # Initialize the optimizer
    net = UNet(n_channels=3, n_classes=3, do_checkpoint=False).to(device)
    if config["ft_net"]:
        net.load_state_dict(torch.load(config["net_name"]))
    else:
        net.load_state_dict(torch.load("/home/ubuntu/DiffWaveOptics/saved_networks/recon_pretrained_dict.pth"))
    net.train()
    param_list = [{'params': net.parameters(), 'lr': 1e-4}]
    if config["use_deeplens"]:
        param_lens = lens.get_optimizer_params()
        param_list += param_lens
    else:
        if not config["ft_net"]:
            for i in range(len(lens.surfaces)):
                # check if surface[i] has c
                if hasattr(lens.surfaces[i], 'c'):
                    c_clone = torch.clone(lens.surfaces[i].c).detach()
                    c_var_i = c_clone.to(datatype).to(device).requires_grad_()
                    lens.surfaces[i].c = c_var_i
                    param_list.append({'params': c_var_i, 'lr': config["lr"]})
                else:
                    print("Surface {} has no curvature".format(i))
    opt = optim.Adam(param_list, lr=1e-5)
    loss_fn_alex = lpips.LPIPS(net="alex").to(datatype).to(device)
    train_list_all, valid_list_all = get_data_list(config)
    loss_list = torch.zeros(config["max_ep"])
    dataloss_list = torch.zeros(config["max_ep"])
    rmsloss_list = torch.zeros(config["max_ep"])
    patience = 0
    min_loss = 1e8
    best_ep = 0

    for ep in range(config["max_ep"]):
        loss = 0.
        print("===\nEpoch: {}, Patience: {}".format(ep, patience))
        proj_list, h_list = [], []
        total_loss = 0.
        total_percloss = 0.
        total_mseloss = 0.
        total_rmsloss = 0.

        for batch_id in range(batch_num):
            for i in range(len(lens.surfaces)):
                if hasattr(lens.surfaces[i], 'c'):
                    print("surface", i, lens.surfaces[i].c.item(),)
            print("Batch id: ", batch_id, batch_num)
            start_img_id = batch_id * batch_size
            end_img_id = min((batch_id + 1) * batch_size, config["train_num"])
            fn_list = train_list_all[start_img_id:end_img_id]
            loss, loss_m, loss_p, rms = train_recon(config, lens, net, opt, fn_list)
            total_loss += loss.item()
            total_mseloss += loss_m.item()
            total_percloss += 1e3 * loss_p.item()
            total_rmsloss += rms.item()
        print(
            "Total Loss: ",
            total_loss /
            (batch_num),
            "data loss",
            total_percloss /
            (batch_num),
            total_mseloss /
            (batch_num),
            "RMS: ",
            total_rmsloss /
            (batch_num))

        with torch.no_grad():
            net.eval()
            total_loss = 0.
            total_percloss = 0.
            total_mseloss = 0.
            rms = 0.
            proj_list = []
            chief_pos_list = []
            h_list = []
            for i, wv in enumerate(wv_sample):
                proj, chief_pos, h_stack, rms_i = render_psf(
                    config, lens, wv, plot=False)
                rms += rms_i
                proj_list.append(proj)
                chief_pos_list.append(chief_pos)
                h_list.append(h_stack)

            for batch_id in range(valid_batch_num):
                print("==\nBatch id: ", batch_id, valid_batch_num)
                start_img_id = batch_id * batch_size
                end_img_id = min(
                    (batch_id + 1) * batch_size,
                    config["valid_num"])
                fn_list = valid_list_all[start_img_id:end_img_id]
                ep_id = ep * 1000 + batch_id
                plot = (batch_id == valid_batch_num - 1)
                loss_m, loss_p = valid_recon(config, lens, net, fn_list, proj_list, chief_pos_list, h_list, ep_id, loss_fn_alex, plot)
                loss = loss_m + loss_p / config["batch_size"]
                print(
                    "batch loss",
                    loss.item(),
                    loss_m.item(),
                    loss_p.item() /
                    config["batch_size"],
                    rms)
                torch.cuda.empty_cache()
                total_loss += loss.item()
                total_mseloss += loss_m.item()
                total_percloss += loss_p.item()

            print("Validation Loss: ",
                  total_loss / (valid_batch_num),
                  "mse: ",
                  255 * np.sqrt(total_mseloss /
                                (mse_weight * valid_batch_num + 1e-8)),
                  "perc loss",
                  total_percloss /
                  (perc_weight * valid_batch_num * batch_size + 1e-8),
                  "RMS: ",
                  rms)
            save_lens(
                lens,
                config["record_folder"] +
                "/lens/lens_{}.pkl".format(ep))
            torch.cuda.empty_cache()
            net.train()

        loss_list[ep] = total_loss
        dataloss_list[ep] = total_mseloss
        rmsloss_list[ep] = rms
        if total_loss < min_loss:
            min_loss = total_loss
            torch.save(net, config["record_folder"] + "/net_opt.pth")
            torch.save(
                net.state_dict(),
                config["record_folder"] +
                "/net_opt_dict.pth")
            if config["use_deeplens"]:
                lens.write_lens_json(config["record_folder"] + "/lens_opt.json")
            else:
                save_lens(lens, config["record_folder"] + "/lens_opt.pkl")
            best_ep = ep
            patience = 0
        else:
            patience += 1
        print("TOTAL LOSS: ", total_loss, valid_batch_num)
        print("The best performance is obtained at {} epoch with {}".format(best_ep, min_loss))
        plot_loss_curve(config, loss_list, dataloss_list, rmsloss_list, best_ep, ep)
        h_dim = int(np.sqrt(h_list[2][:, 0].shape[0]))
        # h_cent = h_list[2][:, 12].view(h_dim, h_dim)
        h_sum = torch.sum(h_list[2], dim=1).view(h_dim, h_dim)
        plt.imshow(torch.abs(h_sum).detach().cpu().numpy())
        plt.colorbar()
        plt.savefig("{}/psf/u_est_{}_{}.png".format(config["record_folder"], ep, 12))
        plt.close()
        torch.cuda.empty_cache()
        if patience >= config["earlystop"]:
            break
    return