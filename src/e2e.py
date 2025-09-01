import diffoptics as do
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
import time
from render import render_psf, psf2meas
from plotter import plot_e2e
from utils import save_lens
import lpips
import sys
sys.path.append("../")


def train_recon(config: Dict[str,
                             Any],
                lens: do.Lensgroup,
                net: torch.nn.Module,
                opt: torch.optim.Optimizer,
                fn_list: list) -> Tuple[torch.Tensor,
                                        torch.Tensor,
                                        torch.Tensor,
                                        torch.Tensor]:
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if config["dtype"] == "float64":
        datatype = torch.float64
    elif config["dtype"] == "float32":
        datatype = torch.float32
    dim = config["dim"]
    use_wave = config["physic"] == "wave"
    loss_fn = nn.MSELoss()
    loss_fn_alex = lpips.LPIPS(net="alex").to(
        datatype).to(device)  # best forward scores
    wv_sample = torch.tensor(
        config["wv_sample"],
        dtype=datatype,
        device=device)
    meas = torch.zeros(
        dim + 1,
        dim + 1,
        3,
        config["batch_size"],
        dtype=datatype,
        device=device)
    gt = torch.zeros_like(meas)
    rms = 0
    efl_sum = 0.
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # Generate PSFs and measurements in different wavelengths
    for i, wv in enumerate(wv_sample):
        proj, chief_pos, h_stack, rms_i = render_psf(
            config, lens, wv, plot=False, use_wave=use_wave)
        meas_i, _full_i, gt_i = psf2meas(
            config, proj, chief_pos, h_stack, wv, i, fn_list)
        meas += meas_i
        gt += gt_i
        rms += rms_i
    batch_img = meas.permute(3, 2, 0, 1).to(torch.float)  # .detach()
    batch_gt = gt.permute(3, 2, 0, 1).to(torch.float)
    pred = net(batch_img)
    loss_m = config["mse_weight"] * loss_fn(pred, batch_gt)
    loss_p = config["perc_weight"] * torch.sum(loss_fn_alex(pred, batch_gt))
    if rms.item() > config["rms_thres"]:
        print("WARNING: ENCOUNTER LARGE RMS: {}".format(rms.item()))
        loss = loss_m + loss_p + config["rms_weight"] * rms
        use_wave = False
    else:
        loss = loss_m + loss_p
        use_wave = True
    print(
        "batch loss",
        loss.item(),
        loss_m.item(),
        loss_p.item(),
        rms.item(),
        rms.item() > config["rms_thres"])
    opt.zero_grad()
    loss.backward()
    opt.step()

    torch.cuda.synchronize()
    end_time = time.perf_counter()
    print("time elapsed in batch computation: ", end_time - start_time)
    torch.cuda.empty_cache()
    return loss, loss_m, loss_p, rms


def valid_recon(config: Dict[str, Any],
                lens: do.Lensgroup,
                net: torch.nn.Module,
                fn_list: list,
                proj_list: list,
                cheif_pos_list: list,
                h_list: list,
                ep_id: int,
                loss_fn_alex: lpips.lpips.LPIPS,
                plot: bool,) -> None:
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if config["dtype"] == "float64":
        datatype = torch.float64
    elif config["dtype"] == "float32":
        datatype = torch.float32
    dim = config["dim"]
    wv_sample = torch.tensor(config["wv_sample"], dtype=datatype, device=device)
    loss_fn = nn.MSELoss()
    meas = torch.zeros(
        dim + 1,
        dim + 1,
        3,
        config["batch_size"],
        dtype=datatype,
        device=device)
    full = torch.zeros_like(meas)[:dim, :dim, :, :]
    gt = torch.zeros_like(meas)

    for i, wv in enumerate(wv_sample):
        torch.cuda.synchronize()
        tic = time.perf_counter()
        meas_i, full_i, gt_i = psf2meas(
            config, proj_list[i], cheif_pos_list[i], h_list[i], wv, i, fn_list)
        torch.cuda.synchronize()
        toc = time.perf_counter()
        print("time elapsed in psf2meas: ", toc - tic)
        meas += meas_i  # * meas_scale
        gt += gt_i
        full += full_i
    batch_img = meas.permute(3, 2, 0, 1).to(torch.float)
    batch_full = full.permute(3, 2, 0, 1).to(torch.float)
    batch_gt = gt.permute(3, 2, 0, 1).to(torch.float)
    pred = net(batch_img)
    loss_m = config["mse_weight"] * loss_fn(pred, batch_gt)
    loss_p = config["perc_weight"] * torch.sum(loss_fn_alex(pred, batch_gt))
    print("valid loss", loss_m.item(), loss_p.item())
    loss = loss_m + loss_p
    if plot:
        try:
            plot_e2e(config=config, meas=batch_img[0, ...], meas_full=batch_full[0, ...], gt=batch_gt[0, ...],
                    pred=pred[0, ...], lens=lens, R=config["sample_rad"], wavelength=wv, ep=ep_id)
        except:
            print("Failed to plot")
            save_lens(lens, config["record_folder"] + "/lens/lens_failed.pkl")
            mm += 1
    return loss_m, loss_p
