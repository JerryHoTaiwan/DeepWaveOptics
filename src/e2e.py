from __future__ import annotations

from typing import Any, Dict, List, Tuple
import time

import torch
import torch.nn as nn
import lpips

import diffoptics as do
from render import render_psf, psf2meas
from plotter import plot_e2e
from utils import save_lens, get_device, resolve_dtype


def train_recon(
    config: Dict[str, Any],
    lens: do.Lensgroup,
    net: torch.nn.Module,
    opt: torch.optim.Optimizer,
    fn_list: List[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One training step:
      - render PSFs for each wavelength
      - synthesize measurements
      - forward UNet and compute losses (MSE + LPIPS)
      - backprop and optimizer step

    Returns:
      (total_loss, mse_loss, perc_loss, rms_sum)
    """
    device = get_device()
    dtype = resolve_dtype(config.get("dtype"))

    dim: int = int(config["dim"])
    batch_size: int = int(config["batch_size"])
    use_wave: bool = (config["physic"] == "wave")

    # Losses
    loss_fn = nn.MSELoss()
    # LPIPS expects float32 images scaled to [-1, 1] typically; your pipeline uses [0,1].
    # We keep your original usage for consistency.
    loss_fn_alex = lpips.LPIPS(net="alex").to(device)

    # Wavelengths
    wv_sample = torch.as_tensor(config["wv_sample"], dtype=dtype, device=device)

    # Allocate measurement/GT stacks
    meas = torch.zeros(dim + 1, dim + 1, 3, batch_size, dtype=dtype, device=device)
    gt   = torch.zeros_like(meas)

    rms_sum = torch.zeros((), dtype=dtype, device=device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    # Render + synthesize for each wavelength
    for ch_idx, wv in enumerate(wv_sample):
        proj, chief_pos, h_stack, rms_i = render_psf(
            config, lens, float(wv), plot=False, use_wave=use_wave
        )
        meas_i, _full_i, gt_i = psf2meas(
            config, proj, chief_pos, h_stack, float(wv), ch_idx, fn_list
        )
        meas += meas_i
        gt   += gt_i
        rms_sum += rms_i

    # UNet expects (N,C,H,W); current tensors are (H,W,C,N)
    batch_img = meas.permute(3, 2, 0, 1).contiguous().to(torch.float32)
    batch_gt  = gt  .permute(3, 2, 0, 1).contiguous().to(torch.float32)

    pred = net(batch_img)

    loss_m = float(config["mse_weight"])  * loss_fn(pred, batch_gt)
    loss_p = float(config["perc_weight"]) * torch.sum(loss_fn_alex(pred, batch_gt))

    # Optional RMS penalty / switching logic
    if rms_sum.item() > float(config.get("rms_thres", float("inf"))):
        print(f"WARNING: large RMS detected: {float(rms_sum.item()):.6f}")
        loss = loss_m + loss_p + float(config.get("rms_weight", 0.0)) * rms_sum
        use_wave = False
    else:
        loss = loss_m + loss_p
        use_wave = True

    print(
        "batch loss:",
        float(loss.item()),
        float(loss_m.item()),
        float(loss_p.item()),
        float(rms_sum.item()),
        rms_sum.item() > float(config.get("rms_thres", float("inf"))),
    )

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    print(f"time elapsed in batch computation: {end_time - start_time:.3f}s")

    torch.cuda.empty_cache()
    return loss.detach(), loss_m.detach(), loss_p.detach(), rms_sum.detach()


def valid_recon(
    config: Dict[str, Any],
    lens: do.Lensgroup,
    net: torch.nn.Module,
    fn_list: List[str],
    proj_list: List[torch.Tensor],
    chief_pos_list: List[torch.Tensor],
    h_list: List[torch.Tensor],
    ep_id: int,
    loss_fn_alex: lpips.LPIPS,
    plot: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Validation step using precomputed PSFs (proj/chief_pos/h_list) for each wavelength.

    Returns:
      (mse_loss, perc_loss)
    """
    device = get_device()
    dtype = resolve_dtype(config.get("dtype"))

    dim: int = int(config["dim"])
    batch_size: int = int(config["batch_size"])

    loss_fn = nn.MSELoss()
    wv_sample = torch.as_tensor(config["wv_sample"], dtype=dtype, device=device)

    meas = torch.zeros(dim + 1, dim + 1, 3, batch_size, dtype=dtype, device=device)
    full = torch.zeros(dim, dim, 3, batch_size, dtype=dtype, device=device)
    gt   = torch.zeros_like(meas)

    # Use PSFs already computed for each wavelength
    for ch_idx, wv in enumerate(wv_sample):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tic = time.perf_counter()

        meas_i, full_i, gt_i = psf2meas(
            config, proj_list[ch_idx], chief_pos_list[ch_idx], h_list[ch_idx], float(wv), ch_idx, fn_list
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        toc = time.perf_counter()
        print(f"time elapsed in psf2meas@{float(wv):.1f}nm: {toc - tic:.3f}s")

        meas += meas_i
        gt   += gt_i
        full += full_i

    batch_img  = meas.permute(3, 2, 0, 1).contiguous().to(torch.float32)
    batch_full = full.permute(3, 2, 0, 1).contiguous().to(torch.float32)
    batch_gt   = gt  .permute(3, 2, 0, 1).contiguous().to(torch.float32)

    pred = net(batch_img)

    loss_m = float(config["mse_weight"])  * loss_fn(pred, batch_gt)
    loss_p = float(config["perc_weight"]) * torch.sum(loss_fn_alex(pred, batch_gt))

    print("valid loss:", float(loss_m.item()), float(loss_p.item()))
    if plot:
        # Use the last wavelength for tagging; safe even if list has length 1.
        last_wv = float(wv_sample[-1].item())
        try:
            plot_e2e(
                config=config,
                meas=batch_img[0, ...],
                meas_full=batch_full[0, ...],
                gt=batch_gt[0, ...],
                pred=pred[0, ...],
                lens=lens,
                R=float(config["sample_rad"]),
                wavelength=last_wv,
                ep=ep_id,
            )
        except Exception as e:
            print(f"Failed to plot: {e}")
            try:
                save_lens(lens, f"{config['record_folder']}/lens/lens_failed.pkl")
            except Exception as e2:
                print(f"Failed to save lens on plot error: {e2}")

    return loss_m.detach(), loss_p.detach()