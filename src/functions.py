from typing import Dict, Any, List
from pathlib import Path
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

import torch
import torch.optim as optim
import lpips

from render import render_psf, psf2meas
from unet_model import UNet
from utils import get_data_list, save_lens, get_device, resolve_dtype
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
    
    
def display(config: Dict[str, Any], lens: Any) -> None:
    """
    Render PSFs and synthesize measurements for a set of wavelengths,
    then save summed stacks (meas/full/gt) to disk.

    Notes:
        - Uses the first image in the batch ([..., 0]) like your original code.
        - Sums across wavelengths; you can divide by len(wavelengths) to average if desired.
        - Saves PNGs with values clipped to [0, 255].
    """
    dim: int = int(config["dim"])
    out_dir = Path(config["display_folder"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Wavelengths: use config override if provided
    wavelengths: List[int] = config.get("render_wavelengths", [440, 510, 650])

    # Input file list (pass-through to psf2meas)
    fn_list = config.get("filelist", ["none.png"])

    # Allocate accumulators (float32)
    meas_stack = np.zeros((dim + 1, dim + 1, 3), dtype=np.float32)
    full_stack = np.zeros((dim, dim, 3), dtype=np.float32)
    gt_stack   = np.zeros((dim + 1, dim + 1, 3), dtype=np.float32)

    for ch_idx, wv in enumerate(wavelengths):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tic = time.perf_counter()

        # 1) Render PSFs
        proj, chief_pos, h_stack, rms_i = render_psf(
            config=config,
            lens=lens,
            wavelength=float(wv),
            plot=False,
            use_wave=True,
        )
        print("==== Finish rendering PSFs ====")

        # 2) Convolve to measurements
        meas, full, gt = psf2meas(
            config=config,
            proj=proj,
            chief_pos=chief_pos,
            h_stack=h_stack,
            wavelength=float(wv),
            channel_idx=ch_idx,   # keep original mapping: channel index per wavelength
            filename=fn_list,
        )
        print("==== Finish rendering measurement ====")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        toc = time.perf_counter()
        print(f"Time for {wv} nm: {toc - tic:.3f} s")

        # Accumulate (use first item in batch dimension to match original behavior)
        meas_np = meas[..., 0].detach().to("cpu", non_blocking=True).numpy()
        full_np = full[..., 0].detach().to("cpu", non_blocking=True).numpy()
        gt_np   = gt  [..., 0].detach().to("cpu", non_blocking=True).numpy()

        meas_stack += meas_np
        full_stack += full_np
        gt_stack   += gt_np

    # Optional: average instead of sum
    # denom = float(len(wavelengths))
    # meas_stack /= denom; full_stack /= denom; gt_stack /= denom

    # Save outputs
    np.save(str(out_dir / "full_stack.npy"), full_stack)

    # Clip to [0,1] before saving as PNG
    meas_png = np.clip(meas_stack, 0.0, 1.0)
    full_png = np.clip(full_stack, 0.0, 1.0)
    gt_png   = np.clip(gt_stack,   0.0, 1.0)

    # Note: cv2.imwrite expects BGR; we preserve your original behavior (no channel swap).
    cv2.imwrite(
        str(out_dir / f"rgb_meas_{config['physic']}.png"),
        (meas_png * 255.0).astype(np.uint8),
    )
    cv2.imwrite(
        str(out_dir / f"rgb_full_{config['physic']}.png"),
        (full_png * 255.0).astype(np.uint8),
    )
    cv2.imwrite(
        str(out_dir / f"rgb_gt_{config['physic']}.png"),
        (gt_png * 255.0).astype(np.uint8),
    )
    
    
def e2e_recon(config: Dict[str, Any], lens: Any) -> None:
    """
    End-to-end reconstruction & lens co-optimization loop.

    Expects in `config` (non-exhaustive):
      - training/validation: "train_num", "valid_num", "batch_size", "max_ep", "earlystop"
      - losses/weights: "mse_weight", "perc_weight", "lr"
      - wavelengths: "wv_sample" (list of nm, optional)
      - network: "ft_net" (bool), "net_name" (optional path)
      - dtype: "float32" or "float64"
      - record_folder: output directory
      - use_deeplens: bool (affects lens param handling)
    """
    device = get_device()
    dtype = resolve_dtype(config.get("dtype"))

    # Required config with sensible fallbacks / checks
    train_num: int = int(config.get("train_num", 0))
    valid_num: int = int(config.get("valid_num", 0))
    batch_size: int = max(1, int(config.get("batch_size", 4)))
    max_ep: int = int(config.get("max_ep", 1))
    earlystop: int = int(config.get("earlystop", max_ep))
    mse_weight: float = float(config.get("mse_weight", 1.0))
    perc_weight: float = float(config.get("perc_weight", 1.0))
    base_lr: float = float(config.get("lr", 1e-5))

    # Batches (ceil to cover leftovers)
    train_batches = math.ceil(max(0, train_num) / batch_size) if train_num else 0
    valid_batches = math.ceil(max(0, valid_num) / batch_size) if valid_num else 0

    # Wavelengths
    wv_list: List[float] = config.get("wv_sample", [440.0, 510.0, 650.0])
    wv_sample = torch.as_tensor(wv_list, dtype=dtype, device=device)

    # -----------------------
    # Initialize the network
    # -----------------------
    net = UNet(n_channels=3, n_classes=3, do_checkpoint=False).to(device)
    if bool(config.get("ft_net", False)):
        # Finetune from provided checkpoint
        ckpt = config.get("net_name", "")
        if not ckpt:
            raise ValueError("ft_net=True but 'net_name' is not provided in config.")
        net.load_state_dict(torch.load(ckpt, map_location=device))
    else:
        # Default: load your pretrained dictionary
        net.load_state_dict(torch.load(
            "/home/ubuntu/DiffWaveOptics/saved_networks/recon_pretrained_dict.pth",
            map_location=device
        ))
    net.train()

    # -----------------------
    # Build optimizer params
    # -----------------------
    param_list: List[Dict[str, Any]] = [{'params': net.parameters(), 'lr': 1e-4}]
    if bool(config.get("use_deeplens", False)):
        # Let the lens object expose its own parameter groups
        param_list += lens.get_optimizer_params()
    else:
        # DiffOptics: expose surface curvature 'c' where available (if not ft_net)
        if not bool(config.get("ft_net", False)):
            for i, surf in enumerate(getattr(lens, "surfaces", [])):
                if hasattr(surf, "c"):
                    c_var = surf.c.detach().to(dtype=dtype, device=device).requires_grad_()
                    surf.c = c_var
                    param_list.append({'params': [c_var], 'lr': base_lr})
                else:
                    print(f"[warn] Surface {i} has no curvature attribute 'c'")

    opt = optim.Adam(param_list, lr=1e-5)

    # Perceptual loss
    loss_fn_alex = lpips.LPIPS(net="alex").to(device)
    # lpips expects float32 inputs; keep model & loss on device (not dtype casted)

    # -----------------------
    # Data lists
    # -----------------------
    train_list_all, valid_list_all = get_data_list(config)

    # Tracking
    loss_list = torch.zeros(max_ep, dtype=torch.float32)
    dataloss_list = torch.zeros(max_ep, dtype=torch.float32)
    rmsloss_list = torch.zeros(max_ep, dtype=torch.float32)
    patience = 0
    min_loss = float("inf")
    best_ep = -1

    # =======================
    # Training / Validation
    # =======================
    for ep in range(max_ep):
        ep_tic = time.perf_counter()
        print(f"\n=== Epoch {ep}/{max_ep-1} | Patience {patience}/{earlystop} ===")

        # -----------------------
        # Train
        # -----------------------
        net.train()
        total_loss = 0.0
        total_mseloss = 0.0
        total_percloss = 0.0
        total_rmsloss = 0.0

        for batch_id in range(train_batches):
            print(f"[train batch {batch_id}/{train_batches-1}]")
            start_img_id = batch_id * batch_size
            end_img_id = min((batch_id + 1) * batch_size, train_num)
            fn_list = train_list_all[start_img_id:end_img_id]

            # (Optional) print current curvatures
            for i, surf in enumerate(getattr(lens, "surfaces", [])):
                if hasattr(surf, "c") and torch.is_tensor(surf.c):
                    print(f"surface {i} c = {float(surf.c.detach())}")

            loss, loss_m, loss_p, rms = train_recon(config, lens, net, opt, fn_list)
            total_loss += float(loss.item())
            total_mseloss += float(loss_m.item())
            total_percloss += float(loss_p.item()) * 1e3  # keep your scale display
            total_rmsloss += float(rms.item())

            torch.cuda.empty_cache()

        denom = max(1, train_batches)  # avoid divide by zero
        print(
            f"[train] loss={total_loss/denom:.6f} | perc(x1e3)={total_percloss/denom:.6f} | "
            f"mse={total_mseloss/denom:.6f} | RMS={total_rmsloss/denom:.6f}"
        )

        # -----------------------
        # Validation
        # -----------------------
        with torch.no_grad():
            net.eval()

            # Precompute PSFs for each wavelength on current lens
            proj_list: List[torch.Tensor] = []
            chief_pos_list: List[torch.Tensor] = []
            h_list: List[torch.Tensor] = []
            rms_total = 0.0

            for wv in wv_sample:
                proj, chief_pos, h_stack, rms_i = render_psf(config, lens, float(wv), plot=False)
                rms_total += float(rms_i)
                proj_list.append(proj)
                chief_pos_list.append(chief_pos)
                h_list.append(h_stack)

            v_total = 0.0
            v_mse = 0.0
            v_perc = 0.0

            for batch_id in range(valid_batches):
                start_img_id = batch_id * batch_size
                end_img_id = min((batch_id + 1) * batch_size, valid_num)
                fn_list = valid_list_all[start_img_id:end_img_id]

                ep_id = ep * 1000 + batch_id
                plot_last = (batch_id == valid_batches - 1)

                loss_m, loss_p = valid_recon(
                    config, lens, net, fn_list,
                    proj_list, chief_pos_list, h_list,
                    ep_id, loss_fn_alex, plot_last
                )

                # Mirror your original aggregation
                loss = loss_m + loss_p / float(batch_size)
                print(f"[valid batch {batch_id}/{valid_batches-1}] "
                      f"loss={float(loss.item()):.6f} | mse={float(loss_m.item()):.6f} | "
                      f"perc/batch={float(loss_p.item())/batch_size:.6f} | RMS_agg={rms_total:.6f}")

                v_total += float(loss.item())
                v_mse += float(loss_m.item())
                v_perc += float(loss_p.item())

                torch.cuda.empty_cache()

            v_denom = max(1, valid_batches)
            # match your printed metrics
            rmse_disp = 255.0 * math.sqrt(max(0.0, v_mse / (mse_weight * v_denom + 1e-8)))
            perc_disp = v_perc / (perc_weight * v_denom * batch_size + 1e-8)
            print(f"[valid] loss={v_total/v_denom:.6f} | rmse={rmse_disp:.3f} | perc={perc_disp:.6f} | RMS={rms_total:.6f}")

            # Save lens snapshot each epoch
            try:
                save_lens(lens, f"{config['record_folder']}/lens/lens_{ep}.pkl")
            except Exception as e:
                print(f"[warn] save_lens failed at epoch {ep}: {e}")

            net.train()

        # -----------------------
        # Book-keeping / early stop
        # -----------------------
        loss_list[ep] = v_total
        dataloss_list[ep] = v_mse
        rmsloss_list[ep] = rms_total

        if v_total < min_loss and ep > 0:
            min_loss = v_total
            best_ep = ep
            patience = 0

            # Save best net + lens
            try:
                torch.save(net, f"{config['record_folder']}/net_opt.pth")
                torch.save(net.state_dict(), f"{config['record_folder']}/net_opt_dict.pth")
                if bool(config.get("use_deeplens", False)):
                    lens.write_lens_json(f"{config['record_folder']}/lens_opt.json")
                else:
                    save_lens(lens, f"{config['record_folder']}/lens_opt.pkl")
            except Exception as e:
                print(f"[warn] checkpoint save failed at epoch {ep}: {e}")
        else:
            patience += 1

        print(f"[epoch {ep}] total_valid_loss={v_total:.6f} (best@{best_ep}={min_loss:.6f})")

        try:
            plot_loss_curve(config, loss_list, dataloss_list, rmsloss_list, best_ep, ep)
        except Exception as e:
            print(f"[warn] plot_loss_curve failed at epoch {ep}: {e}")

        # Optional PSF visualization (guarded)
        try:
            if h_list:
                h_dim = int(math.isqrt(h_list[-1].shape[0])) if h_list[-1].dim() == 2 else int(math.isqrt(h_list[-1].shape[0]))
                h_sum = torch.sum(h_list[-1], dim=1).view(h_dim, h_dim)
                plt.imshow(torch.abs(h_sum).detach().cpu().numpy())
                plt.colorbar()
                plt.savefig(f"{config['record_folder']}/psf/u_est_{ep}_sum.png", bbox_inches="tight")
                plt.close()
        except Exception as e:
            print(f"[warn] PSF viz failed at epoch {ep}: {e}")

        if patience >= earlystop:
            print(f"[early stop] patience {patience} reached (>= {earlystop}).")
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ep_toc = time.perf_counter()
        print(f"[epoch {ep}] time: {ep_toc - ep_tic:.2f}s")

    print(f"Best epoch: {best_ep} with loss {min_loss:.6f}")