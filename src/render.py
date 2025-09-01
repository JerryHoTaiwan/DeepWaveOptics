from __future__ import annotations

from typing import Any, Dict, Tuple, List, Optional
import sys
import math

import torch
import matplotlib.pyplot as plt

from tracer import (
    send_chief_ray,
    trace_all,
    sample_ray_uniform_rot,
    place_img_ongrid,
    place_img_on_grid_exact,
)
from windowconv import multi_memory_efficient_ATA
from utils import (
    compute_RMS,
    create_sensor_grids,
    get_borders,
    generate_bayer,      # assuming this exists in your utils
    get_device,
    resolve_dtype,
)
from plotter import plot_zoomin_img, plot_ray_zoomin_img_sft

# Keep compatibility with your project layout
sys.path.append("..")
import deeplens.optics as deepoptics  # noqa: E402


# =========================
# PSF rendering
# =========================

def render_psf(
    config: Dict[str, Any],
    lens: Any,
    wavelength: float,
    plot: bool = False,
    use_wave: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Render PSFs at a set of reference pixels across the sensor, using either
    the wave optics pipeline (`use_wave=True`) or a ray-based approximation.

    Returns:
        proj:        (H*W, 3) back-projected chief-ray intersections in scene plane
        chief_pos:   (2, H, W) chief-ray landing positions on the sensor
        h_stack:     (ks*ks, Nref) stacked PSF patches (flattened) at each reference pixel
        rms:         scalar (sum of RMS spot sizes across all samples)
    """
    device = get_device()
    dtype = resolve_dtype(config.get("dtype"))
    dim: int = int(config["dim"])

    # Ray sampling for chief-ray mapping
    layer = 2001  # grid size for chief-ray sampling
    psf_idx = 0

    # Pixel geometry
    width = float(config["width"])
    psf_rad = float(config["psf_rad"])
    pixel_size = (2.0 * width / dim) * (1.0 + 1e-7)  # small epsilon to avoid precision issues
    half_pxl_size = pixel_size / 2.0
    psf_rad_grid = int(math.floor(psf_rad / pixel_size))
    print("pixel size:", pixel_size)

    # Chief-ray mapping (back-projection, chief positions/directions/origins)
    with torch.no_grad():
        proj, chief_pos, dir_pick, ori_pick, land_pos = send_chief_ray(
            config, lens, wavelength=wavelength, dim=dim, layer=layer, plot=plot
        )
    # For zoom-in plots (does not affect optimization)
    land_pos = chief_pos[None, ...]

    # Reference pixel selection (grid stride = interval)
    all_idx = torch.arange(dim * dim, device=device).view(dim, dim)
    interval = int(config["interval"])
    ref_idx = all_idx[0::interval, 0::interval].reshape(-1)
    # zoom-in selection: only a subset of ref_idx
    record_idx = {ref_idx[i].item() for i in config["zi_idx"]}

    # Coordinate transforms to match your plotting/indexing convention
    # chief_pos: (2,H,W) -> rotate + flip
    chief_pos_ro = torch.flip(torch.rot90(chief_pos, 1, dims=(1, 2)), dims=(1, 2))
    # directions/origins: (H*W,3) -> rotate accordingly
    dir_ro = torch.rot90(dir_pick.view(dim, dim, 3), 3, dims=(0, 1)).reshape(dim * dim, 3)
    ori_ro = torch.rot90(ori_pick.view(dim, dim, 3), 3, dims=(0, 1)).reshape(dim * dim, 3)

    # Padded sensor for PSF placement
    dim_new = int(dim + 2 * psf_rad_grid)
    max_neg = -(dim_new // 2)

    # Sensor borders (in mm), enlarged by psf_rad on each side
    dx_left, dx_right, dy_top, dy_bot = get_borders(config)
    left_border = dx_left + half_pxl_size - psf_rad
    right_border = dx_right - half_pxl_size + psf_rad
    top_border = dy_top - half_pxl_size + psf_rad
    bot_border = dy_bot + half_pxl_size - psf_rad
    sensor_pos = create_sensor_grids(config, left_border, right_border, top_border, bot_border, dim_new)

    # Accumulators
    rms = torch.tensor(0.0, dtype=dtype, device=device)
    h_stack = torch.zeros(dim_new * dim_new, ref_idx.numel(), dtype=dtype, device=device)

    # Iterate only over reference pixels
    cnt = 0
    for i in ref_idx.tolist():
        # Direction/origin at pixel i
        dvec = dir_ro[i, :]
        ovec = ori_ro[i, :]

        # Convert to view/rot angles (in degrees)
        rad = torch.sqrt(dvec[0] * dvec[0] + dvec[1] * dvec[1] + 1e-11)
        view = torch.rad2deg(torch.atan2(rad, dvec[2]))
        rot = torch.rad2deg(torch.atan2(dvec[1] + 1e-11, dvec[0]))

        offset_y = ovec[0]
        offset_x = ovec[1]

        # Find PSF crop bounds around chief position
        y_idx = i // dim
        x_idx = i % dim
        psf_cent = chief_pos_ro[:, y_idx, x_idx]  # (2,)
        y_cent_grid = torch.sign(psf_cent[0]) * torch.floor(torch.abs(psf_cent[0]) / pixel_size)
        x_cent_grid = torch.sign(psf_cent[1]) * torch.floor(torch.abs(psf_cent[1]) / pixel_size)

        # Convert to indices in padded grid, clamp bounds
        x0 = int(max(0, x_cent_grid.item() - max_neg - psf_rad_grid))
        x1 = int(min(dim_new, x_cent_grid.item() - max_neg + psf_rad_grid + 1))
        y0 = int(dim_new - min(dim_new, y_cent_grid.item() - max_neg + psf_rad_grid) - 1)
        y1 = int(dim_new - max(0, y_cent_grid.item() - max_neg - psf_rad_grid))

        if config["physic"] == "wave" and use_wave:
            # Slice the padded sensor patch for this PSF
            psf_pos = sensor_pos[:, :, y0:y1, x0:x1]
            psf_w, _psf_w_comp, ps, _oss = trace_all(
                lens,
                config["sample_rad"],
                wavelength,
                config,
                psf_idx=psf_idx,
                view=view,
                rot=rot,
                offset_y=offset_y,
                offset_x=offset_x,
                adj_pxl=True,
                land_pos=psf_pos,
            )
            psf = psf_w / (torch.sum(psf_w) + 1e-12)
            rms = rms + compute_RMS(ps)
            irrad = torch.zeros((dim_new, dim_new), dtype=dtype, device=device)
            irrad[y0:y1, x0:x1] = psf

            # Optional zoom-in visualizations
            if i in record_idx and config.get("plot_zoomin", False):
                print("zoom in!", psf_idx, float(offset_y), float(offset_x), float(view), float(rot))
                zi_ratio = float(config["zi_ratio"][str(psf_idx)])
                psf_name = f"{config['display_folder']}psf_zi_{config['physic']}_{int(wavelength)}_{psf_idx}_{config['layers']}.pth"
                irrad_off, irrad_off_comp, _coord = plot_zoomin_img(
                    config, lens, land_pos, wavelength, i, psf_idx, zi_ratio, view, rot, offset_y, offset_x
                )
                _ = irrad_off / (torch.sum(irrad_off) + 1e-12)
                torch.save(irrad_off_comp, psf_name.replace(".pth", "_comp.pth"))

        elif config["physic"] == "ray" or (config["physic"] == "wave" and not use_wave):
            # Geometric PSF (ray-based)
            lens.pixel_size = (2 * width + 2 * psf_rad - 2 * (width / dim)) / (dim_new - 1)
            lens.film_size = [dim_new, dim_new]

            if config.get("use_deeplens", False):
                pointc_ref = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
                ray_g = sample_ray_uniform_rot(
                    wavelength,
                    M=int(config["layers"]),
                    R=float(config["sample_rad"]),
                    view=float(view),
                    rot=float(rot),
                    oy=float(offset_y),
                    ox=float(offset_x),
                    datatype=dtype,
                    device=device,
                )
                ray_g_traced = lens.trace2sensor(ray_g)
                ps = ray_g_traced.o[:, :2]

                # Pack as expected by deepoptics monte-carlo
                ray_g_traced.o = ray_g_traced.o.reshape(-1, 1, 3)
                ray_g_traced.d = ray_g_traced.d.reshape(-1, 1, 3)
                ray_g_traced.ra = ray_g_traced.ra.reshape(-1, 1)

                psf = deepoptics.monte_carlo.forward_integral(
                    ray=ray_g_traced, ps=lens.pixel_size, ks=dim_new, pointc_ref=pointc_ref
                )[0, ...]
                psf = psf / (torch.sum(psf) + 1e-12)
                psf = torch.rot90(psf, 1, dims=(0, 1))
            else:
                ray_g = lens.sample_ray_uniform_rot(
                    wavelength,
                    M=int(config["layers"]),
                    R=float(config["sample_rad"]),
                    view=float(view),
                    rot=float(rot),
                    oy=float(offset_y),
                    ox=float(offset_x),
                )
                psf, ps = lens.render(ray_g)

            irrad = torch.flip(psf, dims=[0])
            rms = rms + compute_RMS(ps[:, :2])

            if config.get("plot_zoomin", False) and i in record_idx:
                print("zoom in geo!", psf_idx, float(view), float(rot))
                print("RMS:", float(compute_RMS(ps[:, :2])))
                irrad_off = plot_ray_zoomin_img_sft(
                    config, lens, land_pos, wavelength, i, view, rot, offset_y, offset_x, psf_idx
                )
                _ = irrad_off / (torch.sum(irrad_off) + 1e-12)
                psf_name = f"{config['display_folder']}psf_zi_ray_{int(wavelength)}_{psf_idx}.pth"
                plt.imshow(psf.detach().cpu().numpy(), cmap="gray")
                plt.colorbar()
                plt.savefig(psf_name.replace(".pth", ".jpg"), bbox_inches="tight")
                plt.close()
        else:
            raise ValueError("Unknown physics in config['physic'] (expected 'wave' or 'ray').")

        # Record into stack
        h_stack[:, cnt] = irrad.reshape(dim_new * dim_new)
        cnt += 1
        psf_idx += 1

    return proj, chief_pos, h_stack, rms


# =========================
# PSF -> measurement synthesis
# =========================

def psf2meas(
    config: Dict[str, Any],
    proj: torch.Tensor,
    chief_pos: torch.Tensor,
    h_stack: torch.Tensor,
    wavelength: Optional[float] = None,
    channel_idx: int = 0,
    filename: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Synthesize measurements from PSFs and a projected scene.

    Args:
        proj:        (H*W, 3) back-projected scene coords (from send_chief_ray)
        chief_pos:   (2, H, W) chief-ray sensor coords
        h_stack:     (ks*ks, Nref) stacked PSF patches from render_psf
        wavelength:  optional nm, for Bayer sensitivity
        channel_idx: channel for source images
        filename:    list of image paths or a single path

    Returns:
        mosaic_img_pad: (H+1, W+1, 3, K) mosaiced (Bayer) measurements (padded)
        full_img_pad:   (H, W, 3, K)      full RGB estimates (flipped to match orientation)
        gt_img_full:    (H+1, W+1, 3, K)  ground-truth resized images (padded, single channel at channel_idx)
    """
    device = get_device()
    dtype = resolve_dtype(config.get("dtype"))

    dim = int(config["dim"])
    width = float(config["width"])
    psf_rad = float(config["psf_rad"])
    pitch = int(config["pitch"])
    interval = int(config["interval"])
    fn_list = filename if isinstance(filename, list) else [filename] if filename is not None else ["none.png"]

    # Padding size (match how PSFs were generated)
    if config.get("pad_sensor", False):
        pxl_size = (2.0 * width / dim) * (1.0 + 1e-7)
        psf_rad_grid = int(math.floor(psf_rad / pxl_size))
        dim_new = int(dim + 2 * psf_rad_grid)
    else:
        dim_new = dim

    with torch.no_grad():
        if config.get("use_proj", True):
            interp_img = place_img_ongrid(
                proj,
                obj_max=float(config.get("obj_max", 0.0)),
                act_max=float(config.get("act_max", 0.0)),
                res=dim,
                channel_idx=channel_idx,
                filename_list=fn_list,
                cfg_dtype=config.get("dtype", "float32"),
            )
        else:
            interp_img = place_img_on_grid_exact(dim, channel_idx, fn_list, cfg_dtype=config.get("dtype", "float32"))

        # Rotate/flip to match PSF orientation; flatten to (H*W, K)
        img_num = interp_img.size(-1)
        interp_img_ro = torch.flip(
            torch.rot90(interp_img.view(dim, dim, img_num), 1, dims=(0, 1)),
            dims=(0, 1),
        ).reshape(dim * dim, img_num)

    # Chief positions in the same orientation used for PSF placement
    chief_pos_ro = torch.flip(torch.rot90(chief_pos, 1, dims=(1, 2)), dims=(1, 2))
    ref_idx = torch.arange(dim * dim, device=device).view(dim, dim)[0::interval, 0::interval].reshape(-1)
    chief_ref = chief_pos_ro[:, 0::interval, 0::interval].reshape(2, -1)

    # Convolutional accumulation using memory-efficient operator
    meas = multi_memory_efficient_ATA(
        config=config,
        h_ref_load=h_stack,
        brightness=interp_img_ro,
        chief_all=chief_pos_ro,
        chief_ref=chief_ref,
        ref_idx=ref_idx,
    )  # expected shape: (dim_new, dim_new, K)

    # Apply color filter (Bayer) and crop to original size
    est_meas_f = torch.zeros((3, dim_new, dim_new, img_num), dtype=dtype, device=device)
    sensitivity = generate_bayer(0, dim_new, dim_new, wavelength)
    est_meas_f = sensitivity[..., None] * meas[None, ...]  # (3,H,W,K)

    diff_size = (dim_new - dim) // 2
    if diff_size > 0:
        est_meas_crop = est_meas_f[:, diff_size:-diff_size, diff_size:-diff_size]
    else:
        est_meas_crop = est_meas_f

    # Assemble mosaiced image (Bayer pattern) with 1-pixel padding
    mosaic_img_pad = torch.zeros((dim + 1, dim + 1, 3, img_num), dtype=dtype, device=device)
    mosaic_img_pad[1:-1:pitch, 1:-1:pitch, 0, :] += est_meas_crop[2, 1::pitch, 1::pitch]  # R at (odd,odd)
    mosaic_img_pad[1:-1:pitch, 0:-1:pitch, 1, :] += est_meas_crop[1, 1::pitch, 0::pitch]  # G at (odd,even)
    mosaic_img_pad[0:-1:pitch, 1:-1:pitch, 1, :] += est_meas_crop[1, 0::pitch, 1::pitch]  # G at (even,odd)
    mosaic_img_pad[0:-1:pitch, 0:-1:pitch, 2, :] += est_meas_crop[0, 0::pitch, 0::pitch]  # B at (even,even)

    # Full RGB (flipped to match earlier convention)
    full_img_pad = torch.flip(est_meas_crop[:3, ...].permute(1, 2, 0, 3), dims=[2])  # (H,W,3,K)

    # Ground-truth (resized) images in same padded canvas; keep on SAME device
    gt_img_full = torch.zeros((dim + 1, dim + 1, 3, img_num), dtype=dtype, device=device)
    gt_img_full[:dim, :dim, channel_idx, :] = interp_img_ro.view(dim, dim, img_num)

    return mosaic_img_pad, full_img_pad, gt_img_full