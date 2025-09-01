from typing import Any, Dict, Tuple
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

from tracer import send_chief_ray, trace_all, sample_ray_uniform_rot, place_img_ongrid, place_img_on_grid_exact
from windowconv import multi_memory_efficient_ATA
from utils import compute_RMS, create_sensor_grids, get_borders, interpolation_symm, generate_bayer
from plotter import plot_zoomin_img, plot_ray_zoomin_img_sft
sys.path.append("../")
import deeplens.optics as deepoptics


def render_psf(config: Dict[str, Any],
               lens: Any,
               wavelength: float,
               plot: bool = False,
               use_wave: bool = True):
    if config["dtype"] == "float64":
        datatype = torch.float64
    elif config["dtype"] == "float32":
        datatype = torch.float32
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    dim = config["dim"]

    layer = 2001
    view = 0.
    rot = 0.
    psf_idx = 0

    # pixel location
    width = config["width"]
    psf_rad = config["psf_rad"]
    pixel_size = (2 * width / dim) * 1.0000001  # avoid precision issues
    half_pxl_size = pixel_size / 2
    psf_rad_grid = psf_rad // pixel_size
    print('pixel size: ', pixel_size)

    with torch.no_grad():
        proj, chief_pos, dir_pick, ori_pick, land_pos = send_chief_ray(
            config, lens, wavelength=wavelength, dim=dim, layer=layer, plot=plot)
    # only affects zoomed-in displaying, not optimization
    land_pos = chief_pos[None, ...]

    # to deal with the inconsitency of index organization
    all_idx = torch.arange(dim * dim).view(dim, dim)
    ref_idx = all_idx[0::config["interval"], 0::config["interval"]].reshape(-1)
    record_idx = [ref_idx[i] for i in config["zi_idx"]]
    chief_pos_rot = torch.rot90(chief_pos, 1, dims=(1, 2))
    chief_pos_ro = torch.flip(chief_pos_rot, dims=(1, 2))  # * 1e-3

    dir_ro = torch.rot90(
        dir_pick.reshape(
            dim, dim, 3), 3, dims=(
            0, 1)).reshape(
                dim * dim, 3)
    ori_ro = torch.rot90(
        ori_pick.reshape(
            dim, dim, 3), 3, dims=(
            0, 1)).reshape(
                dim * dim, 3)

    dim_new = int(dim + 2 * psf_rad_grid)
    max_neg = -(dim_new // 2)
    dx_left, dx_right, dy_top, dy_bot = get_borders(config)
    left_border = dx_left + half_pxl_size - psf_rad
    right_border = dx_right - half_pxl_size + psf_rad
    top_border = dy_top - half_pxl_size + psf_rad
    bot_border = dy_bot + half_pxl_size - psf_rad
    sensor_pos = create_sensor_grids(
        config, left_border, right_border, top_border, bot_border, dim_new)
    cnt = 0
    rms = 0
    h_stack = torch.zeros(
        dim_new * dim_new,
        len(ref_idx),
        dtype=datatype,
        device=device)

    for i in range(dim * dim):
        if i in ref_idx:
            rad = torch.sqrt(dir_ro[i, 0] *
                             dir_ro[i, 0] +
                             dir_ro[i, 1] *
                             dir_ro[i, 1] +
                             1e-11)
            rot = torch.rad2deg(torch.atan2(
                dir_ro[i, 1] + 1e-11, dir_ro[i, 0]))
            view = torch.rad2deg(torch.atan2(rad, dir_ro[i, 2]))
            offset_y = ori_ro[i, 0]
            offset_x = ori_ro[i, 1]

            y_idx = i // dim
            x_idx = i % dim
            half_pxl_size = width / dim
            psf_cent = chief_pos_ro[:, y_idx, x_idx]
            y_cent_grid = torch.sign(
                psf_cent[0]) * (torch.abs(psf_cent[0]) // pixel_size)
            x_cent_grid = torch.sign(
                psf_cent[1]) * (torch.abs(psf_cent[1]) // pixel_size)
            psf_left_bound = int(
                max(0, x_cent_grid - max_neg - psf_rad_grid))
            psf_right_bound = int(
                min(dim_new, x_cent_grid - max_neg + psf_rad_grid + 1))
            psf_up_bound = int(
                dim_new - min(dim_new, y_cent_grid - max_neg + psf_rad_grid) - 1)
            psf_bot_bound = int(
                dim_new - max(0, y_cent_grid - max_neg - psf_rad_grid))

            if config["physic"] == 'wave' and use_wave:

                psf_pos = sensor_pos[:,
                                     :,
                                     psf_up_bound:psf_bot_bound,
                                     psf_left_bound:psf_right_bound]
                psf_w, _psf_w_comp, ps, _oss = trace_all(lens, config["sample_rad"], wavelength, config, psf_idx=psf_idx,
                                            view=view, rot=rot, offset_y=offset_y, offset_x=offset_x, adj_pxl=True, land_pos=psf_pos)
                psf = torch.clone(psf_w) / torch.sum(psf_w)
                rms += compute_RMS(ps)
                irrad = torch.zeros(
                    dim_new, dim_new, dtype=datatype, device=device)
                irrad[psf_up_bound:psf_bot_bound,
                      psf_left_bound:psf_right_bound] = psf

                # record values if needed
                if i in record_idx and config["plot_zoomin"]:
                    print("zoom in!", psf_idx, offset_y.item(), offset_x.item(), view.item(), rot.item())
                    zi_ratio = config["zi_ratio"][str(psf_idx)]
                    psf_name = config["display_folder"] + \
                        "psf_zi_{}_{}_{}_{}.pth".format(config['physic'], int(wavelength), psf_idx, config["layers"])
                    irrad_off, irrad_off_comp, _coord = plot_zoomin_img(
                        config, lens, land_pos, wavelength, i, psf_idx, zi_ratio, view, rot, offset_y, offset_x)
                    irrad_off_norm = irrad_off / torch.sum(irrad_off)
                    torch.save(irrad_off_comp, psf_name.replace(".pth", "_comp.pth"))

            elif config["physic"] == 'ray' or (config["physic"] == 'wave' and not use_wave):
                lens.pixel_size = (
                    2 * width + 2 * psf_rad - 2 * half_pxl_size) / (dim_new - 1)
                lens.film_size = [dim_new, dim_new]
                if config["use_deeplens"]:
                    pointc_ref = torch.tensor([0., 0.]).to(device)
                    ray_g = sample_ray_uniform_rot(
                        wavelength,
                        M=config["layers"],
                        R=config["sample_rad"],
                        view=view,
                        rot=rot,
                        oy=offset_y,
                        ox=offset_x,
                        datatype=datatype,
                        device=device)
                    ray_g_traced = lens.trace2sensor(ray_g)
                    ps = ray_g_traced.o[:, :2]
                    ray_g_traced.o = ray_g_traced.o.reshape(-1, 1, 3)
                    ray_g_traced.d = ray_g_traced.d.reshape(-1, 1, 3)
                    ray_g_traced.ra = ray_g_traced.ra.reshape(-1, 1)
                    psf = deepoptics.monte_carlo.forward_integral(ray=ray_g_traced, ps=lens.pixel_size, ks=dim_new, pointc_ref=pointc_ref)
                    psf = psf[0, ...] / torch.sum(psf)
                    psf = torch.rot90(psf, 1, dims=(0, 1))
                else:
                    ray_g = lens.sample_ray_uniform_rot(
                        wavelength,
                        M=config["layers"],
                        R=config["sample_rad"],
                        view=view,
                        rot=rot,
                        oy=offset_y,
                        ox=offset_x)
                    psf, ps = lens.render(ray_g)
                irrad = torch.flip(psf, dims=[0])
                rms += compute_RMS(ps[:, :2])
                if config["plot_zoomin"] and i in record_idx:
                    print('zoom in geo!', psf_idx, view.item(), rot)
                    print("RMS: ", compute_RMS(ps[:, :2]).item())
                    irrad_off = plot_ray_zoomin_img_sft(
                        config, lens, land_pos, wavelength, i, view, rot, offset_y, offset_x, psf_idx)
                    irrad_off_norm = irrad_off / torch.sum(irrad_off)
                    psf_name = config["display_folder"] + \
                            "psf_zi_ray_{}_{}.pth".format(int(wavelength), psf_idx)
                    import matplotlib.pyplot as plt
                    plt.imshow(psf.detach().cpu().numpy(), cmap='gray')
                    plt.colorbar()
                    plt.savefig(psf_name.replace(".pth", ".jpg"))
                    plt.close()
            else:
                print('ERROR: unknown physics defined')
            irrad_flat = irrad.reshape(dim_new * dim_new)
            h_stack[:, cnt] = irrad_flat
            cnt += 1
            psf_idx += 1
    return proj, chief_pos, h_stack, rms


def psf2meas(config: Dict[str,
                          Any],
             proj,
             chief_pos,
             h_stack,
             wavelength=None,
             channel_idx=0,
             filename='none.png'):
    datatype = torch.float64 if config["dtype"] == "float64" else torch.float32
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    dim = config["dim"]
    width = config["width"]
    psf_rad = config["psf_rad"]
    pitch = config["pitch"]
    interval = config["interval"]
    fn_list = filename

    if config["pad_sensor"]:
        pxl_size = (2 * width / dim) * 1.0000001  # avoid precision issues
        psf_rad_grid = psf_rad // pxl_size
        dim_new = int(dim + 2 * psf_rad_grid)

    with torch.no_grad():
        if config["use_proj"]:
            interp_img = place_img_ongrid(
                proj,
                obj_max=config["obj_max"],
                act_max=config["act_max"],
                res=dim,
                channel_idx=channel_idx,
                filename_list=fn_list,
            )
        else:
            interp_img = place_img_on_grid_exact(dim, channel_idx, fn_list)
        img_num = interp_img.size()[-1]
        interp_img_ro = torch.flip(
            torch.rot90(
                interp_img.view(
                    dim, dim, img_num), 1, dims=(
                    0, 1)), dims=(
                0, 1)).reshape(
            dim * dim, img_num)

    # to deal with the inconsitency of index organization
    all_idx = torch.arange(dim * dim).view(dim, dim)
    chief_pos_rot = torch.rot90(chief_pos, 1, dims=(1, 2))
    chief_pos_ro = torch.flip(chief_pos_rot, dims=(1, 2))  # * 1e-3

    ref_idx = all_idx[0::interval, 0::interval].reshape(-1)
    chief_ref = chief_pos_ro[:, 0::interval, 0::interval].reshape(2, -1)

    meas = multi_memory_efficient_ATA(
        config=config,
        h_ref_load=h_stack,
        brightness=interp_img_ro,
        chief_all=chief_pos_ro,
        chief_ref=chief_ref,
        ref_idx=ref_idx)

    est_meas_f = torch.zeros(
        3,
        dim_new,
        dim_new,
        img_num,
        dtype=datatype,
        device=device)

    sensitivity = generate_bayer(0, dim_new, dim_new, wavelength)
    est_meas_f = sensitivity[..., None] * meas[None, ...]
    diff_size = (est_meas_f.size()[1] - dim) // 2
    est_meas_crop = est_meas_f[:, diff_size:-diff_size, diff_size:-diff_size] # TODO: handle the case when diff_size=0

    mosaic_img_pad = torch.zeros(
        dim + 1,
        dim + 1,
        3,
        len(fn_list),
        dtype=datatype,
        device=device)
    mosaic_img_pad[1:-1:pitch, 1:-1:pitch, 0,
                   :] += torch.clone(est_meas_crop[2, 1::pitch, 1::pitch])
    mosaic_img_pad[1:-1:pitch, 0:-1:pitch, 1,
                   :] += torch.clone(est_meas_crop[1, 1::pitch, 0::pitch])
    mosaic_img_pad[0:-1:pitch, 1:-1:pitch, 1,
                   :] += torch.clone(est_meas_crop[1, 0::pitch, 1::pitch])
    mosaic_img_pad[0:-1:pitch, 0:-1:pitch, 2,
                   :] += torch.clone(est_meas_crop[0, 0::pitch, 0::pitch])

    full_img_pad = torch.flip(
        (est_meas_crop[:3, ...]).permute(1, 2, 0, 3), dims=[2])

    gt_img_full = torch.zeros(
        dim + 1,
        dim + 1,
        3,
        len(fn_list),
        dtype=datatype,
        device=device)
    gt_img_full[:dim, :dim, channel_idx, :] = interp_img_ro.view(
        dim, dim, len(fn_list)).detach().cpu()

    return mosaic_img_pad, full_img_pad, gt_img_full