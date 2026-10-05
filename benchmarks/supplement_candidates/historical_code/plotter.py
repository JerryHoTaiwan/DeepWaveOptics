import cv2
from typing import Any, Dict
import numpy as np
import matplotlib.pyplot as plt
import torch
from tracer import trace_all
from typing import List, Tuple
import sys
from sensor import create_sensor_grids
from utils import get_borders
sys.path.append("../")
import diffoptics as do


def plot_zoomin_img(config: Dict[str, Any],
                    lens: do.Lensgroup,
                    land_pos: torch.Tensor,
                    wavelength: float,
                    i: int,
                    psf_idx: int,
                    zi_ratio: float,
                    view: float,
                    rot: float,
                    offset_y: float,
                    offset_x: float,
                    direct_cent: bool = False,) -> Tuple[torch.Tensor,
                                                         List[float]]:
    width = config["width"]
    dim = config["dim"]
    zi_dim = config["zoomin_dim"]
    if direct_cent:
        cent = land_pos.detach().cpu().numpy()
    else:
        cent = land_pos[0, :, dim - 1 -
                        int(i % dim), int(i // dim)].detach().cpu().numpy()
    dx_left_off = cent[1] - zi_ratio * width
    dx_right_off = cent[1] + zi_ratio * width
    dx_top_off = cent[0] + zi_ratio * width
    dx_bot_off = cent[0] - zi_ratio * width
    offaxis_pos = create_sensor_grids(config, dx_left_off, dx_right_off,
                                      dx_top_off, dx_bot_off, zi_dim)
    irrad_off, _p, _o = trace_all(lens, config["sample_rad"], wavelength, config, view=view,
                                  rot=rot, offset_y=offset_y, offset_x=offset_x, adj_pxl=True, land_pos=offaxis_pos)
    assert not torch.isnan(torch.sum(irrad_off))
    assert torch.sum(irrad_off) > 0
    fig_name = config["display_folder"] + "psf_zoomin_wave_{}_{}_{}.png".format(
        int(wavelength), psf_idx, config["layers"])
    print('width: ', dx_right_off - dx_left_off, 'pixel size: ',
          (dx_right_off - dx_left_off) / zi_dim, 'dim: ', zi_dim)
    plt.imshow(
        (irrad_off / torch.sum(irrad_off)).detach().cpu().numpy(),
        extent=(
            dx_left_off,
            dx_right_off,
            dx_bot_off,
            dx_top_off))
    plt.xlabel('x [mm]')
    plt.ylabel('y [mm]')
    plt.colorbar()
    plt.savefig(
        fig_name)
    plt.close()
    return irrad_off, [dx_left_off, dx_right_off, dx_top_off, dx_bot_off]


def plot_ray_zoomin_img_sft(
        config: Dict[str, Any],
        lens: do.Lensgroup,
        land_pos: torch.Tensor,
        wavelength: float,
        i: int,
        view: float,
        rot: float,
        offset_y: float,
        offset_x: float,
        psf_idx: int = 0,
        direct_cent: bool = False) -> torch.Tensor:
    dim = config["dim"]
    width = config["width"]
    zi_ratio = config["zi_ratio"]
    half_pxl_size = zi_ratio * width / dim

    if direct_cent:
        cent = land_pos.detach().cpu().numpy()
    else:
        cent = land_pos[0, :, dim - 1 -
                        int(i % dim), int(i // dim)].detach().cpu().numpy()
    lens.pixel_size = (2 * zi_ratio * width - 2 * half_pxl_size) / (dim - 1)
    lens.film_size = [dim, dim]
    dx_left_off = cent[1] - zi_ratio * width
    dx_right_off = cent[1] + zi_ratio * width
    dx_top_off = cent[0] + zi_ratio * width
    dx_bot_off = cent[0] - zi_ratio * width
    print('center: ', cent)
    print('width: ', dx_right_off - dx_left_off, 'pixel size: ',
          (dx_right_off - dx_left_off) / dim, 'dim: ', dim)

    rays = lens.sample_ray_uniform_rot(
        wavelength,
        M=config["layers"],
        R=config["sample_rad"],
        view=view,
        rot=rot,
        oy=offset_y,
        ox=offset_x)
    img_rayoptics = lens.render_sft(rays, cent)
    plt.imshow(
        np.flipud(
            img_rayoptics.detach().cpu().numpy()),
        extent=(
            dx_left_off,
            dx_right_off,
            dx_bot_off,
            dx_top_off))
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.colorbar()
    if torch.is_tensor(wavelength):
        plt.savefig(config["display_folder"] +
                    "psf_zoomin_ray_{}_{}.png".format(int(wavelength.item()), psf_idx))
    else:
        plt.savefig(
            config["display_folder"] +
            "psf_zoomin_ray_{}_{}.png".format(
                int(wavelength),
                psf_idx))
    plt.close()
    return img_rayoptics


def plot_e2e(config: Dict[str, Any], meas: torch.Tensor, meas_full: torch.Tensor, gt: torch.Tensor,
             pred: torch.Tensor, lens: do.Lensgroup, R: float, wavelength: float, ep: int) -> None:
    meas_scale = 200 / torch.amax(meas)
    meas_full_scale = 200 / torch.amax(meas_full)
    meas_np = meas_scale.detach().cpu().numpy() * meas.permute(1,
                                                               2, 0).detach().cpu().numpy()
    meas_np_full = meas_full_scale.detach().cpu().numpy(
    ) * meas_full.permute(1, 2, 0).detach().cpu().numpy()
    gt_np = 200 * gt.permute(1, 2, 0).detach().cpu().numpy()
    pred_np = 200 * pred.permute(1, 2, 0).detach().cpu().numpy()
    cv2.imwrite(
        config["record_folder"] +
        '/meas/meas_rgb_mosaic_{}.png'.format(ep),
        np.rot90(meas_np))
    cv2.imwrite(
        config["record_folder"] +
        '/meas_full/meas_{}.png'.format(ep),
        np.rot90(meas_np_full))
    cv2.imwrite(
        config["record_folder"] +
        '/recover_full/recover_rgb_{}.png'.format(ep),
        np.rot90(pred_np))
    cv2.imwrite(
        config["record_folder"] +
        '/gt/gt_full_{}.png'.format(ep),
        np.rot90(gt_np))
    print(
        'max pred',
        np.amax(pred_np),
        config["record_folder"] +
        '/recover_full/recover_rgb_{}.png'.format(ep))

    print('plotting lens')

    _i, _p, oss = trace_all(lens, R, wavelength, config, layers=3, view=0., rot=torch.tensor([0.]))
    ax, fig = lens.plot_raytraces(oss, color='b-', show=False)
    ax.axis('off')
    ax.set_title("")
    fig.savefig(
        config["record_folder"] +
        "/layout/layout_trace_ep_" +
        str(ep) +
        ".png",
        bbox_inches='tight')
    plt.close()

    try:
        _i, _p, oss = trace_all(lens, R, wavelength, config, layers=3, view=20., rot=torch.tensor([-135.]))
        # print (oss.dtype, oss.size())
        ax, fig = lens.plot_raytraces(oss, color='b-', show=False)
        ax.axis('off')
        ax.set_title("")
        fig.savefig(
            config["record_folder"] +
            "/layout_off/layout_trace_off_ep_" +
            str(ep) +
            ".png",
            bbox_inches='tight')
        plt.close()
    except:
        print("Failed to plot off axis")
    return


def plot_loss_curve(config: Dict[str, Any], loss_list: torch.Tensor, dataloss_list: torch.Tensor, rmsloss_list: torch.Tensor, best_idx: int, ep: int) -> None:
    plt.plot(loss_list.numpy()[:ep + 1])
    plt.savefig(config["record_folder"] + '/loss_trend.png')
    plt.close()
    plt.plot(dataloss_list.numpy()[:ep + 1])
    plt.title('Best at {}'.format(best_idx))
    plt.savefig(config["record_folder"] + '/loss_data_trend.png')
    plt.close()
    plt.plot(rmsloss_list.numpy()[:ep + 1])
    plt.savefig(config["record_folder"] + '/loss_rms_trend.png')
    plt.close()
    
    
def plot_diff_psf(config: Dict[str, Any], U: torch.Tensor, I: torch.Tensor, h_load: torch.Tensor, surf: torch.Tensor, view: float) -> None:
    dx_left, dx_right, dy_top, dy_bot = get_borders(config)
    folder = config["record_folder"]
    plt.close()
    plt.imshow(surf.detach().cpu().numpy())
    plt.colorbar()
    plt.savefig('{}/hm_surf_fit.png'.format(folder))
    plt.close()        
    plt.imshow(h_load.detach().cpu().numpy())
    plt.colorbar()
    plt.savefig('{}/hm_load.png'.format(folder))
    plt.close()
    plt.imshow(torch.abs(torch.rot90(I, 1) / torch.sum(I)).detach().cpu().numpy(), extent=(dx_left, dx_right, dy_bot, dy_top))
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.colorbar()
    plt.savefig('{}/free_ref_geo2_{}.png'.format(folder, int(view)))
    plt.close()
    plt.imshow(torch.abs(torch.rot90(torch.flipud(U) / torch.sum(U), 1)).detach().cpu().numpy(), extent=(dy_bot, dy_top, dx_left, dx_right))
    plt.colorbar()
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.savefig('{}/free_ref_wave2_{}.png'.format(folder, int(view)))
    plt.close()
    np.save('free_ref.npy', torch.abs(U).detach().cpu().numpy())
    return


def plot_lens_in_e2e(lens: do.Lensgroup, config: Dict[str, Any], wavelength: float, ep: int) -> None:
    _i, _p, oss_on = trace_all(lens, config["sample_rad"], wavelength, config, layers=3, view=torch.tensor([0.]), rot=torch.tensor([0.]))
    ax, fig = lens.plot_raytraces(oss_on, color='b-', show=False)
    ax.axis('off')
    ax.set_title("")
    fig.savefig(config["record_folder"] + "layout/layout_{}.png".format(ep), bbox_inches="tight", transparent=False)
    plt.close()
    return