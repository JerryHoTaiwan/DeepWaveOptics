import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple
import sys
from tracer import trace_all
from utils import create_sensor_grids
sys.path.append("../")
import diffoptics as do
import deeplens.optics as deepoptics


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
    # cent *= 0
    # width = 1.2
    # zi_ratio = 1.0
    dx_left_off = cent[1] - zi_ratio * width
    dx_right_off = cent[1] + zi_ratio * width
    dx_top_off = cent[0] + zi_ratio * width
    dx_bot_off = cent[0] - zi_ratio * width
    offaxis_pos = create_sensor_grids(config, dx_left_off, dx_right_off,
                                      dx_top_off, dx_bot_off, zi_dim)
    irrad_off, irrad_off_comp, ps, _o = trace_all(lens, config["sample_rad"], wavelength, config, view=view,
                                  rot=rot, offset_y=offset_y, offset_x=offset_x, adj_pxl=True, land_pos=offaxis_pos)
    assert not torch.isnan(torch.sum(irrad_off))
    assert torch.abs(torch.sum(irrad_off)) > 0
    fig_name = config["display_folder"] + "psf_zoomin_wave_{}_{}_{}.png".format(
        int(wavelength), psf_idx, config["layers"])
    print('width: ', dx_right_off - dx_left_off, 'pixel size: ',
          (dx_right_off - dx_left_off) / zi_dim, 'dim: ', zi_dim)
    plt.imshow(
        (irrad_off / torch.sum(irrad_off)).detach().cpu().numpy(), cmap='gray',
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
    
    scatter_name = config["display_folder"] + "psf_zoomin_wave_{}_{}_{}_scatter.png".format(int(wavelength), psf_idx, config["layers"])
    plt.scatter(ps[:, 0].detach().cpu().numpy(), ps[:, 1].detach().cpu().numpy(), s=1)
    plt.savefig(scatter_name)
    plt.close()
    return irrad_off, irrad_off_comp, [dx_left_off, dx_right_off, dx_top_off, dx_bot_off]


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
    # zi_ratio = config["zi_ratio"][str(psf_idx)]
    zi_ratio = 0.2
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

    if config["use_deeplens"]:
        ray_g = sample_ray_uniform_rot(
            wavelength,
            M=config["layers"],
            R=config["sample_rad"],
            view=view,
            rot=rot,
            oy=offset_y,
            ox=offset_x,
            datatype=land_pos.dtype,
            device=land_pos.device)

        cent = torch.from_numpy(cent).to(ray_g.o.device)
        ray_g_traced = lens.trace2sensor(ray_g)
        ray_g_traced.o = ray_g_traced.o.reshape(-1, 1, 3)
        ray_g_traced.d = ray_g_traced.d.reshape(-1, 1, 3)
        ray_g_traced.ra = ray_g_traced.ra.reshape(-1, 1)
        img_rayoptics = deepoptics.monte_carlo.forward_integral(ray=ray_g_traced, ps=lens.pixel_size, ks=dim, pointc_ref=-cent)
        print(torch.sum(img_rayoptics))
        assert torch.sum(img_rayoptics) > 0
        
        img_rayoptics = img_rayoptics[0, ...] / torch.sum(img_rayoptics)
        img_rayoptics = torch.rot90(img_rayoptics, 1, dims=(0, 1))
        
        valid_idx = torch.where(ray_g_traced.ra > 0)[0]
        valid_o = ray_g_traced.o[valid_idx, 0, :].detach().cpu().numpy()
        plt.scatter(valid_o[:, 0], valid_o[:, 1], s=1)
        plt.savefig(config["display_folder"] + "psf_zoomin_ray_{}_{}_scatter.png".format(int(wavelength), psf_idx))
        plt.close()

    else:

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
            dx_top_off), cmap='gray')
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
