
from typing import Any, Dict, List
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from tracer import trace_all
sys.path.append("../")
import diffoptics as do
import torch


def initialize_lens(config: Dict[str, Any],
                    d_list: List[float],
                    device: torch.device,) -> List[do.Surface]:

    system_scale = config["system_scale"]
    R = config["lens_rad"]

    if config["rand_c"] and not config["single_lens"]:
        c_rand = 1 / ((10 + 30 * torch.rand(6)) * system_scale)
        surfaces = [
            do.Aspheric(system_scale * R, system_scale * 0.0, c=c_rand[0], device=device),
            do.Aspheric(system_scale * R, system_scale * d_list[0], c=c_rand[1], device=device),
            do.Aspheric(system_scale * R, system_scale * sum(d_list[:2]), c=-c_rand[2], device=device),
            do.Aspheric(system_scale * R, system_scale * sum(d_list[:3]), c=c_rand[3], device=device),
            do.Aspheric(system_scale * R, system_scale * sum(d_list[:4]), c=c_rand[4], device=device),
            do.Aspheric(system_scale * R, system_scale * sum(d_list[:5]), c=-c_rand[5], device=device)
        ]
    elif config["rand_c"] and config["single_lens"]:
        c_rand = 1 / ((15 + 30 * torch.rand(2)) * system_scale)
        surfaces = [
            do.Aspheric(
                system_scale * R,
                system_scale * 0.0,
                c=c_rand[0],
                device=device),
            do.Aspheric(
                system_scale * R,
                system_scale * d_list[0],
                c=c_rand[1],
                device=device)]
    elif not config["rand_c"] and config["single_lens"]:
        print('single')
        surfaces = [do.Aspheric(system_scale * R,
                                system_scale * 0.0,
                                c=1 / (9.3506 * system_scale),
                                device=device),
                    do.Aspheric(system_scale * R,
                                system_scale * d_list[0],
                                c=1 / (87.50819 * system_scale),
                                device=device)]
    else:
        surfaces = [do.Aspheric(system_scale * R,
                                system_scale * 0.0,
                                c=1 / (15.25 * system_scale),
                                device=device),
                    do.Aspheric(system_scale * R,
                                system_scale * d_list[0],
                                c=1 / (31.70 * system_scale),
                                device=device),
                    do.Aspheric(system_scale * R,
                                system_scale * sum(d_list[:2]),
                                c=-1 / (52.62 * system_scale),
                                device=device),
                    do.Aspheric(system_scale * R,
                                system_scale * sum(d_list[:3]),
                                c=1 / (14.77 * system_scale),
                                device=device),
                    do.Aspheric(system_scale * R,
                                system_scale * sum(d_list[:4]),
                                c=1 / (29.10 * system_scale),
                                device=device),
                    do.Aspheric(system_scale * R,
                                system_scale * sum(d_list[:5]),
                                c=-1 / (32.03 * system_scale),
                                device=device)]
    return surfaces


def initialize_materials(config: Dict[str, Any]) -> List[do.Material]:
    if config["single_lens"]:
        materials = [
            do.Material('air'),
            do.Material('n-laf2'),
            do.Material('air')
        ]
    else:
        materials = [
            do.Material('air'),
            do.Material('n-laf2'),
            do.Material('air'),
            do.Material('n-sf10'),
            do.Material('air'),
            do.Material('n-laf2'),
            do.Material('air')
        ]
    return materials


def load_lens(filename: str) -> do.Lensgroup:
    with open(filename, 'rb') as pickle_file:
        surf = pickle.load(pickle_file)
    return surf


def plot_lens_with_ray(lens: do.Lensgroup, config: Dict[str, Any], wavelength: float) -> None:
    # Trace rays through the lens system
    _i, _p, oss_on = trace_all(lens, config["sample_rad"], wavelength, config, layers=3, view=torch.tensor([0.]), rot=torch.tensor([0.]))
    ax, fig = lens.plot_raytraces(oss_on, color='b-', show=False)
    ax.axis('off')
    ax.set_title("")
    fig.savefig(config["display_folder"] + "layout_on.png", bbox_inches="tight", transparent=False)
    plt.close()

    _i, _p, oss_mid = trace_all(lens, config["sample_rad"], wavelength, config, layers=3, view=config["view_mid"], rot=torch.tensor([-180.]), offset_y=config["offset_mid"], offset_x=0.)
    _i, _p, oss_off = trace_all(lens, config["sample_rad"], wavelength, config, layers=3, view=config["view_off"], rot=torch.tensor([-180.]), offset_y=config["offset_off"], offset_x=0.)
    ax, fig = lens.plot_raytraces(oss_off, color='r-', show=False)
    ax.axis('off')
    ax.set_title("")
    fig.savefig(config["display_folder"] + "layout_off.png", bbox_inches='tight')
    plt.close()

    ax, fig = lens.plot_multi_raytraces([oss_on, oss_off], color=['blueviolet', 'orange'], show=False)
    ax.axis('off')
    ax.set_title("")
    scalebar_x = np.linspace(config["scalebar_x_left"], config["scalebar_x_right"], 8).reshape(8) * 1
    scalebar_y = -0.7 * np.ones(8).reshape(8) * 1
    ax.plot(scalebar_x, scalebar_y, 'black')
    fig.savefig(config["display_folder"] + "layout_merged.png", bbox_inches='tight')
    plt.close()
    return