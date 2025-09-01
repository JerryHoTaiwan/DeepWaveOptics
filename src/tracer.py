from typing import Any, Dict, List, Tuple, Callable
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from waveoptics import gen_aperture, create_wf, display_wf
from utils import create_sensor_grids, get_borders, interpolation_symm, point2line_distance, hit_sphere_parallel, normalize_vector_torch

sys.path.append("../")
import deeplens.optics as deepoptics
import diffoptics as do
from deeplens import GeoLens


def plot_overlay_sensor(ps_back: torch.Tensor, Img_pos_all: torch.Tensor, proj_back_chief: torch.Tensor) -> None:
    # Plot the distribution of landing positions and pixel locations
    plt.scatter(ps_back[:, 1].detach().cpu().numpy(
    ), ps_back[:, 0].detach().cpu().numpy(), s=4, color='blue')
    plt.scatter(Img_pos_all[1, :, :].detach().cpu().numpy(
    ), Img_pos_all[0, :, :].detach().cpu().numpy(), s=4, color='red')
    plt.savefig('pos_cmp_normal.png')
    plt.close()
    plt.scatter(proj_back_chief[:, 1].detach().cpu().numpy(
    ), proj_back_chief[:, 0].detach().cpu().numpy(), s=4)
    plt.savefig('proj_back_chief_normal.png')
    plt.close()
    proj_back_chief_2d = proj_back_chief.view(513, 513, 3)
    plt.scatter(proj_back_chief_2d[0::32,
                                    0::32,
                                    1].detach().cpu().numpy().reshape(-1),
                proj_back_chief_2d[0::32,
                                    0::32,
                                    0].detach().cpu().numpy().reshape(-1),
                s=10)
    plt.savefig('proj_back_chief_normal_sparse.png')
    plt.close()
    return


def get_exit_pupil(lens: do.Lensgroup, wavelength: float, rad: float, stop: int=3) -> torch.Tensor:
    ray_scatter = lens.sample_ray_common_o(
        rad,
        wavelength,
        oy=0.01,
        M=5000,
        mode="get_chief")
    ps, oss, _v = lens.trace_to_sensor_r(ray_scatter)
    pts_stop = oss[:, stop, :]

    # find the ray that is the closest to the center of aperture
    rad = torch.sqrt(pts_stop[:, 0] * pts_stop[:, 0] +
                     pts_stop[:, 1] * pts_stop[:, 1])
    chief_idx = torch.argmin(rad)

    # check the last segment of chief ray
    last_start = oss[chief_idx, -2, :]
    last_end = oss[chief_idx, -1, :]
    last_vec = last_end - last_start
    y_dir = last_end[0] - last_start[0]
    y_diff = 0 - last_start[0]
    t_xp = y_diff / y_dir
    xp_pos = last_start + t_xp * last_vec
    return xp_pos[2]


def sample_ray_common_o(R, wavelength, ox=0., oy=0., dist_z=-12., M=15, mode="none", 
                        z_in_sample=1., datatype=torch.float, device=torch.device('cpu')):
    if mode == "get_chief":
        N = 2 * M - 1
        o = torch.zeros(N, 3, device=device)
    elif mode == "circ":
        N = 3 * M * M - 3 * M + 1
        o = torch.zeros(N, 3, device=device)
    else:
        o = torch.zeros(M * M, 3, device=device)
    o[:, 0] = oy
    o[:, 1] = ox
    o[:, 2] = dist_z
    if mode == "get_chief":
        tgt_pos = torch.zeros(N, 3, device=device)
        tgt_y = torch.linspace(-R, R, N, device=device)
        tgt_pos[:, 0] = tgt_y
        d = tgt_pos - o
    elif mode == "grid":
        ori = torch.tensor([0., 0., dist_z], dtype=datatype, device=device)[None, None, :]
        y, x = torch.meshgrid(
            torch.linspace(-R, R, M, dtype=datatype, device=device),
            torch.linspace(-R, R, M, dtype=datatype, device=device),
            indexing='ij'
        )
        z = z_in_sample * torch.ones_like(x) + dist_z
        pos = torch.stack((y, x, z), axis=2)
        d = (pos - ori).view(M * M, 3)
    elif mode == "circ":
        # unused for now but can be used for circular sampling
        # the for loop can be optimized
        r_sample = torch.linspace(0, R, M, device=device, dtype=datatype)
        d = torch.zeros_like(o)
        cnt = 0
        for i, r in enumerate(r_sample):
            num = 6 * i + 1
            if i == 0:
                phi_sample = torch.tensor([0.], device=device, dtype=datatype)
            else:
                phi_sample = torch.linspace(0, 2 * torch.pi, num, device=device, dtype=datatype)[:-1]
            pos = torch.zeros(3, len(phi_sample), device=device, dtype=datatype)
            pos[0, :] = r * torch.sin(phi_sample)
            pos[1, :] = r * torch.cos(phi_sample)
            ori = o[cnt:cnt+len(phi_sample), :]
            d[cnt:cnt+len(phi_sample), :] = (pos.T - ori)
            cnt += len(phi_sample)
    d_norm = torch.norm(d, dim=1)[:, None]
    d = d / d_norm
    wvln_um = wavelength / 1000 # convert to um
    return deepoptics.Ray(o.to(datatype), d.to(datatype), wvln=wvln_um, device=device)


def init_opl_prl(
        config: Dict[str, Any],
        rays: deepoptics.Ray,
        view: float,
        rot: float,
        depth: float,) -> deepoptics.Ray:
    """
    For DeepLens Rays only. Initialize the optical path length for the rays according to the incident angles. 

    Args:
        config (Dict[str, Any]): configurations. Specify datatype and device.
        rays (deepoptics.Ray): Parallel rays
        view (float): viewing angle of the ray directions
        rot (float): rotation angle of the ray directions
        depth (float): a large enough distance to make sure the rays are parallel

    Returns:
        deepoptics.Ray: rays with updated opl
    """
    if config["dtype"] == "float64":
        datatype = torch.float64
    elif config["dtype"] == "float32":
        datatype = torch.float32
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')    

    # make sure opls are initialized to zero
    assert torch.sum(rays.opl) == 0
    r_idx_air = 1.0 # refractive index of air

    # the first source at a distance
    angle = torch.deg2rad(torch.tensor([view], dtype=datatype, device=device))
    phi = torch.deg2rad(torch.tensor([rot], dtype=datatype, device=device))
    
    # an alternative way to calculate the first distance
    sign = -torch.sign(phi + 1e-8)
    baseline = torch.tensor([torch.tan(phi + torch.pi / 2), -1, rays.o[0, 0] - torch.tan(phi + torch.pi / 2) * rays.o[0, 1]]) # ax+by+c=0
    if torch.abs(torch.abs(rot) - 180) < 1e-2 and torch.sign(rot) == -1:
        sign *= -1
    first_dist = point2line_distance(rays.o, baseline) * torch.sin(angle) * sign
    rays.opl += first_dist * r_idx_air
    return rays


def compute_opl_prl(
        config: Dict[str, Any],
        oss: torch.Tensor,
        lens: do.Lensgroup,
        wavelength: float,
        pt_at_sphere: torch.Tensor,
        view: float,
        rot: float,
        depth: float,) -> torch.Tensor:
    if config["dtype"] == "float64":
        datatype = torch.float64
    elif config["dtype"] == "float32":
        datatype = torch.float32
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    opl = torch.zeros(oss.shape[0], dtype=datatype).to(device)
    r_idx_air = lens.materials[0].ior(wavelength)

    # the first source at a distance: remove it in the futre, the alternative way is better
    angle = torch.deg2rad(torch.tensor([view], dtype=datatype, device=device))
    phi = torch.deg2rad(torch.tensor([rot], dtype=datatype, device=device))
    y_pos = torch.tan(-angle) * torch.cos(phi) * depth * (-1)
    x_pos = torch.tan(-angle) * torch.sin(phi) * depth * (-1)
    source = torch.tensor([0., 0., depth], device=device,
                          dtype=datatype)[None, :]
    source[0, 0] = y_pos
    source[0, 1] = x_pos
    first_diff = oss[:, 0, :] - source
    first_dist = torch.norm(first_diff, dim=1)
    
    # an alternative way to calculate the first distance
    sign = -torch.sign(phi + 1e-8)
    baseline = torch.tensor([torch.tan(phi + torch.pi / 2), -1, oss[0, 0, 0] - torch.tan(phi + torch.pi / 2) * oss[0, 0, 1]]) # ax+by+c=0
    if torch.abs(torch.abs(rot) - 180) < 1e-2 and torch.sign(rot) == -1:
        sign *= -1
    first_dist = point2line_distance(oss[:, 0, :], baseline) * torch.sin(angle) * sign
    opl += first_dist * r_idx_air

    for i in range(0, len(lens.materials)):
        diff = (oss[:, i + 1, :] - oss[:, i, :])
        assert not torch.isnan(torch.sum(diff))
        dist = torch.norm(diff, dim=1)
        r_idx = lens.materials[i].ior(wavelength)
        opl += dist * r_idx
    last_diff = (pt_at_sphere - oss[:, -1, :])
    last_dist = torch.sqrt(torch.sum(last_diff * last_diff, dim=1))
    opl -= last_dist * r_idx_air
    return opl



def send_chief_ray(
        config: Dict[str, Any],
        lens: do.Lensgroup,
        wavelength: float=440.,
        dim: int=255,
        layer: int=400,
        plot: bool=False):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if config["dtype"] == "float64":
        datatype = torch.float64
    elif config["dtype"] == "float32":
        datatype = torch.float32

    dx_left, dx_right, dy_top, dy_bot = get_borders(config)
    stop = config["stop_ind"]

    # front: before the aperture, back: after the aperture
    # depend on the curvature at aperture
    if config["use_deeplens"]:
        stop_d = lens.surfaces[stop - 1].d # make sure to change it in the future
    else:
        stop_d = lens.surfaces[stop - 1].d # make the value 0 when comparing with real experiments
    materials_front_rev = lens.materials[:stop]
    surfaces_front_rev = lens.surfaces[:stop - 1]
    materials_back = lens.materials[stop - 1:]
    surfaces_back = lens.surfaces[stop - 1:]
    if config["use_deeplens"]:
        front_system_rev = GeoLens()
        back_system = GeoLens()
        d_sensor = lens.d_sensor
        back_system.load_external(surfaces_back, materials_back, config["width"], d_sensor)
        front_system_rev.load_external(surfaces_front_rev, materials_front_rev, config["width"], -1e-6)
        ray_stop = sample_ray_common_o(
            R=3 * config["system_scale"],
            wavelength=wavelength,
            M=layer,
            dist_z=stop_d,
            mode="grid",
            z_in_sample=config["z_in_sample"],
            datatype=datatype,
            device=device
            )
        ray_stop.o[:, 2] -= 1e-6 # hacking: make sure the ray is in front of the aperture
        ori_stop = torch.clone(ray_stop.o)
        dir_stop = torch.clone(ray_stop.d)
        ray_back = back_system.trace2sensor(ray_stop)
        ps_back = ray_back.o
    else:
        front_system_rev = do.Lensgroup(device=device)
        back_system = do.Lensgroup(device=device)
        front_system_rev.load(surfaces_front_rev, materials_front_rev)
        front_system_rev.d_sensor = -1e-6
        back_system.load(surfaces_back, materials_back)
        back_system.d_sensor = lens.d_sensor  
        # tracing back
        ray_stop = back_system.sample_ray_common_o(
            R=4 * config["system_scale"],
            wavelength=wavelength,
            M=layer,
            dist_z=stop_d,
            mode="grid",
            z_in_sample=config["z_in_sample"],
            )
        ori_stop = torch.clone(ray_stop.o)
        dir_stop = torch.clone(ray_stop.d)
        ps_back, _oss_back, _valid = back_system.trace_to_sensor_r(ray_stop)
    xy_pos = (ps_back[:, :2])[:, :, None, None]
    land_pos = torch.zeros(1, 2, dim, dim, dtype=datatype, device=device)

    # force the datetype to be float
    sub_sp = int((config["dim"] - 1) / config["chief_gap"]) + 1
    chief_idx = torch.zeros(sub_sp, sub_sp, dtype=torch.long, device=device)
    Img_pos = create_sensor_grids(config, dx_left, dx_right, dy_top, dy_bot, sub_sp) # sensor coordinates
    # batch the computation of diff
    if Img_pos.shape[2] > 16:
        # update: batch the computation along the width of sensor to avoid out of memory
        batch_size = 4
        batch_num = int(Img_pos.shape[2] / batch_size) + 1
        chief_idx = torch.zeros(Img_pos.shape[2], Img_pos.shape[3], device=device)
        for i in range(batch_num):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, Img_pos.shape[2])
            diff_tmp = torch.norm(Img_pos[:, :, start_idx:end_idx, :] - xy_pos, dim=1)
            chief_idx[start_idx:end_idx] = torch.argmin(diff_tmp, dim=0)
    else:
        diff = torch.norm(Img_pos - xy_pos, dim=1)
        chief_idx = torch.argmin(diff, dim=0)
    plt.scatter(xy_pos[:, 0, 0, 0].detach().cpu().numpy(), xy_pos[:, 1, 0, 0].detach().cpu().numpy(), s=0.1)
    plt.scatter(Img_pos[0, 1, :, :].detach().cpu().numpy(), Img_pos[0, 0, :, :].detach().cpu().numpy(), s=4)
    plt.savefig("chief_ray_all.png")
    plt.close()

    # the entire img grids
    chief_idx_rs = chief_idx.view(sub_sp * sub_sp).to(torch.long)
    Img_pos_all = create_sensor_grids(config, dx_left, dx_right, dy_top, dy_bot, dim)[0, ...]
    grid_y = Img_pos_all[0, :, :]
    grid_x = Img_pos_all[1, :, :]
    dir_tmp = dir_stop[chief_idx_rs, :]
    rot_approx = torch.flip(
        torch.rot90(
            torch.rad2deg(
                torch.atan2(
                    grid_y, grid_x)), 1), dims=(
            1,))
    rad = torch.sqrt(dir_tmp[:, 0] * dir_tmp[:, 0] +
                     dir_tmp[:, 1] * dir_tmp[:, 1] + 1e-15)
    view = torch.rad2deg(torch.atan2(rad + 1e-15, dir_tmp[:, 2] + 1e-15))
    view_ds = view.view(sub_sp, sub_sp)
    view_mts = interpolation_symm(view_ds, shape=(config["dim"], config["dim"]))
    if config["use_deeplens"] or True:
        view_mts[::config["chief_gap"], ::config["chief_gap"]] = view_ds

    # TODO: after interpolating all angles, convert them to directional vector
    # and trace back to front lens system
    dir_stop_interp = torch.zeros(
        config["dim"] * config["dim"],
        3,
        dtype=datatype,
        device=device)
    rad_interp = torch.sin(torch.deg2rad(view_mts)).view(config["dim"] * config["dim"])
    dir_stop_interp[:, 2] = torch.cos(
        torch.deg2rad(view_mts)).view(config["dim"] * config["dim"])
    dir_stop_interp[:, 1] = rad_interp * \
        torch.sin(torch.deg2rad(rot_approx)).reshape(config["dim"] * config["dim"])
    dir_stop_interp[:, 0] = rad_interp * \
        torch.cos(torch.deg2rad(rot_approx)).reshape(config["dim"] * config["dim"])

    # tracing front
    ori_stop_interp = (ori_stop[0, :])[None, :].repeat(config["dim"] * config["dim"], 1)
    if config["use_deeplens"]:
        ray_stop_interp_inv = deepoptics.Ray(ori_stop_interp, -
                                    dir_stop_interp, wavelength / 1000, device=device)
        ray_back = front_system_rev.trace2sensor(ray_stop_interp_inv, record=False)
        d_first_unit_interp = -torch.clone(ray_back.d)
        o_first_interp = torch.clone(ray_back.o)
    else:
        ray_stop_interp_inv = do.Ray(ori_stop_interp, -
                                    dir_stop_interp, wavelength, device=device)
        _ps, oss_front_interp, _valid = front_system_rev.trace_to_sensor_r(
            ray_stop_interp_inv)
        d_first_interp = torch.clone(
            oss_front_interp[:, -2, :] - oss_front_interp[:, -1, :])
        d_first_norm_interp = torch.norm(d_first_interp, dim=1)
        d_first_unit_interp = d_first_interp / d_first_norm_interp[:, None]
        o_first_interp = torch.clone(oss_front_interp[:, -1, :])
    tz = -config["dist_scene"] / d_first_unit_interp[:, 2]
    proj_back_chief = o_first_interp + d_first_unit_interp * tz[:, None]

    if plot:
        plot_overlay_sensor(ps_back, Img_pos_all, proj_back_chief)

    # skip the dot product as we know the mathematical solution
    cos_first = d_first_unit_interp[:, 2]
    theta_first = torch.acos(cos_first)
    print(
        "FOV: ",
        torch.rad2deg(
            torch.amax(theta_first)).item())

    check_unique = torch.unique(chief_idx)
    assert len(check_unique) == sub_sp * sub_sp
    assert not torch.isnan(torch.sum(proj_back_chief))
    return proj_back_chief, Img_pos_all, d_first_unit_interp, o_first_interp, land_pos


def trace_all(
        lens: do.Lensgroup,
        R: float,
        wavelength: float,
        config: Dict[str, Any],
        layers: int=100,
        psf_idx: int=-1,
        view: int=0,
        rot: int=0,
        offset_y: int=0,
        offset_x: int=0,
        depth: int=-1e5,
        adj_pxl: bool=False,
        land_pos: bool=None,
        use_parallel: bool=True,
        origin: torch.Tensor=None,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rad = config["sample_rad"] / 2
    stop_ind = config["stop_ind"]


    # dynamically controll the sampling rate
    if torch.is_tensor(view):
        abs_view = torch.abs(view)
    else:
        abs_view = np.abs(view)

    if layers > 10:
        if abs_view > 30:
            layers = 100
        elif abs_view > 18 and layers > 10:
            layers = 100  # 140
        elif abs_view <= 18 and abs_view <= 15 and layers > 10:
            layers = 100
        elif abs_view <= 15 and layers > 10:
            layers = 50
        if not config["single_lens"]:
            layers = config["layers"]
    layers = config["layers"]

    if config["use_deeplens"]:
        datatype = torch.float32 if config["dtype"] == "float32" else torch.float64
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        xp_pos, _pupilr = lens.exit_pupil()
        if use_parallel:
            traced_ray = sample_ray_uniform_rot(wavelength, M=layers, R=R, view=view, rot=rot, oy=offset_y, ox=offset_x, datatype=datatype, device=device)
            traced_ray = init_opl_prl(config, traced_ray, view, rot, depth)
        else:
            traced_ray = sample_ray_common_o(R, wavelength, M=layers, dist_z=origin[2], ox=origin[1], oy=origin[0], mode="circ", datatype=datatype, device=device)
        traced_ray.coherent = True
        traced_ray = lens.trace2sensor(traced_ray)
        center = traced_ray.o[0]
        radius = lens.d_sensor - xp_pos
        ps = traced_ray.o
        dir_last = -traced_ray.d
        opl = traced_ray.opl
        _intersect, _t0, _t1, intersect_0, _intersect_1 = hit_sphere_parallel(
            ps, dir_last, center, radius) 
        last_path = intersect_0 - ps
        last_dist = torch.norm(last_path, dim=1)
        opl -= last_dist
        phase = 2 * torch.pi * (opl * 1e6 / wavelength) 
        oss = None
    else:
        if "single_psf" in config["blocks"]:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            xp_pos = torch.tensor([config["xp_pos"]], device=device)
        else:
            xp_pos = get_exit_pupil(
                lens=lens,
                wavelength=wavelength,
                rad=rad,
                stop=stop_ind)
        ray_init = lens.sample_ray_uniform_rot(
            wavelength, M=layers, R=R, view=view, rot=rot, oy=offset_y, ox=offset_x)
        ps, oss, _v = lens.trace_to_sensor_r(ray_init)

        # get the chief ray and plot circle
        center = ps[0]  # sphere center
        radius = lens.d_sensor - xp_pos

        ori_from_sensor = oss[:, -1, :]
        ori_before_sensor = oss[:, -2, :]
        dir_last = normalize_vector_torch(ori_before_sensor - ori_from_sensor)
        _intersect, _t0, _t1, intersect_0, _intersect_1 = hit_sphere_parallel(
            ori_from_sensor, dir_last, center, radius)
        opl = compute_opl_prl(
            config,
            oss,
            lens,
            wavelength,
            intersect_0,
            depth=depth,
            view=view,
            rot=rot)
        # opd = opl - opl[0]
        phase = 2 * torch.pi * (opl * 1e6 / wavelength)  # * 0.
    wf = create_wf(pts=intersect_0, phase=phase, uni_phase=False)
    if torch.is_tensor(lens.d_sensor):
        focal_pos = lens.d_sensor
    else:
        focal_pos = torch.tensor([lens.d_sensor])

    if config["plot_wf"]:
        print('display')
        display_wf(wf, psf_idx, config)

    U = gen_aperture(
        wf=wf,
        sphere_cent=center,
        focal_pos=focal_pos,
        wavelength=wavelength,
        config=config,
        adj_pxl=adj_pxl,
        land_pos=land_pos,
        psf_idx=psf_idx)
    irrad = torch.abs(U * torch.conj(U))  # / (norm * norm)
    return irrad, U, ps[..., :2], oss


def sample_ray_uniform_rot(wavelength, view=0.0, rot=0.0, M=15, R=None, oy=0., ox=0., datatype=torch.float, device=torch.device('cpu')):
    N = 3 * M * M - 3 * M + 1
    o = torch.zeros(N, 3, device=device, dtype=datatype)
    r_sample = torch.linspace(0, R, M, device=device, dtype=datatype)
    cnt = 0
    cutoff = 800

    for i, r in enumerate(r_sample):
        num = 6 * i + 1
        if i == 0:
            phi_sample = torch.tensor([0.], device=device, dtype=datatype)
        else:
            if i > cutoff:
                num = 6 * cutoff + 1
            phi_sample = torch.linspace(0, 2 * torch.pi, num, device=device, dtype=datatype)[:-1] 
        xp = r * torch.cos(phi_sample)
        yp = r * torch.sin(phi_sample)
        o[cnt:cnt+len(phi_sample), 0] = yp
        o[cnt:cnt+len(phi_sample), 1] = xp
        cnt += len(phi_sample)
    o[:, 0] += oy
    o[:, 1] += ox

    angle = torch.deg2rad(torch.tensor([view], dtype=datatype, device=device))
    phi = torch.deg2rad(torch.tensor([rot], dtype=datatype, device=device))
    d = torch.stack((
        torch.sin(angle) * torch.cos(phi) * torch.ones_like(o[:, 0]),
        torch.sin(angle) * torch.sin(phi) * torch.ones_like(o[:, 0]),
        torch.cos(angle) * torch.ones_like(o[:, 0])), axis=-1
    )
    wvln_um = wavelength / 1000
    return deepoptics.Ray(o[:cnt, :], d[:cnt, :], wvln=wvln_um, device=device)
