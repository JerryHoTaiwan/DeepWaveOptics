import torch
import sys
from typing import Any, Dict, Tuple, List
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from waveoptics import dummy_wf, display_wf, gen_aperture
from sensor import create_sensor_grids
from utils import normalize_vector_torch, hit_sphere_parallel, my_interpolation_symm, get_borders, point2line_distance
sys.path.append("../")
import diffoptics as do


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


def compute_efl(lens: do.Lensgroup, config: Dict[str, Any], wavelength: float) -> float:
    R = config["sample_rad"]
    _i, _p, oss_on = trace_all(lens, R, wavelength, config, layers=2, view=torch.tensor([0.]), rot=torch.tensor([0.]))
    d_last = oss_on[1, -1, :] - oss_on[1, -2, :]
    level = oss_on[1, 0, 1]
    hit_axis_frac = (0 - oss_on[1, -1, 1]) / d_last[1]
    hit_level_frac = (level - oss_on[1, -1, 1]) / d_last[1]
    pts_hit_axis = oss_on[1, -1, :] + d_last * hit_axis_frac
    pts_hit_level = oss_on[1, -1, :] + d_last * hit_level_frac
    efl = pts_hit_axis[2] - pts_hit_level[2]
    print("Effective Focal Length: ", (pts_hit_axis[2] - pts_hit_level[2]).item())
    return efl


def get_exit_pupil(lens: do.Lensgroup, wavelength: float, rad: float, stop: int=3) -> torch.Tensor:
    ray_scatter = lens.sample_ray_common_o(
        rad,
        wavelength,
        oy=0.01,
        M=50000,
        mode="get_chief")
    ps, oss = lens.trace_to_sensor_r(ray_scatter)
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

    # the first source at a distance
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
        ):
    rad = config["sample_rad"] / 2
    stop_ind = config["stop_ind"]
    xp_pos = get_exit_pupil(
        lens=lens,
        wavelength=wavelength,
        rad=rad,
        stop=stop_ind)

    # dynamically controll the sampling rate
    if torch.is_tensor(view):
        abs_view = torch.abs(view)
    else:
        abs_view = np.abs(view)

    if layers > 10:
        if abs_view > 18 and layers > 10:
            layers = 150  # 140
        elif abs_view <= 18 and abs_view <= 15 and layers > 10:
            layers = 70
        elif abs_view <= 15 and layers > 10:
            layers = 50
        if not config["single_lens"]:
            layers = 60

    ray_init = lens.sample_ray_uniform_rot(
        wavelength, M=layers, R=R, view=view, rot=rot, oy=offset_y, ox=offset_x)
    # ray_init = lens.sample_ray_common_o(R=R, wavelength=wavelength, M=layers, mode="circ", dist_z=config["light_source_pos"])
    ps, oss = lens.trace_to_sensor_r(ray_init)

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

    opd = opl - opl[0]
    phase = 2 * torch.pi * (opd * 1e6 / wavelength)  # * 0.
    wf = dummy_wf(pts=intersect_0, phase=phase, uni_phase=False)
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
    return irrad, ps[..., :2], oss


def send_chief_ray(
        config: Dict[str, Any],
        lens: do.Lensgroup,
        wavelength: float=440.,
        dim: int=255,
        layer: int=400,
        plot: bool=False):
    datatype = torch.float
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dx_left, dx_right, dy_top, dy_bot = get_borders(config)
    stop = config["stop_ind"]

    # front: before the aperture, back: after the aperture
    front_system_rev = do.Lensgroup(device=device)
    back_system = do.Lensgroup(device=device)

    # depend on the curvature at aperture
    stop_d = lens.surfaces[stop - 1].d
    materials_front_rev = lens.materials[:stop]
    surfaces_front_rev = lens.surfaces[:stop - 1]
    materials_back = lens.materials[stop - 1:]
    surfaces_back = lens.surfaces[stop - 1:]
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
    ps_back, _oss_back = back_system.trace_to_sensor_r(ray_stop)
    xy_pos = (ps_back[:, :2])[:, :, None, None]
    land_pos = torch.zeros(1, 2, dim, dim, dtype=datatype, device=device)

    # force the datetype to be float
    sub_sp = int((config["dim"] - 1) / config["chief_gap"]) + 1
    chief_idx = torch.zeros(sub_sp, sub_sp, dtype=torch.long, device=device)
    Img_pos = create_sensor_grids(config, dx_left, dx_right, dy_top, dy_bot, sub_sp)
    diff = torch.norm(Img_pos - xy_pos, dim=1)
    chief_idx = torch.argmin(diff, dim=0)

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
    view_mts = my_interpolation_symm(view_ds, shape=(config["dim"], config["dim"]))

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
    ray_stop_interp_inv = do.Ray(ori_stop_interp, -
                                 dir_stop_interp, wavelength, device=device)
    _ps, oss_front_interp = front_system_rev.trace_to_sensor_r(
        ray_stop_interp_inv)
    d_first_interp = torch.clone(
        oss_front_interp[:, -2, :] - oss_front_interp[:, -1, :])
    d_first_norm_interp = torch.norm(d_first_interp, dim=1)
    d_first_unit_interp = d_first_interp / d_first_norm_interp[:, None]
    o_first_interp = torch.clone(oss_front_interp[:, -1, :])
    tz = -config["dist_scene"] / d_first_unit_interp[:, 2]
    proj_back_chief = o_first_interp + d_first_unit_interp * tz[:, None]
    dir_pick = d_first_unit_interp
    ori_pick = o_first_interp

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
    return proj_back_chief, Img_pos_all, dir_pick, ori_pick, land_pos


def place_img_ongrid(
        pos: torch.Tensor,
        obj_max: float,
        act_max: float,
        channel_idx: int=0,
        res: int=511,
        filename_list: List[str]=['cameraman.png'],
        ) -> torch.Tensor:
    datatype = torch.float
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    act_max  = torch.amax(pos[:, :2]) * 1.02
    obj_max = torch.amax(pos[:, :2])
    pxl_size = (2 * act_max) / res
    obj_pxls = int(obj_max // pxl_size)
    start_idx = int(res // 2 - obj_pxls)
    end_idx = int(res // 2 + obj_pxls) + 1

    img_stack = np.zeros(
        (end_idx - start_idx,
            end_idx - start_idx,
            len(filename_list)))
    for i, fn in enumerate(filename_list):
        img = np.rot90(np.flip(cv2.imread(fn)[:, :, channel_idx] / 255., axis=1), 3)
        img_rs = cv2.resize(
            img, (end_idx - start_idx, end_idx - start_idx))
        img_stack[:, :, i] = img_rs
    if not torch.amax(pos[:, :2]) < act_max:
        print('scene boundary', torch.amax(pos[:, :2]), act_max)
    assert torch.amax(pos[:, :2]) < act_max

    grids = torch.zeros(
        res,
        res,
        len(filename_list),
        dtype=datatype,
        device=device)
    grids[start_idx:end_idx, start_idx:end_idx,
          :] = torch.from_numpy(img_stack).to(device)

    pos_on_grid = pos[:, :2] / pxl_size + res // 2
    top_y = (torch.ceil(pos_on_grid[:, 0])).to(torch.long)
    bot_y = (torch.floor(pos_on_grid[:, 0])).to(torch.long)
    right_x = (torch.ceil(pos_on_grid[:, 1])).to(torch.long)
    left_x = (torch.floor(pos_on_grid[:, 1])).to(torch.long)

    alpha = (pos_on_grid[:, 0] - bot_y)[:, None]
    beta = (pos_on_grid[:, 1] - left_x)[:, None]
    lu_val = grids[top_y, left_x]
    lb_val = grids[bot_y, left_x]
    ru_val = grids[top_y, right_x] 
    rb_val = grids[bot_y, right_x] 
    interp_val = alpha * beta * ru_val + \
        (1 - alpha) * beta * rb_val + alpha * (1 - beta) * lu_val + (1 - alpha) * (1 - beta) * lb_val
    return interp_val


def place_img_on_grid_exact(
    res: int,
    channel_idx: int,
    filename_list: List[str],
    ) -> torch.Tensor:
    # for validation only: freeze the input scene intensity (make this step
    # stupid but for fair comparison...)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    img_stack2 = np.zeros((res, res, len(filename_list)))
    for j, fn in enumerate(filename_list):
        img = cv2.imread(fn)[:, :, channel_idx] / 255.
        img_rs = cv2.resize(img, (res, res))
        img_stack2[:, :, j] = img_rs
    interp_val = torch.from_numpy(img_stack2).view(
        res * res, len(filename_list)).to(device)
    return interp_val