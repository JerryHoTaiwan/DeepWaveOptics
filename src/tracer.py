from __future__ import annotations

import math
import sys
from typing import Any, Dict, List, Tuple, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

from waveoptics import gen_aperture, create_wf, display_wf
from utils import (
    create_sensor_grids,
    get_borders,
    interpolation_symm,
    point2line_distance,
    hit_sphere_parallel,
    normalize_vector_torch,
    get_device,
    resolve_dtype,
)

# Keep compatibility with your existing project layout if needed
sys.path.append("..")
import deeplens.optics as deepoptics  # noqa: E402
import diffoptics as do               # noqa: E402
from deeplens import GeoLens          # noqa: E402


# =========================
# Visualization helpers
# =========================

def plot_overlay_sensor(
    ps_back: torch.Tensor,
    Img_pos_all: torch.Tensor,
    proj_back_chief: torch.Tensor,
    prefix: str = ""
) -> None:
    """
    Plot landing positions vs. pixel locations and sparse samples of the back-projection.

    Args:
        ps_back:           (N,3) points on sensor (e.g., traced back)
        Img_pos_all:       (2,H,W) sensor grid, channel 0=y, 1=x (in mm)
        proj_back_chief:   (H*W,3) positions projected back toward the object side
        prefix:            optional filename prefix
    """
    # Scatter all back-traced points
    plt.scatter(ps_back[:, 1].detach().cpu().numpy(),
                ps_back[:, 0].detach().cpu().numpy(), s=4, color="blue")
    plt.scatter(Img_pos_all[1].detach().cpu().numpy().reshape(-1),
                Img_pos_all[0].detach().cpu().numpy().reshape(-1), s=4, color="red")
    plt.savefig(f"{prefix}pos_cmp_normal.png", bbox_inches="tight")
    plt.close()

    # Dense back-projection
    plt.scatter(proj_back_chief[:, 1].detach().cpu().numpy(),
                proj_back_chief[:, 0].detach().cpu().numpy(), s=4)
    plt.savefig(f"{prefix}proj_back_chief_normal.png", bbox_inches="tight")
    plt.close()

    # Sparse sub-sampling for readability (every ~32 pixels if square)
    N = proj_back_chief.shape[0]
    s = int(round(math.sqrt(N)))
    if s * s == N:
        grid = proj_back_chief.view(s, s, 3)
        step = max(1, s // 16)  # ~16x16 samples
        plt.scatter(grid[0::step, 0::step, 1].detach().cpu().numpy().reshape(-1),
                    grid[0::step, 0::step, 0].detach().cpu().numpy().reshape(-1), s=10)
        plt.savefig(f"{prefix}proj_back_chief_normal_sparse.png", bbox_inches="tight")
        plt.close()


# =========================
# Pupil / ray sampling
# =========================

def get_exit_pupil(lens: do.Lensgroup, wavelength: float, rad: float, stop: int = 3) -> torch.Tensor:
    """
    Estimate exit pupil axial position from the last segment of the chief ray.

    Returns:
        z-position (scalar tensor) of the exit pupil.
    """
    ray_scatter = lens.sample_ray_common_o(rad, wavelength, oy=0.01, M=5000, mode="get_chief")
    _ps, oss, _v = lens.trace_to_sensor_r(ray_scatter)
    pts_stop = oss[:, stop, :]  # points at the stop surface

    # Chief ray ≈ closest to stop center
    rad_to_center = torch.sqrt(pts_stop[:, 0] ** 2 + pts_stop[:, 1] ** 2)
    chief_idx = torch.argmin(rad_to_center)

    # Last segment of chief ray, intersect y=0 plane
    last_start = oss[chief_idx, -2, :]
    last_end = oss[chief_idx, -1, :]
    last_vec = last_end - last_start
    y_dir = last_vec[0]
    y_diff = -last_start[0]
    t_xp = y_diff / (y_dir + 1e-12)
    xp_pos = last_start + t_xp * last_vec
    return xp_pos[2]


def sample_ray_common_o(
    R: float,
    wavelength: float,
    ox: float = 0.0,
    oy: float = 0.0,
    dist_z: float = -12.0,
    M: int = 15,
    mode: str = "none",
    z_in_sample: float = 1.0,
    datatype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> deepoptics.Ray:
    """
    Generate rays from a plane at z=dist_z toward a target set.
    Modes:
        - "get_chief": 1D line along y across [-R, R], odd count, returns N=2*M-1
        - "grid":      MxM grid across [-R, R]x[-R, R] at z=(dist_z+z_in_sample)
        - "circ":      roughly circular sampling with 6*i+1 points per ring

    Returns:
        deepoptics.Ray with origins `o` and directions `d` (unit), wavelength in microns.
    """
    device = device or get_device()

    if mode == "get_chief":
        N = 2 * M - 1
        o = torch.zeros((N, 3), device=device, dtype=datatype)
    elif mode == "circ":
        N = 3 * M * M - 3 * M + 1
        o = torch.zeros((N, 3), device=device, dtype=datatype)
    else:
        N = M * M
        o = torch.zeros((N, 3), device=device, dtype=datatype)

    o[:, 0] = oy
    o[:, 1] = ox
    o[:, 2] = dist_z

    if mode == "get_chief":
        tgt_y = torch.linspace(-R, R, N, dtype=datatype, device=device)
        tgt_pos = torch.stack((tgt_y, torch.zeros_like(tgt_y), torch.zeros_like(tgt_y)), dim=-1)  # (N,3)
        d = tgt_pos - o
    elif mode == "grid":
        y, x = torch.meshgrid(
            torch.linspace(-R, R, M, dtype=datatype, device=device),
            torch.linspace(-R, R, M, dtype=datatype, device=device),
            indexing="ij",
        )
        z = torch.full_like(x, fill_value=z_in_sample) + dist_z
        pos = torch.stack((y, x, z), dim=-1).reshape(-1, 3)
        d = pos - o
    elif mode == "circ":
        r_sample = torch.linspace(0, R, M, device=device, dtype=datatype)
        d = torch.zeros_like(o)
        cnt = 0
        for i, r in enumerate(r_sample):
            num = 6 * i + 1
            phi_sample = torch.tensor([0.0], device=device, dtype=datatype) if i == 0 \
                         else torch.linspace(0, 2 * torch.pi, num, device=device, dtype=datatype)[:-1]
            xp = r * torch.cos(phi_sample)
            yp = r * torch.sin(phi_sample)
            sz = phi_sample.numel()
            d[cnt:cnt + sz, 0] = yp
            d[cnt:cnt + sz, 1] = xp
            cnt += sz
        d = d - torch.zeros_like(d)  # explicit; origins already set
    else:
        # default to grid
        y, x = torch.meshgrid(
            torch.linspace(-R, R, M, dtype=datatype, device=device),
            torch.linspace(-R, R, M, dtype=datatype, device=device),
            indexing="ij",
        )
        z = torch.full_like(x, fill_value=z_in_sample) + dist_z
        pos = torch.stack((y, x, z), dim=-1).reshape(-1, 3)
        d = pos - o

    d = d / (torch.norm(d, dim=1, keepdim=True) + 1e-12)
    wvln_um = wavelength / 1000.0  # microns
    return deepoptics.Ray(o, d, wvln=wvln_um, device=device)


def sample_ray_uniform_rot(
    wavelength: float,
    view: float = 0.0,
    rot: float = 0.0,
    M: int = 15,
    R: Optional[float] = None,
    oy: float = 0.0,
    ox: float = 0.0,
    datatype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> deepoptics.Ray:
    """
    Circular sampling with uniform azimuthal density per ring; directions are uniform tilt (view, rot).
    """
    device = device or get_device()
    assert R is not None, "R (aperture radius) must be provided."

    N = 3 * M * M - 3 * M + 1
    o = torch.zeros((N, 3), device=device, dtype=datatype)

    r_sample = torch.linspace(0, R, M, device=device, dtype=datatype)
    cnt = 0
    cutoff = 800  # clamp azimuthal samples for very large M
    for i, r in enumerate(r_sample):
        num = 6 * i + 1 if i <= cutoff else 6 * cutoff + 1
        phi_sample = torch.tensor([0.0], device=device, dtype=datatype) if i == 0 \
                     else torch.linspace(0, 2 * torch.pi, num, device=device, dtype=datatype)[:-1]
        xp = r * torch.cos(phi_sample)
        yp = r * torch.sin(phi_sample)
        k = phi_sample.numel()
        o[cnt:cnt + k, 0] = yp
        o[cnt:cnt + k, 1] = xp
        cnt += k

    o[:, 0] += oy
    o[:, 1] += ox

    angle = torch.deg2rad(torch.tensor([view], dtype=datatype, device=device))
    phi = torch.deg2rad(torch.tensor([rot], dtype=datatype, device=device))
    d = torch.stack((
        torch.sin(angle) * torch.cos(phi) * torch.ones_like(o[:, 0]),
        torch.sin(angle) * torch.sin(phi) * torch.ones_like(o[:, 0]),
        torch.cos(angle) * torch.ones_like(o[:, 0]),
    ), dim=-1)
    wvln_um = wavelength / 1000.0
    return deepoptics.Ray(o[:cnt, :], d[:cnt, :], wvln=wvln_um, device=device)


# =========================
# OPL utilities (DeepLens-compatible)
# =========================

def init_opl_prl(
    config: Dict[str, Any],
    rays: deepoptics.Ray,
    view: float,
    rot: float,
    depth: float,
) -> deepoptics.Ray:
    """
    Initialize optical path length (OPL) for parallel rays for DeepLens.
    Uses signed distance from a baseline line to estimate first segment.

    Returns:
        rays with updated .opl
    """
    dtype = resolve_dtype(config.get("dtype"))
    device = get_device()

    # OPL should be zeroed on entry
    assert torch.allclose(rays.opl, torch.zeros_like(rays.opl)), "Expected initial rays.opl == 0."

    r_idx_air = 1.0
    angle = torch.deg2rad(torch.tensor([view], dtype=dtype, device=device))
    phi = torch.deg2rad(torch.tensor([rot], dtype=dtype, device=device))

    # Line ax + by + c = 0 through the ray origin, rotated by phi + 90°
    a = torch.tan(phi + torch.pi / 2)
    baseline = torch.tensor([a, -1.0, rays.o[0, 0] - a * rays.o[0, 1]], dtype=dtype, device=device)

    # Correct sign for near-180° cases
    sign = -torch.sign(phi + 1e-8)
    rot_t = rot if isinstance(rot, torch.Tensor) else torch.tensor(rot, dtype=dtype, device=device)

    if (torch.abs(torch.abs(rot_t) - 180) < 1e-2) and (torch.sign(rot_t) == -1):
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
    depth: float,
) -> torch.Tensor:
    """
    Compute OPL through the system (DiffOptics path): air -> materials[i] -> ... -> exit sphere.
    """
    dtype = resolve_dtype(config.get("dtype"))
    device = get_device()

    opl = torch.zeros((oss.shape[0],), dtype=dtype, device=device)
    r_idx_air = lens.materials[0].ior(wavelength)

    # Use baseline method for first air segment (stable for far field)
    angle = torch.deg2rad(torch.tensor([view], dtype=dtype, device=device))
    phi = torch.deg2rad(torch.tensor([rot], dtype=dtype, device=device))
    a = torch.tan(phi + torch.pi / 2)
    baseline = torch.tensor([a, -1.0, oss[0, 0, 0] - a * oss[0, 0, 1]], dtype=dtype, device=device)
    sign = -torch.sign(phi + 1e-8)
    rot_t = rot if isinstance(rot, torch.Tensor) else torch.tensor(rot, dtype=dtype, device=device)

    if (torch.abs(torch.abs(rot_t) - 180) < 1e-2) and (torch.sign(rot_t) == -1):
        sign *= -1
    first_dist = point2line_distance(oss[:, 0, :], baseline) * torch.sin(angle) * sign
    opl += first_dist * r_idx_air

    # Material segments
    for i in range(len(lens.materials)):
        diff = oss[:, i + 1, :] - oss[:, i, :]
        dist = torch.norm(diff, dim=1)
        r_idx = lens.materials[i].ior(wavelength)
        opl += dist * r_idx

    # Subtract exit-air segment to the exit sphere
    last_diff = pt_at_sphere - oss[:, -1, :]
    last_dist = torch.norm(last_diff, dim=1)
    opl -= last_dist * r_idx_air
    return opl


# =========================
# Chief-ray mapping & tracing
# =========================

def send_chief_ray(
    config: Dict[str, Any],
    lens: do.Lensgroup | GeoLens,
    wavelength: float = 440.0,
    dim: int = 255,
    layer: int = 400,
    plot: bool = False,
):
    """
    Build a mapping from sensor pixels to object-side chief rays, by:
      1) tracing a dense grid of rays around the stop to the sensor (back system),
      2) choosing the nearest ray per coarse sensor bin,
      3) interpolating view angles across the full-res grid,
      4) tracing those interpolated directions through the front system.
    """
    device = get_device()
    dtype = resolve_dtype(config.get("dtype"))

    dx_left, dx_right, dy_top, dy_bot = get_borders(config)
    stop = config["stop_ind"]

    # Split system around the stop index
    if config["use_deeplens"]:
        stop_d = lens.surfaces[stop - 1].d
        materials_front_rev = lens.materials[:stop]
        surfaces_front_rev = lens.surfaces[:stop - 1]
        materials_back = lens.materials[stop - 1:]
        surfaces_back = lens.surfaces[stop - 1:]

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
            datatype=dtype,
            device=device,
        )
        # ensure rays are just in front of the aperture
        ray_stop.o[:, 2] -= 1e-6
        ori_stop = ray_stop.o.clone()
        dir_stop = ray_stop.d.clone()

        ray_back = back_system.trace2sensor(ray_stop)
        ps_back = ray_back.o
    else:
        stop_d = lens.surfaces[stop - 1].d
        materials_front_rev = lens.materials[:stop]
        surfaces_front_rev = lens.surfaces[:stop - 1]
        materials_back = lens.materials[stop - 1:]
        surfaces_back = lens.surfaces[stop - 1:]

        front_system_rev = do.Lensgroup(device=device)
        back_system = do.Lensgroup(device=device)
        front_system_rev.load(surfaces_front_rev, materials_front_rev)
        front_system_rev.d_sensor = -1e-6
        back_system.load(surfaces_back, materials_back)
        back_system.d_sensor = lens.d_sensor

        ray_stop = back_system.sample_ray_common_o(
            R=4 * config["system_scale"],
            wavelength=wavelength,
            M=layer,
            dist_z=stop_d,
            mode="grid",
            z_in_sample=config["z_in_sample"],
        )
        ori_stop = ray_stop.o.clone()
        dir_stop = ray_stop.d.clone()
        ps_back, _oss_back, _valid = back_system.trace_to_sensor_r(ray_stop)

    # Build coarse sensor bins and choose nearest traced ray per bin
    sub_sp = int((config["dim"] - 1) / config["chief_gap"]) + 1
    Img_pos = create_sensor_grids(config, dx_left, dx_right, dy_top, dy_bot, sub_sp)  # (1,2,Hs,Ws)
    xy_pos = ps_back[:, :2][:, :, None, None]  # (N,2,1,1)

    if Img_pos.shape[2] > 16:
        batch_size = 4
        Hs = Img_pos.shape[2]
        chief_idx = torch.zeros((Hs, Img_pos.shape[3]), device=device, dtype=torch.long)
        for i in range(0, Hs, batch_size):
            end_i = min(Hs, i + batch_size)
            diff_tmp = torch.norm(Img_pos[:, :, i:end_i, :] - xy_pos, dim=1)
            chief_idx[i:end_i] = torch.argmin(diff_tmp, dim=0)
    else:
        diff = torch.norm(Img_pos - xy_pos, dim=1)
        chief_idx = torch.argmin(diff, dim=0)

    # Quick visualization of all traced points vs. coarse grid
    plt.scatter(xy_pos[:, 0, 0, 0].detach().cpu().numpy(),
                xy_pos[:, 1, 0, 0].detach().cpu().numpy(), s=0.1)
    plt.scatter(Img_pos[0, 1].detach().cpu().numpy().reshape(-1),
                Img_pos[0, 0].detach().cpu().numpy().reshape(-1), s=4)
    plt.savefig("chief_ray_all.png", bbox_inches="tight")
    plt.close()

    # Interpolate view angles over full-res grid
    chief_idx_rs = chief_idx.view(sub_sp * sub_sp).to(torch.long)
    Img_pos_all = create_sensor_grids(config, dx_left, dx_right, dy_top, dy_bot, dim)[0, ...]  # (2,H,W)

    dir_tmp = dir_stop[chief_idx_rs, :]  # (Hs*Ws,3)
    rot_approx = torch.flip(
        torch.rot90(torch.rad2deg(torch.atan2(Img_pos_all[0], Img_pos_all[1])), 1),
        dims=(1,),
    )
    rad = torch.sqrt(dir_tmp[:, 0] ** 2 + dir_tmp[:, 1] ** 2 + 1e-15)
    view = torch.rad2deg(torch.atan2(rad + 1e-15, dir_tmp[:, 2] + 1e-15))
    view_ds = view.view(sub_sp, sub_sp)
    view_mts = interpolation_symm(view_ds, shape=(config["dim"], config["dim"]))
    # Keep known samples exact
    view_mts[::config["chief_gap"], ::config["chief_gap"]] = view_ds

    # Convert angles to directions at each full-res pixel
    H = W = config["dim"]
    dir_stop_interp = torch.zeros((H * W, 3), dtype=dtype, device=device)
    rad_interp = torch.sin(torch.deg2rad(view_mts)).reshape(-1)
    dir_stop_interp[:, 2] = torch.cos(torch.deg2rad(view_mts)).reshape(-1)
    dir_stop_interp[:, 1] = rad_interp * torch.sin(torch.deg2rad(rot_approx)).reshape(-1)
    dir_stop_interp[:, 0] = rad_interp * torch.cos(torch.deg2rad(rot_approx)).reshape(-1)

    # Trace front system (invert directions)
    ori_stop_interp = ori_stop[0, :].unsqueeze(0).repeat(H * W, 1)

    if config["use_deeplens"]:
        ray_inv = deepoptics.Ray(ori_stop_interp, -dir_stop_interp, wavelength / 1000.0, device=device)
        ray_front = front_system_rev.trace2sensor(ray_inv, record=False)
        d_first_unit_interp = -ray_front.d.clone()
        o_first_interp = ray_front.o.clone()
    else:
        ray_inv = do.Ray(ori_stop_interp, -dir_stop_interp, wavelength, device=device)
        _ps, oss_front_interp, _valid = front_system_rev.trace_to_sensor_r(ray_inv)
        d_first = oss_front_interp[:, -2, :] - oss_front_interp[:, -1, :]
        d_first_unit_interp = d_first / (torch.norm(d_first, dim=1, keepdim=True) + 1e-12)
        o_first_interp = oss_front_interp[:, -1, :].clone()

    # Project to scene plane at distance dist_scene (negative z direction)
    tz = -float(config["dist_scene"]) / (d_first_unit_interp[:, 2] + 1e-12)
    proj_back_chief = o_first_interp + d_first_unit_interp * tz[:, None]

    if plot:
        plot_overlay_sensor(ps_back, Img_pos_all, proj_back_chief)

    theta_first = torch.acos(torch.clamp(d_first_unit_interp[:, 2], -1.0, 1.0))
    print("FOV (max theta):", torch.rad2deg(torch.max(theta_first)).item())

    # Sanity checks
    assert len(torch.unique(chief_idx)) == sub_sp * sub_sp
    assert torch.isfinite(proj_back_chief).all()

    return proj_back_chief, Img_pos_all, d_first_unit_interp, o_first_interp, torch.zeros((1, 2, H, W), dtype=dtype, device=device)


def trace_all(
    lens: do.Lensgroup,
    R: float,
    wavelength: float,
    config: Dict[str, Any],
    layers: int = 100,
    psf_idx: int = -1,
    view: float = 0.0,
    rot: float = 0.0,
    offset_y: float = 0.0,
    offset_x: float = 0.0,
    depth: float = -1e5,
    adj_pxl: bool = False,
    land_pos: Optional[torch.Tensor] = None,
    use_parallel: bool = True,
    origin: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Main driver: sample rays, trace through system, build wavefront, propagate to sensor, compute irradiance.
    Returns:
        irrad (H,W), U (H,W) complex, ps (N,2) sensor hits (if available), oss (N,S,3) path (if available)
    """
    device = get_device()
    dtype = resolve_dtype(config.get("dtype"))

    rad = config["sample_rad"] / 2.0
    stop_ind = config["stop_ind"]

    # Dynamic sampling (optional; then overridden by config["layers"])
    if layers > 10:
        abs_view = float(abs(view))
        if abs_view > 30:
            layers = 100
        elif abs_view > 18:
            layers = 100
        elif abs_view <= 15:
            layers = 50
        if not config.get("single_lens", False):
            layers = int(config.get("layers", layers))
    layers = int(config.get("layers", layers))

    if config.get("use_deeplens", False):
        xp_pos, _pupilr = lens.exit_pupil()
        if use_parallel:
            traced_ray = sample_ray_uniform_rot(
                wavelength, M=layers, R=R, view=view, rot=rot, oy=offset_y, ox=offset_x, datatype=dtype, device=device
            )
            traced_ray = init_opl_prl(config, traced_ray, view, rot, depth)
        else:
            assert origin is not None, "origin must be provided when use_parallel=False"
            traced_ray = sample_ray_common_o(
                R, wavelength, M=layers, dist_z=float(origin[2]), ox=float(origin[1]), oy=float(origin[0]),
                mode="circ", datatype=dtype, device=device
            )
        traced_ray.coherent = True
        traced_ray = lens.trace2sensor(traced_ray)

        center = traced_ray.o[0]                      # sphere center
        radius = lens.d_sensor - xp_pos               # exit sphere radius
        ps = traced_ray.o
        dir_last = -traced_ray.d
        opl = traced_ray.opl
        _intersect, _t0, _t1, intersect_0, _ = hit_sphere_parallel(ps, dir_last, center, radius)
        last_path = intersect_0 - ps
        last_dist = torch.norm(last_path, dim=1)
        opl = opl - last_dist
        phase = 2 * torch.pi * (opl * 1e6 / wavelength)
        oss = None
    else:
        if "single_psf" in config.get("blocks", []):
            xp_pos = torch.as_tensor([config["xp_pos"]], device=device, dtype=dtype)
        else:
            xp_pos = get_exit_pupil(lens=lens, wavelength=wavelength, rad=rad, stop=stop_ind)

        ray_init = lens.sample_ray_uniform_rot(wavelength, M=layers, R=R, view=view, rot=rot, oy=offset_y, ox=offset_x)
        ps, oss, _v = lens.trace_to_sensor_r(ray_init)

        center = ps[0]
        radius = lens.d_sensor - xp_pos

        ori_from_sensor = oss[:, -1, :]
        ori_before_sensor = oss[:, -2, :]
        dir_last = normalize_vector_torch(ori_before_sensor - ori_from_sensor)
        _intersect, _t0, _t1, intersect_0, _ = hit_sphere_parallel(ori_from_sensor, dir_last, center, radius)

        opl = compute_opl_prl(
            config=config,
            oss=oss,
            lens=lens,
            wavelength=wavelength,
            pt_at_sphere=intersect_0,
            depth=depth,
            view=view,
            rot=rot,
        )
        phase = 2 * torch.pi * (opl * 1e6 / wavelength)

    # Build wavefront and propagate
    wf = create_wf(pts=intersect_0, phase=phase, uni_phase=False)
    focal_pos = lens.d_sensor if torch.is_tensor(lens.d_sensor) else torch.tensor([lens.d_sensor], device=device, dtype=dtype)

    if config.get("plot_wf", False):
        display_wf(wf, psf_idx, config)

    U = gen_aperture(
        wf=wf,
        sphere_cent=center,
        focal_pos=focal_pos,
        wavelength=wavelength,
        config=config,
        adj_pxl=adj_pxl,
        land_pos=land_pos,
        psf_idx=psf_idx,
    )
    irrad = torch.abs(U) ** 2

    return irrad, U, (ps[..., :2] if 'ps' in locals() else torch.empty(0)), (oss if 'oss' in locals() else torch.empty(0))


def place_img_ongrid(
    pos: torch.Tensor,
    obj_max: float,
    act_max: float,
    channel_idx: int = 0,
    res: int = 511,
    filename_list: Optional[List[str]] = None,
    cfg_dtype: Optional[str] = "float32",
) -> torch.Tensor:
    """
    Bilinearly sample an image stack at positions `pos[:, :2]` placed on a square grid.

    Args:
        pos: (N,3) or (N,2) tensor of positions (y, x [, z]) in scene units.
        obj_max: nominal object max radius (ignored, recomputed internally for safety).
        act_max: nominal active radius (ignored, recomputed internally for safety).
        channel_idx: cv2 channel index to extract (BGR order).
        res: grid resolution (res x res).
        filename_list: list of image filepaths. Defaults to ["cameraman.png"].
        cfg_dtype: dtype string ("float32" or "float64").

    Returns:
        (N, K) tensor of interpolated values, where K = len(filename_list).
    """
    if filename_list is None:
        filename_list = ["cameraman.png"]

    device = get_device()
    dtype = resolve_dtype(cfg_dtype)

    # Recompute extents from data (kept for backward compatibility)
    act_max_val = torch.amax(pos[:, :2]).item() * 1.02
    obj_max_val = torch.amax(pos[:, :2]).item()

    pxl_size = (2.0 * act_max_val) / float(res)
    obj_pxls = int(obj_max_val // pxl_size)
    start_idx = int(res // 2 - obj_pxls)
    end_idx = int(res // 2 + obj_pxls) + 1
    side = max(1, min(end_idx - start_idx, res))

    # Load images -> (side, side, K) stack
    img_stack = np.zeros((side, side, len(filename_list)), dtype=np.float32)
    for i, fn in enumerate(filename_list):
        im = cv2.imread(fn, cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(f"Could not read image: {fn}")
        ch = im[:, :, channel_idx].astype(np.float32) / 255.0
        # Preserve orientation: flip then rotate
        img_proc = np.rot90(np.flip(ch, axis=1), 3)
        img_rs = cv2.resize(img_proc, (side, side), interpolation=cv2.INTER_AREA)
        img_stack[:, :, i] = img_rs

    # Check that all positions lie inside
    max_pos = torch.amax(pos[:, :2])
    if not (max_pos < act_max_val):
        print("Scene boundary exceeded:", max_pos.item(), act_max_val)
    assert max_pos < act_max_val

    # Place stack into full grid
    grids = torch.zeros((res, res, len(filename_list)), dtype=dtype, device=device)
    s0, e0 = max(0, start_idx), min(res, end_idx)
    sub_y0 = max(0, -start_idx)
    sub_y1 = sub_y0 + (e0 - s0)
    grids[s0:e0, s0:e0, :] = torch.from_numpy(img_stack[sub_y0:sub_y1, sub_y0:sub_y1, :]).to(
        device=device, dtype=dtype
    )

    # Position to grid coordinates
    center_idx = (res - 1) / 2.0
    pos_on_grid = pos[:, :2].to(dtype=torch.float32, device=device) / pxl_size + center_idx
    y, x = pos_on_grid[:, 0], pos_on_grid[:, 1]

    # Bilinear interpolation
    y0 = torch.clamp(torch.floor(y).to(torch.long), 0, res - 1)
    x0 = torch.clamp(torch.floor(x).to(torch.long), 0, res - 1)
    y1 = torch.clamp(y0 + 1, 0, res - 1)
    x1 = torch.clamp(x0 + 1, 0, res - 1)

    wy = (y - y0.to(y.dtype)).unsqueeze(1)
    wx = (x - x0.to(x.dtype)).unsqueeze(1)

    lb = grids[y0, x0]
    rb = grids[y0, x1]
    lu = grids[y1, x0]
    ru = grids[y1, x1]

    b0 = lb * (1 - wx) + rb * wx
    b1 = lu * (1 - wx) + ru * wx
    interp_val = b0 * (1 - wy) + b1 * wy
    return interp_val


def place_img_on_grid_exact(
    res: int,
    channel_idx: int,
    filename_list: List[str],
    cfg_dtype: Optional[str] = "float32",
) -> torch.Tensor:
    """
    Validation helper: resize each image to (res,res) and flatten.

    Args:
        res: output side length.
        channel_idx: cv2 channel index to extract (BGR order).
        filename_list: list of image file paths.
        cfg_dtype: dtype string ("float32" or "float64").

    Returns:
        (res*res, K) tensor in [0,1].
    """
    device = get_device()
    dtype = resolve_dtype(cfg_dtype)

    stack = np.zeros((res, res, len(filename_list)), dtype=np.float32)
    for j, fn in enumerate(filename_list):
        im = cv2.imread(fn, cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(f"Could not read image: {fn}")
        ch = im[:, :, channel_idx].astype(np.float32) / 255.0
        img_rs = cv2.resize(ch, (res, res), interpolation=cv2.INTER_AREA)
        stack[:, :, j] = img_rs

    out = torch.from_numpy(stack).to(device=device, dtype=dtype)
    return out.view(res * res, len(filename_list))