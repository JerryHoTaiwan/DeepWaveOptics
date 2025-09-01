from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.checkpoint as ckpt

from utils import create_sensor_with_depth


# -------------------------
# Helpers
# -------------------------

def get_device() -> torch.device:
    """Return CUDA if available, else CPU."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def as_col(t: torch.Tensor, length: int, default_val: float = 0.0, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Ensure tensor is (N,1). If t is None, create a (N,1) tensor filled with default_val.
    """
    device = device or get_device()
    if t is None:
        return torch.full((length, 1), float(default_val), dtype=torch.float32, device=device)
    t = t.to(device)
    if t.ndim == 1:
        t = t.unsqueeze(1)
    if t.shape[0] != length or t.shape[1] != 1:
        raise ValueError(f"Expected shape ({length},1), got {tuple(t.shape)}")
    return t


# -------------------------
# Wavefront container
# -------------------------

class Wavefront_Tensor:
    """
    Lightweight container for wavefront point cloud + per-point attributes.

    info keys:
        valid: Tuple[index, index] or 1D index tensor
        x, y:             (N, 4) float32
        ci:               (N, 3) float32  (point centers)
        mag:              (N, 1) float32
        phase:            (N, 1) float32  (radians)
        path, latest_path:(N, 1) float32
        vec:              (N, 3) float32
        para: dict with 'L','M','N' -> (N, 3) float32
    """

    def __init__(self, dim: int, device: torch.device):
        self.dim: int = int(dim)
        self.device: torch.device = device
        self.info: Dict[str, Any] = {}

    def update(
        self,
        valid: Tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        ci: torch.Tensor,
        mag: torch.Tensor,
        phase: torch.Tensor,
        path: torch.Tensor,
        latest_path: torch.Tensor,
        vec: torch.Tensor,
        para: Dict[str, torch.Tensor],
    ) -> None:
        """Select valid points and stash everything in self.info."""
        # Normalize valid to a single index tensor in first dim.
        if isinstance(valid, tuple):
            valid_idx = valid[0]
        else:
            valid_idx = valid
        self.info["valid"] = (valid_idx, valid_idx)

        self.info["x"] = x[valid_idx, :].to(self.device)
        self.info["y"] = y[valid_idx, :].to(self.device)
        self.info["ci"] = ci[valid_idx, :].to(self.device)
        self.info["mag"] = mag[valid_idx, :].to(self.device)
        self.info["phase"] = phase[valid_idx, :].to(self.device)
        self.info["path"] = path[valid_idx, :].to(self.device)
        self.info["latest_path"] = latest_path[valid_idx, :].to(self.device)
        self.info["vec"] = vec[valid_idx, :].to(self.device)

        self.info["para"] = {
            "L": para["L"][valid_idx, :].to(self.device),
            "M": para["M"][valid_idx, :].to(self.device),
            "N": para["N"][valid_idx, :].to(self.device),
        }

    def save(self, key: str, filename: str | Path) -> None:
        with Path(filename).open("wb") as fp:
            pickle.dump(self.info[key], fp)

    def load(self, key: str, filename: str | Path) -> None:
        with Path(filename).open("rb") as fp:
            self.info[key] = pickle.load(fp)

    def __len__(self) -> int:
        return self.dim


# -------------------------
# Construction
# -------------------------

def create_wf(
    pts: torch.Tensor,
    phase: Optional[torch.Tensor] = None,
    mag: Optional[torch.Tensor] = None,
    uni_mag: bool = True,
    uni_phase: bool = True,
) -> Wavefront_Tensor:
    """
    Create an initial Wavefront_Tensor from points.

    Args:
        pts:   (N,3) point positions (meters or consistent unit; later converted)
        phase: optional (N,) or (N,1) radians
        mag:   optional (N,) or (N,1) amplitudes
        uni_mag/uni_phase: if True, use ones/zeros respectively

    Returns:
        Wavefront_Tensor with fields initialized.
    """
    device = get_device()
    pts = pts.to(device)
    N = int(pts.shape[0])

    valid_idx = torch.arange(N, device=device)

    wf = Wavefront_Tensor(dim=N, device=device)

    x = torch.zeros((N, 4), dtype=torch.float32, device=device)
    y = torch.zeros((N, 4), dtype=torch.float32, device=device)
    center = pts.clone()

    # amplitude & phase as (N,1)
    ml = torch.ones((N, 1), dtype=torch.float32, device=device) if uni_mag else as_col(mag, N, 1.0, device)
    pl = torch.zeros((N, 1), dtype=torch.float32, device=device) if uni_phase else as_col(phase, N, 0.0, device)

    s = torch.zeros((N, 1), dtype=torch.float32, device=device)  # path length accumulator
    vec = torch.zeros((N, 3), dtype=torch.float32, device=device)
    L = torch.zeros((N, 3), dtype=torch.float32, device=device)
    M = torch.zeros((N, 3), dtype=torch.float32, device=device)
    Np = torch.zeros((N, 3), dtype=torch.float32, device=device)

    wf.update(
        valid=(valid_idx, valid_idx),
        x=x,
        y=y,
        ci=center,
        mag=ml,
        phase=pl,
        path=s,
        latest_path=s,
        vec=vec,
        para={"L": L, "M": M, "N": Np},
    )
    return wf


# -------------------------
# Visualization
# -------------------------

def display_wf(wf: Wavefront_Tensor, psf_idx: int, config: Dict[str, Any], show_3d: bool = False) -> None:
    """
    Scatter-plot the wavefront phase over (y,x) coordinates, with wrapped/unwrapped variants.
    """
    outdir = Path(config["display_folder"])
    outdir.mkdir(parents=True, exist_ok=True)

    ci = wf.info["ci"].detach().cpu().numpy()   # (N,3)
    ph = wf.info["phase"].detach().cpu().numpy().squeeze(-1)  # (N,)
    y = ci[:, 0]
    x = ci[:, 1]

    if show_3d:
        ax = plt.axes(projection="3d")
        ax.scatter3D(x, y, ci[:, 2], c=np.sin(ph), cmap="bwr", s=3)
        ax.view_init(elev=50.0, azim=30.0, roll=-90)
        plt.savefig(outdir / f"wf_normal_3d_{psf_idx}.png", bbox_inches="tight")
        plt.close()

    # Unwrapped
    plt.scatter(x=x, y=y, c=ph, cmap="bwr", s=3)
    plt.colorbar()
    plt.savefig(outdir / f"wf_normal2_{psf_idx}.png", bbox_inches="tight")
    plt.close()

    # Wrapped to [-pi, pi)
    wrapped = (ph + np.pi) % (2 * np.pi) - np.pi
    plt.scatter(x=x, y=y, c=wrapped, cmap="bwr", s=3)
    plt.colorbar()
    plt.savefig(outdir / f"wf_normal2_{psf_idx}_wrapped.png", bbox_inches="tight")
    plt.close()


# -------------------------
# Huygens propagation (chunked)
# -------------------------

def gen_aperture(
    wf: Wavefront_Tensor,
    sphere_cent: torch.Tensor,            # (unused here but kept for compatibility)
    focal_pos: float | torch.Tensor,
    wavelength: float,                    # in nm
    config: Dict[str, Any],
    figname: str = "noname",
    adj_pxl: bool = False,
    land_pos: Optional[torch.Tensor] = None,
    psf_idx: int = -1,
) -> torch.Tensor:
    """
    Generate complex field U on the sensor using a batched Huygens sum.

    Returns:
        U: (H,W) complex64 (or (4,H,W) if use_color_filter)
    """
    device = get_device()

    # Resolve grid size
    if adj_pxl:
        if land_pos is None:
            raise ValueError("adj_pxl=True requires land_pos tensor.")
        y_dim = land_pos.size(2)
        x_dim = land_pos.size(3)
    else:
        y_dim = int(config["dim"])
        x_dim = int(config["dim"])

    batch_size = int(config["batch_size"])

    ml = wf.info["mag"].to(device)      # (N,1)
    pl = wf.info["phase"].to(device)    # (N,1)
    center = wf.info["ci"].to(device)   # (N,3)

    if config.get("use_color_filter", False):
        U = torch.zeros(4, y_dim, x_dim, dtype=torch.complex64, device=device)
    else:
        U = torch.zeros(y_dim, x_dim, dtype=torch.complex64, device=device)

    # number of row-chunks
    batch_cnt = (y_dim + batch_size - 1) // batch_size

    focal_pos_t = torch.as_tensor(focal_pos, dtype=torch.float32, device=device)

    for i in range(batch_cnt):
        start_idx = i * batch_size
        end_idx = min(y_dim, start_idx + batch_size)

        U_patch = ckpt.checkpoint(
            Huygens_batch,
            config,
            y_dim,
            x_dim,
            ml,
            pl,
            center,
            start_idx,
            end_idx,
            focal_pos_t,
            float(wavelength),
            adj_pxl,
            land_pos if land_pos is not None else torch.tensor(0.0, device=device),
            use_reentrant=False,  # PyTorch 2.x friendly; remove if older
        )

        if config.get("use_color_filter", False):
            U[:, start_idx:end_idx] = U_patch
        else:
            U[start_idx:end_idx] = U_patch

    return U


def Huygens_batch(
    config: Dict[str, Any],
    y_dim: int,
    x_dim: int,
    ml: torch.Tensor,                   # (N,1) float32
    pl: torch.Tensor,                   # (N,1) float32 (radians)
    center: torch.Tensor,               # (N,3) float32 (meters or mm; converted below)
    start_idx: int,
    end_idx: int,
    focal_pos: torch.Tensor,            # float32 (mm)
    wavelength: float,                  # nm
    adj_pxl: bool = False,
    land_pos: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    """
    Compute Huygens integral for a row-chunk [start_idx:end_idx].
    Returns:
        U_P1: (chunk_width, x_dim) complex64 or (4, chunk_width, x_dim) if use_color_filter
    """
    device = ml.device
    f32 = torch.float32
    cpx = torch.complex64

    # Wavenumber in air: k = 2*pi / lambda
    lam_m = torch.tensor(wavelength, dtype=f32, device=device) * 1e-9
    k_air = (2.0 * torch.pi) / lam_m  # float32
    scalar = torch.tensor(1.0, dtype=cpx, device=device) / (1j * lam_m.to(cpx))  # complex64

    # Sensor grid positions
    if adj_pxl:
        # land_pos expected shape: (1, 2, H, W) in millimeters
        yl = land_pos.size(-1)
        Img_pos = torch.zeros((end_idx - start_idx, yl, 3, 1), dtype=f32, device=device)
        Img_pos[:, :, 0, 0] = land_pos[0, 0, start_idx:end_idx, :] * 1e-3  # mm -> m
        Img_pos[:, :, 1, 0] = land_pos[0, 1, start_idx:end_idx, :] * 1e-3
        Img_pos[:, :, 2, 0] = focal_pos * 1e-3
    else:
        Img_pos = create_sensor_with_depth(config, start_idx, end_idx, x_dim, y_dim, focal_pos)

    # Wavefront point positions (meters)
    # center: (N,3) in mm? If in mm, convert; if already in m, drop the 1e-3.
    Wf_pos = center.T[None, :, :] * 1e-3          # (1,3,N)
    Wf_pos_expand = center.T[None, None, :, :] * 1e-3  # (1,1,3,N)

    # Approximate contributing area using distance between first two samples (fallback if N<2)
    if Wf_pos.shape[-1] >= 2:
        rad_wf_pos = torch.norm(Wf_pos[0, :, 1] - Wf_pos[0, :, 0]) / 2.0
        approx_area = (rad_wf_pos ** 2) * torch.pi
    else:
        approx_area = torch.tensor(1.0, dtype=f32, device=device)
    if not torch.isfinite(approx_area) or approx_area <= 0:
        approx_area = torch.tensor(1.0, dtype=f32, device=device)

    # Complex source weights
    # ml/pl are (N,1) -> squeeze to (N,)
    U_P0 = (ml.squeeze(-1).to(cpx) * torch.exp(1j * pl.squeeze(-1).to(cpx)))[None, None, :]  # (1,1,N)

    # Huygens sum
    v01 = Img_pos - Wf_pos_expand                          # (H,W,3,1) - (1,1,3,N) -> (H,W,3,N) by broadcast
    r01 = torch.norm(v01, dim=2)                           # (H,W,N)
    raw_phase = torch.exp(1j * k_air.to(cpx) * r01.to(cpx)) / r01.to(cpx)  # (H,W,N)
    U_P1_P0 = U_P0 * raw_phase                             # (H,W,N)
    U_P1 = torch.sum(U_P1_P0, dim=2) * scalar / Wf_pos.size(-1)  # (H,W)

    return (U_P1 * approx_area.to(cpx)).to(cpx)