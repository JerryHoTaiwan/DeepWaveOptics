import json
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, Sequence
import os
import pickle
import torch


# =========================
# Device / dtype utilities
# =========================

def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def resolve_dtype(cfg_dtype: Optional[str], default: torch.dtype = torch.float32) -> torch.dtype:
    """Map a config dtype string to a torch dtype."""
    if cfg_dtype is None:
        return default
    if cfg_dtype.lower() == "float64":
        return torch.float64
    if cfg_dtype.lower() == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype string: {cfg_dtype}")


def normalize_vector_torch(vector: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(vector, dim=1)[:, None]
    norm_vector = vector / (norm)
    return norm_vector


def hit_sphere_parallel(
    origin: torch.Tensor,
    direction: torch.Tensor,
    center: torch.Tensor,
    r: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized ray-sphere intersection.

    Args:
        origin:    (N,3) ray origins
        direction: (N,3) ray directions (need not be normalized)
        center:    (3,)  sphere center
        r:         radius (scalar)

    Returns:
        intersect:  (N,) bool mask whether discriminant > 0
        t0, t1:     (N,1) parametric distances (may be NaN for non-intersections)
        p0, p1:     (N,3) intersection points (may be NaN for non-intersections)
    """
    if origin.shape != direction.shape or origin.size(-1) != 3:
        raise ValueError("origin and direction must both be (N,3)")

    oc = origin - center.view(1, 3)          # (N,3)
    a = torch.sum(direction * direction, dim=1, keepdim=True)     # (N,1)
    b = 2.0 * torch.sum(oc * direction, dim=1, keepdim=True)      # (N,1)
    c = torch.sum(oc * oc, dim=1, keepdim=True) - (r * r)         # (N,1)

    discriminant = b * b - 4.0 * a * c                            # (N,1)
    intersect = (discriminant > 1e-8).squeeze(1)                  # (N,)

    # Clamp to avoid NaNs in sqrt; we’ll mask later anyway.
    disc_clamped = torch.clamp(discriminant, min=0.0)
    sqrt_disc = torch.sqrt(disc_clamped)

    denom = 2.0 * a
    # To avoid /0, where a==0 set outputs to NaN; mask will handle
    safe_denom = denom != 0
    t0 = torch.full_like(denom, float("nan"))
    t1 = torch.full_like(denom, float("nan"))
    t0[safe_denom] = (-b[safe_denom] + sqrt_disc[safe_denom]) / denom[safe_denom]
    t1[safe_denom] = (-b[safe_denom] - sqrt_disc[safe_denom]) / denom[safe_denom]

    p0 = origin + t0 * direction
    p1 = origin + t1 * direction
    return intersect, t0, t1, p0, p1


def point2line_distance(point: torch.Tensor, line: torch.Tensor) -> torch.Tensor:
    """
    Signed distance from points to a 2D line ax + by + c = 0.

    Args:
        point: (N,3) or (N,2); only x,y are used
        line:  (3,) with [a,b,c]

    Returns:
        (N,) signed distances.
    """
    a, b, c = line[..., 0], line[..., 1], line[..., 2]
    x = point[..., 0]
    y = point[..., 1]
    denom = torch.sqrt(a * a + b * b)
    # Prevent divide-by-zero
    denom = torch.clamp(denom, min=1e-12)
    signed = (a * x + b * y + c) / denom
    return signed


def interpolation_symm(data_in: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    interval = int((shape[0] - 1) / (data_in.size()[0] - 1))
    data_out = torch.zeros(shape, device=data_in.device)

    reg_y = torch.linspace(0, data_in.size()[0] - 1, data_in.size()[0], device=data_in.device) - data_in.size()[0] // 2
    reg_x = torch.linspace(0, data_in.size()[1] - 1, data_in.size()[1], device=data_in.device) - data_in.size()[1] // 2
    reg_grid_y, reg_grid_x = torch.meshgrid(reg_y, reg_x)
    dis_cent = torch.sqrt(reg_grid_y * reg_grid_y + reg_grid_x * reg_grid_x) * interval
    slope = data_in / (dis_cent + 1e-8)
    slope_med = torch.median(slope)

    reg_y_full = torch.linspace(0, shape[0]-1, shape[0], device=data_in.device) - shape[0] // 2
    reg_x_full = torch.linspace(0, shape[1]-1, shape[1], device=data_in.device) - shape[1] // 2
    reg_grid_y_full, reg_grid_x_full = torch.meshgrid(reg_y_full, reg_x_full)
    dis_cent_full = torch.sqrt(reg_grid_y_full * reg_grid_y_full + reg_grid_x_full * reg_grid_x_full)
    data_out = slope_med * dis_cent_full
    return data_out

# =========================
# Sensor / grid generators
# =========================

def create_sensor_with_depth(
    config: Dict[str, Any],
    start_idx: int,
    end_idx: int,
    x_dim: int,
    y_dim: int,
    focal_pos: float
) -> torch.Tensor:
    """
    Create a chunk of sensor positions with depth (meter units), shape (rows, x_dim, 3, 1).

    y spans [start_idx:end_idx), x spans [0:x_dim).
    Coordinates are derived from config center/width/dim with half-pixel offset.
    """
    device = get_device()
    dtype = resolve_dtype(config.get("dtype"), default=torch.float32)

    rows = end_idx - start_idx
    Img_pos = torch.zeros((rows, x_dim, 3, 1), dtype=dtype, device=device)

    dx_left, dx_right, dy_top, dy_bot = get_borders(config)
    half_pxl = float(config["width"]) / float(config["dim"])

    # x from left->right, y from top->bottom (consistent with your original)
    fx = torch.linspace(dx_left + half_pxl, dx_right - half_pxl, x_dim, dtype=dtype, device=device)
    fy_full = torch.linspace(dy_top - half_pxl, dy_bot + half_pxl, y_dim, dtype=dtype, device=device)
    fy = fy_full[start_idx:end_idx]

    # 'ij' indexing: first index is y (rows), second is x (cols)
    grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")

    # Convert mm to m
    Img_pos[:, :, 0, 0] = grid_y * 1e-3
    Img_pos[:, :, 1, 0] = grid_x * 1e-3
    Img_pos[:, :, 2, 0] = float(focal_pos) * 1e-3
    return Img_pos


def create_sensor_grids(
    config: Dict[str, Any],
    left_border: float,
    right_border: float,
    top_border: float,
    bot_border: float,
    dim: int
) -> torch.Tensor:
    """
    Create a full-resolution 2D sensor grid (1,2,H,W) in the same units as borders (likely mm).

    Channel 0 = y, Channel 1 = x.
    """
    dtype = resolve_dtype(config.get("dtype"), default=torch.float32)
    device = get_device()

    fx = torch.linspace(left_border, right_border, dim, dtype=dtype, device=device)
    fy = torch.linspace(top_border, bot_border, dim, dtype=dtype, device=device)

    sensor_pos = torch.zeros((1, 2, dim, dim), dtype=dtype, device=device)
    gy, gx = torch.meshgrid(fy, fx, indexing="ij")
    sensor_pos[0, 0] = gy
    sensor_pos[0, 1] = gx
    return sensor_pos


def get_borders(config: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """
    Borders (in the same units as `center` and `width`, typically mm):
      dx_left, dx_right, dy_top, dy_bot
    """
    cx, cy = float(config["center"][0]), float(config["center"][1])
    w = float(config["width"])
    dx_left = cy - w
    dx_right = cy + w
    dy_top = cx + w
    dy_bot = cx - w
    return dx_left, dx_right, dy_top, dy_bot


def generate_bayer(
        start_idx: int,
        end_idx: int,
        dim: int,
        wavelength: float) -> torch.Tensor:
    # in r g b g order
    if torch.is_tensor(wavelength):
        rgb = wave2rgb(wavelength.item())
    else:
        rgb = wave2rgb(wavelength)
    if torch.cuda.is_available:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    pixels = torch.zeros(4, end_idx - start_idx, dim, device=device)
    pixels[0, :, :] = rgb[0]
    pixels[1, :, :] = rgb[1]
    pixels[2, :, :] = rgb[2]
    pixels[3, :, :] = rgb[1]
    return pixels


def wave2rgb(wave: float) -> Tuple[float, float, float]:
    # This is a port of javascript code from
    # http://stackoverflow.com/a/14917481
    gamma = 0.8
    intensity_max = 1

    if wave < 380:
        red, green, blue = 0, 0, 0
    elif wave < 440:
        red = -(wave - 440) / (440 - 380)
        green, blue = 0, 1
    elif wave < 490:
        red = 0
        green = (wave - 440) / (490 - 440)
        blue = 1
    elif wave < 510:
        red, green = 0, 1
        blue = -(wave - 510) / (510 - 490)
    elif wave < 580:
        red = (wave - 510) / (580 - 510)
        green, blue = 1, 0
    elif wave < 645:
        red = 1
        green = -(wave - 645) / (645 - 580)
        blue = 0
    elif wave <= 780:
        red, green, blue = 1, 0, 0
    else:
        red, green, blue = 0, 0, 0

    # let the intensity fall of near the vision limits
    if wave < 380:
        factor = 0
    elif wave < 420:
        factor = 0.3 + 0.7 * (wave - 380) / (420 - 380)
    elif wave < 700:
        factor = 1
    elif wave <= 780:
        factor = 0.3 + 0.7 * (780 - wave) / (780 - 700)
    else:
        factor = 0

    def f(c):
        if c == 0:
            return 0
        else:
            return intensity_max * pow(c * factor, gamma)

    return f(red), f(green), f(blue)


# =========================
# Misc utilities
# =========================


def compute_RMS(ps: torch.Tensor) -> torch.Tensor:
    """
    RMS of radial distances from the origin for a set of 2D points ps:(N,2).
    Returns a scalar tensor (std of radii).
    """
    if ps.ndim != 2 or ps.size(1) != 2:
        raise ValueError("ps must be (N,2)")
    rad = torch.norm(ps, dim=1)
    return torch.std(rad)


def load_config(path: str) -> Dict[str, Any]:
    """Load a JSON config file into a dictionary."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r") as f:
        return json.load(f)


def build_folder(config: Dict[str, Any]) -> None:
    """
    Create display and record folder trees.

    Uses pathlib with mkdir(parents=True, exist_ok=True). Leading slashes are stripped from
    subfolder names to avoid absolute paths.
    """
    display_root = Path(config["display_folder"])
    record_root = Path(config["record_folder"])

    # Original list cleaned: remove leading slashes and duplicates, keep structure intent.
    subfolders: Sequence[str] = [
        "lens",
        "layout",
        "layout_off",
        "meas",
        "meas_on",
        "meas_on1",
        "meas_off",
        "meas_full",
        "recover_demosaic",
        "recover_full",
        "gt",
        "psf",
        "pred_on",
        "pred_on1",
        "pred_off",
    ]

    display_root.mkdir(parents=True, exist_ok=True)
    record_root.mkdir(parents=True, exist_ok=True)

    for sub in subfolders:
        # Ensure relative
        sub = sub.lstrip("/\\")
        (display_root / sub).mkdir(parents=True, exist_ok=True)
        (record_root / sub).mkdir(parents=True, exist_ok=True)
        
        
def get_data_list(config: Dict[str, Any]) -> Tuple[str, str]:
    train_list_all, valid_list_all = [], []
    for img_id in range(1, config["train_num"]+1):
        img_id_str = str(img_id).zfill(4)
        if os.path.isfile("/mnt2/DIV2K_train_HR/{}.png".format(img_id_str)):
            train_list_all.append("/mnt2/DIV2K_train_HR/{}.png".format(img_id_str))
    for img_id in range(801, 801+config["valid_num"]):
        img_id_str = str(img_id).zfill(4)
        if os.path.isfile("/mnt2/DIV2K_valid_HR/{}.png".format(img_id_str)):
            valid_list_all.append("/mnt2/DIV2K_valid_HR/{}.png".format(img_id_str))
    return train_list_all, valid_list_all


def save_lens(surf, filename: str='surf.pkl') -> None:
    with open(filename, 'wb') as pickle_file:
        pickle.dump(surf, pickle_file)