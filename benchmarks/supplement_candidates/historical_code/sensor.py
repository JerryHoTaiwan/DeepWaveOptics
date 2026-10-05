import torch
from typing import Tuple, Any, Dict
from utils import get_borders


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


def generate_pixel_full(dim1: int, dim2: int) -> torch.Tensor:
    if torch.cuda.is_available:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    axis_1 = torch.arange(dim1)
    axis_2 = torch.arange(dim2)
    grid_y, grid_x = torch.meshgrid(axis_1, axis_2)
    h_array = (torch.remainder(grid_x, 2) + 2 *
               torch.remainder(grid_y, 2)).to(device) * 5e-7 + 4e-7
    # with full resolution
    h_full = torch.zeros(4, dim1, dim2).to(device)
    h_full[0, :, :] = h_array[0, 0]
    h_full[1, :, :] = h_array[0, 1]
    h_full[2, :, :] = h_array[1, 0]
    h_full[3, :, :] = h_array[1, 1]
    return h_full


def BW_trans(h: torch.Tensor, theta: float, lamb: float) -> torch.Tensor:
    if torch.cuda.is_available:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    A = torch.tensor([0.], device=device)
    R = torch.tensor([0.9], device=device)
    n2 = 1.
    phi = 0.
    tau = torch.pow((1 - (A / (1 - R))), 2)
    F = (4 * R / torch.pow(1 - R, 2))
    cos = torch.cos(theta)[None, ...]
    delta = (4 * torch.pi / lamb) * n2 * h * cos + 2 * phi
    g = tau / (1 + F * torch.pow(torch.sin(delta / 2), 2))
    return g


def get_incident_angle(v01: torch.Tensor) -> torch.Tensor:
    if torch.cuda.is_available:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    dz = torch.tensor([0., 0., 1.], dtype=v01.dtype).to(
        device).view(1, 1, 3, 1)
    cos = torch.sum(v01 * dz, dim=2)
    theta = torch.acos(cos)
    return theta


def create_sensor_with_depth(config: Dict[str,
                                          Any],
                             start_idx: int,
                             end_idx: int,
                             x_dim: int,
                             y_dim: int,
                             focal_pos: float) -> torch.Tensor:
    if torch.cuda.is_available:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if config["dtype"] == "float64":
        datatype = torch.float64
    elif config["dtype"] == "float32":
        datatype = torch.float32
    Img_pos = torch.zeros(
        end_idx - start_idx,
        x_dim,
        3,
        1,
        dtype=datatype).to(device)
    dx_left, dx_right, dy_top, dy_bot = get_borders(config)
    half_pxl_size = config["width"] / config["dim"]
    fx = torch.linspace(
        dx_left +
        half_pxl_size,
        dx_right -
        half_pxl_size,
        x_dim,
        dtype=datatype,
        device=device)
    fy = torch.linspace(
        dy_top - half_pxl_size,
        dy_bot + half_pxl_size,
        y_dim,
        dtype=datatype,
        device=device)[
        start_idx:end_idx]
    # This is correct... Surprisingly
    grid_y, grid_x = torch.meshgrid(fy, fx)
    Img_pos[:, :, 0, 0] = grid_y * 1e-3
    Img_pos[:, :, 1, 0] = grid_x * 1e-3
    Img_pos[:, :, 2, 0] = focal_pos * 1e-3
    return Img_pos


def create_sensor_grids(config: Dict[str, Any], left_border: int, right_border: int,
                        top_border: int, bot_border: int, dim: int) -> torch.Tensor:
    if config["dtype"] == "float64":
        datatype = torch.float64
    elif config["dtype"] == "float32":
        datatype = torch.float32
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    fx = torch.linspace(
        left_border,
        right_border,
        dim,
        dtype=datatype,
        device=device)
    fy = torch.linspace(
        top_border,
        bot_border,
        dim,
        dtype=datatype,
        device=device)
    sensor_pos = torch.zeros(
        1, 2, dim, dim, dtype=datatype).to(device)

    grid_y, grid_x = torch.meshgrid(fy, fx)  # This is correct... Surprisingly
    sensor_pos[0, 0, :, :] = grid_y
    sensor_pos[0, 1, :, :] = grid_x
    return sensor_pos
