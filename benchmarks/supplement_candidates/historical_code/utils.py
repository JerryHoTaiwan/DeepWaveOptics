from typing import Any, Dict, Tuple
import os
import torch
import pickle


def point2line_distance(point: torch.Tensor, line: torch.Tensor) -> torch.Tensor:
    # point: Nx3
    # line: ax+by+c=0, 1x3
    a = line[0]
    b = line[1]
    c = line[2]
    x = point[:, 0]
    y = point[:, 1]
    d = torch.abs(a * x + b * y + c) / torch.sqrt(a * a + b * b)
    sign = torch.sign(a * x + b * y + c)
    return d * sign


def build_folder(config: Dict[str, Any]) -> None:
    subfolders = [
        '/lens/',
        '/layout/',
        'layout_off',
        '/meas/',
        'meas_on',
        'meas_on1',
        'meas_off',
        '/meas_full/',
        '/recover_demosaic/',
        '/recover_full/',
        '/gt/',
        '/psf/',
        'pred_on',
        'pred_on1',
        'pred_off']
    if not os.path.exists(config["display_folder"]):
        os.mkdir(config["display_folder"])
    if not os.path.exists(config["record_folder"]):
        os.mkdir(config["record_folder"])
    for sub in subfolders:
        if not os.path.exists(config["record_folder"] + sub):
            os.mkdir(config["record_folder"] + sub)
        if not os.path.exists(config["display_folder"] + sub):
            os.mkdir(config["display_folder"] + sub)
    return


def normalize_vector_torch(vector: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(vector, dim=1)[:, None]
    norm_vector = vector / (norm)
    return norm_vector


def hit_sphere_parallel(origin: torch.Tensor,
                        direction: torch.Tensor,
                        center: torch.Tensor,
                        r: float) -> Tuple[torch.Tensor,
                                           torch.Tensor,
                                           torch.Tensor,
                                           torch.Tensor,
                                           torch.Tensor]:
    # check if there's valid t
    # origin: Nx3
    # direction: Nx3
    oc = origin - center[None, :]  # Nx3
    a = torch.sum(direction * direction, dim=1)[:, None]  # Nx1
    b = 2 * torch.sum(oc * direction, dim=1)[:, None]  # Nx1
    c = torch.sum(oc * oc, dim=1)[:, None] - r * r  # Nx1
    discriminant = b * b - 4 * a * c  # Nx1
    intersect = (discriminant > 10e-8).squeeze()  # N

    # avoid passing Nan in gradient
    disc_valid = torch.clone(discriminant)

    # if yes, check if the intersections lie in in aperture
    t0 = (-b + torch.sqrt(disc_valid)) / (2 * a)
    t1 = (-b - torch.sqrt(disc_valid)) / (2 * a)

    intersect_0 = origin + t0 * direction  # Nx3
    intersect_1 = origin + t1 * direction  # Nx3
    return intersect, t0, t1, intersect_0, intersect_1


def compute_RMS(ps):
    rad = torch.norm(ps, dim=1)
    r_std = torch.std(rad)
    return r_std


def my_interpolation_symm(data_in: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    num = len(data_in.reshape(-1))
    weights = torch.zeros(shape[0], shape[1], num)
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


def get_borders(config: Dict[str, Any]) -> Tuple[int, int, int, int]:
    dx_left = config["center"][1] - config["width"]
    dx_right = config["center"][1] + config["width"]
    dy_top = config["center"][0] + config["width"]
    dy_bot = config["center"][0] - config["width"]
    return dx_left, dx_right, dy_top, dy_bot


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
        
        
def load_lens(filename: str) -> Any:
    with open(filename, 'rb') as pickle_file:
        surf = pickle.load(pickle_file)
    return surf