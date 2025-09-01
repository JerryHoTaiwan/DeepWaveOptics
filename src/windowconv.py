import torch
import math
from typing import Any, Dict, Tuple, List
import torch.nn as nn
import torch.nn.functional as F
from utils import get_borders


def tri_func_2d(interval: int) -> torch.Tensor:
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    idx = torch.arange(0, 2 * (interval + 1) + 1, dtype=torch.float32, device=device)
    cent = interval + 1
    tri = (1 - torch.abs(idx - cent) / (interval + 1)).view(2 * interval + 3, 1)
    tri_dual = tri @ tri.T
    return tri_dual


def fill_multi(c: torch.Tensor, v: torch.Tensor, ref_idx: int) -> Tuple[torch.Tensor, torch.Tensor, Any, Any, Any, Any]:
    N = c.size()[0]
    max_sft = int(torch.max(torch.abs(c)).item())
    fill_cx = torch.linspace(-max_sft, max_sft, 2 * max_sft + 1, dtype=torch.float).to(c.device)
    fill_cy = torch.linspace(-max_sft, max_sft, 2 * max_sft + 1, dtype=torch.float).to(c.device)
    fill_c_y_grid, fill_c_x_grid = torch.meshgrid(fill_cy, fill_cx)

    fill_v = torch.zeros_like(fill_c_x_grid)[..., None].repeat(1, 1, v.size()[-1])
    fill_c = torch.zeros(fill_cy.size()[0], fill_cx.size()[0], 2).to(c.device)
    fill_c[:, :, 0] = fill_c_y_grid
    fill_c[:, :, 1] = fill_c_x_grid
    ori_idx_new_y = []
    ori_idx_new_x = []
    ref_idx_new_y = []
    ref_idx_new_x = []

    idx_tensor = (c + max_sft).detach().to(torch.int64)
    ori_idx_new_y = idx_tensor[:, :, 0].view(-1).tolist()
    ori_idx_new_x = idx_tensor[:, :, 1].view(-1).tolist()
    ref_idx_new_y = (idx_tensor[:, :, 0].view(-1)[ref_idx]).tolist()
    ref_idx_new_x = (idx_tensor[:, :, 1].view(-1)[ref_idx]).tolist()
    fill_v[idx_tensor[:, :, 0], idx_tensor[:, :, 1], :] = v
    return fill_c, fill_v, ori_idx_new_y, ori_idx_new_x, ref_idx_new_y, ref_idx_new_x


def register(h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    v_dim = c.size()[0]
    meas_dim = h.size()[0]
    max_sft = int(torch.max(torch.abs(c)).item())
    h_3d = h.view(meas_dim, meas_dim, v_dim * v_dim)
    h_reg2_3d = torch.zeros(h.size()[0] + 2 * max_sft, h.size()[1] + 2 * max_sft, v_dim * v_dim, dtype=torch.float, device=c.device) # pad
    ar_ni = torch.arange(0, meas_dim)[:, None].to(c.device)
    start_idx_y2 = (max_sft - c[:, :, 0]).view(1, -1).to(torch.int64).to(c.device)
    start_idx_x2 = (max_sft - c[:, :, 1]).view(1, -1).to(torch.int64).to(c.device)
    idx_y2 = (start_idx_y2 + ar_ni)[:, None, :].repeat(1, meas_dim, 1)
    idx_x2 = (start_idx_x2 + ar_ni)[None, :, :].repeat(meas_dim, 1, 1)
    z_idx = torch.arange(v_dim * v_dim)[None, None, :]
    h_reg2_3d[idx_y2, idx_x2, z_idx] = h_3d
    h_reg2 = h_reg2_3d.view(h.size()[0] + 2 * max_sft, h.size()[1] + 2 * max_sft, v_dim, v_dim)
    return h_reg2


def wind_conv_multi(ref_y: List[float], ref_x: List[float], h_ref_reg: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # check wind_conv_multi4 for the last version
    new_width = h_ref_reg.size()[0]
    v_dim = v.size()[0]
    ref_dim = h_ref_reg.size()[2]
    new_cent = int(new_width // 2)
    half_vlen = int(v_dim // 2)
    new_interval = ref_x[1] - ref_x[0]
    img_num = v.size()[-1]
    device = h_ref_reg.device
    psf_width = 80

    pad_size = (new_width - v_dim) / 2
    v_pad = torch.zeros(new_width, new_width, img_num, dtype=torch.float).to(device)
    v_pad[new_cent-half_vlen:new_cent+half_vlen+1, new_cent-half_vlen:new_cent+half_vlen+1, :] = v #* 0 + 1
    tri_dual = tri_func_2d(new_interval - 1).to(torch.float)
    weights = torch.zeros(new_width, new_width, len(ref_y), dtype=torch.float).to(device)
    
    ar_ni = torch.arange(0, 2 * new_interval + 1)[:, None].to(device)
    start_idx_y2 = (torch.tensor(ref_y) + pad_size - (new_interval)).to(torch.int64)[None, :].to(device)
    start_idx_x2 = (torch.tensor(ref_x) + pad_size - (new_interval)).to(torch.int64)[None, :].to(device)
    rep_dim = tri_dual.size()[0] # actually 2 * interval + 1
    idx_y2 = (start_idx_y2 + ar_ni)[:, None, :].repeat(1, rep_dim, 1)
    idx_x2 = (start_idx_x2 + ar_ni)[None, :, :].repeat(rep_dim, 1, 1)
    z_idx = torch.arange(len(ref_y))[None, None, :]
    ref_y_idx2 = (torch.arange(len(ref_y)) // ref_dim).to(torch.int64).to(device)
    ref_x_idx2 = (torch.arange(len(ref_y)) % ref_dim).to(torch.int64).to(device)

    h_ref_reg_3d = (h_ref_reg[:, :, ref_y_idx2, ref_x_idx2]).to(device)
    weights[idx_y2, idx_x2, z_idx] = tri_dual[:, :, None]
    weights[0:int(pad_size), :, :] = 0.
    weights[:, 0:int(pad_size), :] = 0.
    weights[-int(pad_size):, :, :] = 0.
    weights[:, -int(pad_size):, :] = 0.

    wv = v_pad[:, :, None, :] * weights[:, :, :, None]
    cent = int(wv.size()[0] // 2)
    wv_rs = wv.permute(3, 2, 0, 1)
    filter = h_ref_reg_3d[cent-psf_width:cent+psf_width+1, cent-psf_width:cent+psf_width+1, :].permute(2, 0, 1)[None, ...]
    meas_wd = F.conv2d(wv_rs, filter, padding=psf_width).permute(2, 3, 0, 1)[..., 0]
    return meas_wd, weights, h_ref_reg_3d


def multi_memory_efficient_ATA(config: Dict[str, Any], h_ref_load: torch.Tensor, brightness: torch.Tensor, chief_all: torch.Tensor, chief_ref: torch.Tensor, ref_idx: int) -> torch.Tensor:
    meas_dim = config["dim"]
    num_field = config["dim"]
    interval = config["interval"]
    dx_left, dx_right, dy_top, dy_bot = get_borders(config)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    unit_x = (dx_right - dx_left) / meas_dim
    unit_y = (dy_top - dy_bot) / meas_dim
    ref_dim = int((num_field - 1) // interval + 1)

    # the dimension of chief ray in simulation and demo are opposite, not a bug
    c_x = (torch.sign(chief_all[1, :]) * torch.round(torch.abs(chief_all[1, :]) / unit_x).to(torch.int64)).view(num_field, num_field)
    c_y = (torch.sign(chief_all[0, :]) * torch.round(torch.abs(chief_all[0, :])/ unit_y).to(torch.int64)).view(num_field, num_field)
    c = torch.zeros(num_field, num_field, 2, dtype=torch.float).to(device)
    c[:, :, 0] = c_x
    c[:, :, 1] = c_y
    c_ref_x = (torch.sign(chief_ref[1, :]) * torch.round(torch.abs(chief_ref[1, :]) / unit_x).to(torch.int64)).view(ref_dim, ref_dim)
    c_ref_y = (torch.sign(chief_ref[0, :]) * torch.round(torch.abs(chief_ref[0, :])/ unit_y).to(torch.int64)).view(ref_dim, ref_dim)
    c_ref = torch.zeros(ref_dim, ref_dim, 2, dtype=torch.float).to(device)
    c_ref[:, :, 0] = c_ref_x
    c_ref[:, :, 1] = c_ref_y
    img_num = brightness.size()[-1]
    v = brightness.view(num_field, num_field, img_num).to(torch.float)
    v = torch.rot90(v, 1, dims=[0, 1])
    c, v, _ori_idx_y, _ori_idx_x, ref_idx_y, ref_idx_x = fill_multi(c, v, ref_idx)
    
    v = torch.rot90(v, 3, dims=[0, 1]) # new

    side = int(math.sqrt(h_ref_load.size()[0]))
    if config["pad_sensor"]:
        h_ref = h_ref_load.view(side, side, ref_dim, ref_dim).to(torch.float32)
    else:
        h_ref = h_ref_load.view(meas_dim, meas_dim, ref_dim, ref_dim).to(torch.float32)
    h_ref_rot = torch.rot90(h_ref, 3, dims=[0, 1])
    h_ref_reg = register(h_ref_rot, c_ref)
    new_width = h_ref_reg.size()[0]
    if config["pad_sensor"]:
        diff_size = (new_width - side) / 2
    else:
        diff_size = (new_width - meas_dim) / 2
    meas_wd, _w_load, _h_load = wind_conv_multi(ref_idx_y, ref_idx_x, h_ref_reg, v)
    meas_wd_rot = meas_wd

    meas_wd_rot[:int(diff_size), :, :] = 0.
    meas_wd_rot[-int(diff_size):, :, :] = 0.
    meas_wd_rot[:, :int(diff_size), :] = 0.
    meas_wd_rot[:, -int(diff_size):, :] = 0.
    meas_wd_crop = meas_wd_rot[int(diff_size):-int(diff_size), int(diff_size):-int(diff_size), :]
    return meas_wd_crop
