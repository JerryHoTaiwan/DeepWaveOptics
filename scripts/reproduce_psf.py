#!/usr/bin/env python3
"""Render one recovered PSF case without image data or training dependencies."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/reproduction/singlet_40deg.json")
    parser.add_argument("--dim", type=int, help="Override sensor samples (smoke tests only)")
    parser.add_argument("--layers", type=int, help="Override radial ray samples (smoke tests only)")
    parser.add_argument("--output", help="Override output folder")
    parser.add_argument("--check", action="store_true", help="Validate files and print config without importing torch")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = json.loads(config_path.read_text())
    lens_path = ROOT / config["lens_txt"]
    if not lens_path.is_file():
        raise FileNotFoundError(lens_path)
    for key in ("dim", "layers"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if config["dim"] < 2 or config["layers"] < 2:
        raise ValueError("dim and layers must be at least 2")
    if args.check:
        print(json.dumps(config, indent=2))
        return

    import numpy as np
    import torch
    from tracer import trace_all
    from utils import create_sensor_grids, get_device
    import diffoptics as do

    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    device = get_device()
    lens = do.Lensgroup(device=device)
    lens.load_file(str(lens_path))
    if "sensor_z_mm" in config:
        lens.d_sensor = torch.as_tensor(config["sensor_z_mm"], device=device)
    cy, cx = config["center"]  # internal order is y, x
    width = config["width"]  # half-width in mm
    grid = create_sensor_grids(config, cx-width, cx+width, cy+width, cy-width, config["dim"])
    with torch.no_grad():
        intensity, _, hits, _ = trace_all(
            lens, config["sample_rad"], config["disp_wv"], config,
            view=config["view"], rot=config["rot"],
            offset_y=config["offset_y"], offset_x=config["offset_x"],
            adj_pxl=True, land_pos=grid,
        )
    if not torch.isfinite(intensity).all() or intensity.sum() <= 0:
        raise RuntimeError("PSF contains invalid values or has zero energy")
    psf = (intensity / intensity.sum()).cpu().numpy()
    out = Path(args.output) if args.output else ROOT / config["output_folder"]
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "psf.npy", psf)
    metadata = dict(config)
    metadata.update(sensor_z_mm=float(lens.d_sensor),
                    sensor_spacing_mm=2*width/(config["dim"]-1),
                    traced_ray_count=int(hits.shape[0]),
                    device=str(device), torch_version=torch.__version__,
                    normalization="unit sum", smoke_test=bool(args.dim or args.layers))
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
