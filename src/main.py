# main.py
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch
import matplotlib.pyplot as plt

# Local utils
from utils import build_folder, load_config

# Add project root (if needed) before third-party/local imports that live one level up
sys.path.append("..")
from deeplens import GeoLens  # noqa: E402
import diffoptics as do        # noqa: E402


# ---------- Utilities ----------

def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    # cuda seed only if cuda is available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    # (Optional) make runs more reproducible (may slow things down)
    # torch.use_deterministic_algorithms(True, warn_only=True)


def set_torch_runtime() -> None:
    torch.autograd.set_detect_anomaly(True)
    torch.set_printoptions(precision=10)


# ---------- Lens construction ----------

def build_lens(config: Dict[str, Any], device: torch.device):
    """
    Build either a DeepLens GeoLens or a DiffOptics Lensgroup based on config.
    Returns (lens, extras) where 'extras' may include derived parameters used later.
    """
    # Common-derived parameters for DiffOptics branch
    extras: Dict[str, Any] = {}

    if config.get("use_deeplens", False):
        print("Using DeepLens lens model.")
        lens = GeoLens(filename=config["lens_name"])
        return lens, extras

    print("Using DiffOptics lens model.")
    lens = do.Lensgroup(device=device)

    if config.get("load_surface", False):
        print("Loading lens from file:", config["lens_name"])
        from initialization import load_lens  # local import to avoid circulars
        lens = load_lens(config["lens_name"])
        # Make sure d_sensor is a plain tensor, not a leaf with grad history
        if torch.is_tensor(lens.d_sensor):
            lens.d_sensor = lens.d_sensor.detach()

    elif config.get("load_surface_from_txt", False):
        print("Loading lens surfaces from txt:", config["lens_txt"])
        lens.load_file(config["lens_txt"])
        # If you need to keep these around:
        extras["surfaces"] = lens.surfaces
        extras["materials"] = lens.materials

    else:
        print("Using given lens parameters as initialization.")
        from initialization import initialize_lens, initialize_materials  # avoid top-level import

        d_list = [4.96, 2.68, 1.31, 2.73, 3.4]
        system_scale = config["system_scale"]

        if config.get("single_lens", False):
            # “longer” comment retained from your code
            lens.d_sensor = torch.tensor([system_scale * 56.43], device=device)
        else:
            lens.d_sensor = torch.tensor(system_scale * (41.58 - 0.23 + sum(d_list)), device=device)

        surfaces = initialize_lens(config, d_list, device)
        materials = initialize_materials(config)
        lens.load(surfaces, materials)
        extras["surfaces"] = surfaces
        extras["materials"] = materials

    # ---- Rendering parameters (DiffOptics branch only) ----
    wavelength_nm = torch.tensor([config["disp_wv"]], device=device, dtype=torch.float32)  # nm
    width = float(config["width"])
    dim = int(config["dim"])
    psf_rad = float(config["psf_rad"])

    pxl_size = 2 * width / dim
    half_pxl = width / dim
    psf_rad_grid = psf_rad // pxl_size
    dim_new = int(dim + 2 * psf_rad_grid)

    # Fill into the lens object
    lens.pixel_size = (2 * width + 2 * psf_rad - 2 * half_pxl) / (dim_new - 1)
    lens.film_size = [dim_new * 1.4, dim_new * 1.4]

    # Save extras you might need downstream
    extras.update(
        dict(
            wavelength_nm=wavelength_nm,
            pxl_size=pxl_size,
            half_pxl=half_pxl,
            psf_rad_grid=psf_rad_grid,
            dim_new=dim_new,
        )
    )
    return lens, extras


# ---------- Block execution ----------

def load_block_registry(module_name: str = "functions") -> Dict[str, Callable[[Dict[str, Any], Any], None]]:
    """
    Build a registry of callable blocks from a module (e.g., your `functions.py`).
    Only public callables (not starting with '_') are registered.
    """
    mod = importlib.import_module(module_name)
    registry: Dict[str, Callable] = {}
    for name in dir(mod):
        if name.startswith("_"):
            continue
        fn = getattr(mod, name)
        if callable(fn):
            registry[name] = fn
    return registry


def run_blocks(block_names: List[str], registry: Dict[str, Callable], config: Dict[str, Any], lens: Any) -> None:
    for block in block_names:
        if block not in registry:
            raise KeyError(
                f"Block '{block}' not found in registry. "
                f"Available: {', '.join(sorted(registry.keys()))}"
            )
        print(f"[Block] {block}")
        registry[block](config, lens)
        
        

def visualize_lens(config: Dict[str, Any], lens: do.Lensgroup) -> None:
    """
    Visualize and save the 2D lens setup.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary. Must include "display_folder".
        Optional key "use_deeplens" (bool) determines how plotting is handled.
    lens : do.Lensgroup
        Lens group object to visualize.
    """
    if config.get("use_deeplens", False):
        ax, fig = lens.plot_setup2D()
    else:
        ax, fig = lens.plot_setup2D(show=False)

    save_path = Path(config["display_folder"]) / "lens_setup.png"
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"Saved lens setup visualization to {save_path}")


# ---------- Entry point ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    args = parser.parse_args()

    config = load_config(args.config)
    build_folder(config)  # create output folders early

    device = get_device()
    print(f"Using device: {device}")

    set_torch_runtime()
    set_seed(int(config["seed"]))

    lens, _extras = build_lens(config, device)
    visualize_lens(config, lens)

    # Blocks to run (ordered)
    blocks = list(config.get("blocks", []))
    if not blocks:
        print("No blocks specified in config['blocks']; nothing to do.")
        return

    # Build the block registry once
    registry = load_block_registry("functions")
    run_blocks(blocks, registry, config, lens)


if __name__ == "__main__":
    main()