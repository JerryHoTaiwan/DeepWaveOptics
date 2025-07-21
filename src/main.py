import os
from param import param_map
import argparse
import json
import sys
import time
import torch
from initialization import initialize_lens, initialize_materials, load_lens, plot_lens_with_ray
from tracer import compute_efl
from utils import build_folder
from blocks import * # be aware of duplicate function names
sys.path.append("../")
from deeplens import GeoLens
import diffoptics as do
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--pipeline", type=str, help="The pipeline to run")
args = parser.parse_args()


def show_lens(lens: do.Lensgroup) -> None:
    for i, s in enumerate(lens.surfaces):
        print(f"Surface {i}: {s}")


if __name__ == "__main__":

    filepath = os.path.join(
        "/home/ubuntu/DiffWaveOptics/pipeline", param_map[args.pipeline])
    config = json.load(open(filepath))
    build_folder(config)  # Create folders for saving results

    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    torch.autograd.set_detect_anomaly(True)
    torch.random.manual_seed(config["seed"])
    torch.cuda.random.manual_seed(config["seed"])
    torch.set_printoptions(10)
    if config["dtype"] == "float64":
        datatype = torch.float64
    elif config["dtype"] == "float32":
        datatype = torch.float32

    # Define a lens system
    start = time.perf_counter()
    lens = do.Lensgroup(device=device)
    d_list = [4.96, 2.68, 1.31, 2.73, 3.4]
    system_scale = config["system_scale"]
    if config["single_lens"]:
        # lens.d_sensor = torch.tensor([system_scale * (14.96)], device=device) # for shorter
        lens.d_sensor = torch.tensor([system_scale * (56.43)], device=device) # for longer
    else:
        lens.d_sensor = torch.tensor(
            system_scale * (41.58 - 0.23 + sum(d_list)), device=device)

    if config["load_surface"]:
        tmp_d = lens.d_sensor
        del lens
        lens = load_lens(config["lens_name"])
        if torch.is_tensor(lens.d_sensor):
            lens.d_sensor = lens.d_sensor.detach()
    elif config["load_surface_from_txt"]:
        lens.load_file(config["lens_txt"])
        surfaces = lens.surfaces
        materials = lens.materials
    else:
        surfaces = initialize_lens(config, d_list, device)
        materials = initialize_materials(config)
        lens.load(surfaces, materials)

    wavelength = torch.Tensor([config["disp_wv"]]).to(device)  # unit: nm
    pxl_size = 2 * config["width"] / config["dim"]
    half_pxl_size = config["width"] / config["dim"]
    psf_rad_grid = config["psf_rad"] // pxl_size
    dim_new = int(config["dim"] + 2 * psf_rad_grid)
    lens.pixel_size = (
        2 * config["width"] + 2 * config["psf_rad"] - 2 * half_pxl_size) / (dim_new - 1)
    lens.film_size = [dim_new * 1.4, dim_new * 1.4]

    # Compute the focal length as a reference
    if config["single_lens"]:
        materials = lens.materials
        ri = materials[1].ior(wavelength)
        est_f = 1 / ((ri - 1) * (-lens.surfaces[1].c - -lens.surfaces[0].c + (
            (ri - 1) * lens.surfaces[1].d * -lens.surfaces[1].c * -lens.surfaces[0].c) / (ri)))
        print('===\nestimated focal length: ', est_f)
        print('curvature', lens.surfaces[0].c, lens.surfaces[1].c, '\n===')

    if config["use_deeplens"]:
        lens = GeoLens(filename=config["lens_name"])
        # lens.analysis(render=False)

    if config["is_lens"] and not config["use_deeplens"] and False:
        ax, fig = lens.plot_setup2D()
        fig.savefig("./layout_shape.png", bbox_inches='tight')
        plt.close()
        plot_lens_with_ray(lens, config, wavelength.item())
        compute_efl(lens, config, wavelength)

    # Copy the json file to the display folder
    os.system(f"cp {filepath} {config['display_folder']}")
    os.system(f"cp {filepath} {config['record_folder']}")

    # Run blocks
    for block in config["blocks"]:
        print("Block: ", block)
        function = globals()[block]
        function(config, lens)

