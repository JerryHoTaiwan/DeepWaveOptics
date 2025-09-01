import json
from pathlib import Path
from typing import Any, Dict
import os


def load_config(path: str) -> Dict[str, Any]:
    """Load a JSON config file into a dictionary."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        config = json.load(f)
    return config


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