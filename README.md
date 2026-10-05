# A Differentiable Wave Optics Model for End-to-End Computational Imaging System Optimization

[![Project Website](https://img.shields.io/badge/Website-Live-blue)](https://jerryhotaiwan.github.io/diff_wave_optics/)

📢 **This work has been accepted to ICCV 2025.**

---

<p align="center">
  <img src="images/DiffWave.png" alt="Teaser Figure" width="80%"/>
</p>

---

This repository provides a differentiable wave optics simulator for **point spread function (PSF) generation**, **image formation**, and **end-to-end optimization** of computational imaging systems.  
It is designed for research in computational imaging, holography, and optics-aware machine learning.

---

## Features

- **Differentiable Wave Optics Simulation**  
  Forward model with gradient support for optimization.

- **PSF Generation**  
  Simulate diffraction-limited or aberrated point spread functions.

- **Image Formation**  
  Render images under custom optical systems.

- **End-to-End Optimization**  
  Jointly optimize optical parameters and downstream tasks.

- **Config-Driven Pipeline**  
  All experiments are configured via JSON files for reproducibility.

---

## PSF reproduction assets

Recovered lens files, a standalone 40-degree singlet runner, archived DWO
PSFs, and documented sampling conventions are available in
[benchmarks/README.md](benchmarks/README.md). Start with
`python scripts/reproduce_psf.py --check` from the repository root.
The documentation distinguishes recovered experiments from figure settings
whose provenance remains unresolved.

## Installation

Clone the repository:
```bash
git clone https://github.com/JerryHoTaiwan/DeepWaveOptics.git
cd DeepWaveOptics
```

Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

---

## Project Structure

```
DeepWaveOptics/
├── configs/          # JSON configs (default/experiments/env)
├── src/              # Core implementation
│   ├── functions.py  # Callable blocks (PSF, render, optimize, etc.)
│   ├── utils.py      # Config + folder utilities
│   ├── main.py       # Entry point
│   └── initialization.py
├── results/          # Outputs (PSFs, images, logs, checkpoints)
└── README.md
```

---

## Examples

### Run with a Config File
All experiments are controlled via JSON configs:
```bash
python main.py --config configs/experiments/show_psfs.json
```

- **Generate PSFs**
```bash
python main.py --config configs/show_psfs.json
```

- **Render Images**
```bash
python main.py --config configs/display_triplet.json
```

- **End-to-End Optimization**
```bash
python main.py --config configs/train_cellphonelens.json
```

---

## Citation

If you use this code in your work, please cite:

```
@article{ho2024differentiable,
  title={A Differentiable Wave Optics Model for End-to-End Computational Imaging System Optimization},
  author={Ho, Chi-Jui and Belhe, Yash and Rotenberg, Steve and Ramamoorthi, Ravi and Li, Tzu-Mao and Antipa, Nicholas},
  journal={arXiv preprint arXiv:2412.09774},
  year={2024}
}
```

---

## Acknowledgments

This project builds on the excellent work of the [DeepLens](https://github.com/singer-yang/DeepLens) and DiffOptics frameworks developed by Xinge Yang *et al.*. Their differentiable ray–wave model provided the foundation for extending this project toward end-to-end optimization of computational imaging systems. We gratefully acknowledge their contributions and encourage readers to also cite their work:

```
@article{yang2024end,
  title={End-to-End Hybrid Refractive-Diffractive Lens Design with Differentiable Ray-Wave Model},
  author={Yang, Xinge and Souza, Matheus and Wang, Kunyi and Chakravarthula, Praneeth and Fu, Qiang and Heidrich, Wolfgang},
  journal={arXiv preprint arXiv:2406.00834},
  year={2024}
}
```
