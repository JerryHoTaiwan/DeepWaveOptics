# Recovered PSF reproduction assets

This package repairs missing public assets and documents recovered experimental
evidence. It is not yet a claim of exact reproduction of every published figure.
No assets were taken from Diffuser_RT.

## Run one recovered case

From the repository root, with the project dependencies installed:

```bash
python scripts/reproduce_psf.py --check
python scripts/reproduce_psf.py
```

The default is the recovered 40-degree singlet case. It writes a unit-sum
`psf.npy` and `metadata.json` to `results/reproduction/singlet_40deg`.
It does not require DIV2K, LPIPS, or a pretrained network. CPU is supported;
use a GPU for practical benchmark runs. A quick execution check is:

```bash
python scripts/reproduce_psf.py --dim 33 --layers 12
```

Those overrides change physical sampling and are not an accuracy benchmark.
The script requires torch, numpy, scipy, matplotlib, opencv-python, meshio,
torchvision, Pillow, and the dependencies imported by the bundled DeepLens code.
The original requirements file contains local editable-package assumptions;
the repository root and src are added to the import path by this script.

## Recovered singlet conditions

| Parameter | Value |
|---|---|
| Lens | `lenses/single_fov40.txt` (unaltered development lens file) |
| Lens coordinates | millimeters; first surface z=0, second surface z=3 |
| Sensor z | 33 mm (3 mm thickness + 30 mm last-surface/image gap) |
| Illumination | parallel rays; circular disk radius 1.5 mm at z=0 |
| Beam origin shift | internal y=-1.9003209864 mm, x=0 |
| Incidence | view=40 degrees, rot=0 degrees |
| Wavelength | 532 nm |
| Reference sphere exit-pupil z | 3 mm, supplied by historical config |
| PSF center, internal (y,x) | (34.8607623332, 0) mm |
| Sensor crop half-width | 0.407 mm |
| Sensor samples | 257 x 257, endpoint-inclusive |
| Actual sample spacing | 0.814/256 = 0.0031796875 mm |
| Effective radial sampling layers | 100 |
| Normalization | intensity = abs(field)^2, then unit sum |

The archived config requested 300 layers, but the historical `trace_all`
at development commit `c6a4f40` caps single-lens fields above 30 degrees at
100 layers. The public refactor instead always uses config.layers. This recovered
runner explicitly uses 100 and records the archived request as `archived_layers`.
For this ring sampler (below its cutoff), the generated ray count is
1 + 3*M*(M-1); the surviving count is recorded separately in metadata.

## Angle, illumination, and chief-ray conventions

Internal ray coordinates are (y,x,z). With view=theta and rot=phi in degrees,
the direction vector is

```
d = (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))
```

These are polar tilt and azimuth, not independent x/y Euler angles. For signed
axis slopes alpha_y=atan2(d_y,d_z), alpha_x=atan2(d_x,d_z), the equivalent
polar parameters are theta=atan(sqrt(tan(alpha_y)^2+tan(alpha_x)^2)) and
phi=atan2(tan(alpha_x),tan(alpha_y)) for forward rays.
This formula documents the code convention; it does not establish the
two-axis angle labels used in the supplementary figures.

`sample_rad` is an illumination radius, not a diameter or automatically the
physical stop radius. `stop_ind` is used as a one-based split location through
`surfaces[stop_ind-1]`. Lens surface apertures additionally clip rays.
`send_chief_ray` builds a sensor-to-object mapping from a stop-centered ray fan,
nearest-bin selection and interpolation. The recovered single-case runner
uses the archived beam shift directly rather than recomputing that mapping.

## Wavelength and color

The original public `show_psf` function uses a hard-coded 532 nm wavelength;
`disp_wv=650` in its config does not change that function's wavelength.
The new single-case runner uses `disp_wv` explicitly.
The public `display` function defaults to [440,510,650] nm.
Historical `short_src/stack_multi_psf.py` stacks those arrays in that order,
scales each channel independently to peak 250, and writes with OpenCV (BGR).
Thus 440/510/650 correspond to displayed blue/green/red in that script.
That establishes one historical visualization convention, not confirmed
provenance for the exact color composition of Supplement Figures 1–2.

## Reference arrays

`reference/archived_dwo_*.npz` contains losslessly compressed saved DWO
outputs for focused/defocused Cooke and 35/40-degree singlet. Source hashes
are recorded in `manifest.json`. Load with:

```python
import numpy as np
psf = np.load('benchmarks/reference/archived_dwo_fov40.npz')['psf']
```

These saved development outputs are useful numerical targets, but their complete
execution provenance has not been recovered. No resampling, normalization or
precision conversion was applied during packaging. Zemax reference arrays,
export headers, and models are not supplied in this package. The recovered
text lens prescriptions are DWO inputs; their conversion provenance from any
original Zemax model has not been established.

The historical SSIM script peak-normalizes each PSF, resizes the candidate
with OpenCV to reference resolution, and uses the reference intensity range.
It does not establish matched physical pixel spacing or crop alignment.
Array equality and image similarity alone are insufficient to certify a fair
physical accuracy or timing comparison.

## Figure mapping and unresolved evidence

- [Original arXiv supplementary Figures 9–10](https://arxiv.org/html/2412.09774v1#S6.SS1)
  describe Cooke and singlet PSFs over 0–20-degree incidence.
  `archive/supp_*_candidate.json` are historical candidates, not verified
  figure-generating configs. Their grids are selected in sensor coordinates;
  they do not directly specify that two-axis angle grid.
- [Revised main Figure 6](https://arxiv.org/html/2412.09774v2#S4.SS1)
  compares monochromatic 532 nm focused/defocused Cooke and 35/40-degree singlet.
  The default runner recovers a 40-degree singlet experiment; exact equality
  to the published panel still requires checking.
- `archive/cooke_comparison.json` and `lenses/cooke_cmp.txt` are recovered Cooke
  evidence. The text lens loads sensor z=55.43902587890625 mm. It is not a
  verified focused/defocused pair. The exact focused sensor distance and
  defocus offset remain unresolved; no value was guessed.
- No matching original Zemax models or confirmed model-to-text/JSON conversion
  provenance were recovered. These materials are outside this release update.
- The public refactor differs from historical code in sampling control and
  phase handling (historical code subtracts the first OPL before forming
  phase). This patch leaves optical propagation unchanged and does not
  certify historical numerical or timing parity.

The full 257x257 recovered 40-degree case ran successfully on CPU with
29701 traced rays. Its peak-normalized SSIM against the same-size saved DWO
40-degree array is 0.9987598123897564 (no resizing). See
[validation.json](validation.json). This is not a Zemax SSIM measurement or
proof of published-panel identity; GPU timing was not rerun.
