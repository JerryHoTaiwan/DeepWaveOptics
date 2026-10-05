# Four recovered monochromatic PSF cases

These cases address the focused/defocused Cooke and wide-angle singlet comparison
in revised main Figure 6. All use 532 nm, a circular illumination radius of
1.5 mm, 100 radial sampling layers and a 257 × 257 endpoint-inclusive sensor
crop. Coordinates and distances are in mm. See README.md for angle and aperture
conventions. The complete beam shifts, centers and lens paths are in each JSON.

| Configuration | Sensor z | Crop half-width | Peak-normalized SSIM to saved DWO |
|---|---:|---:|---:|
| cooke_focused.json | 60.43902587890625 | 0.0721675 | 0.9996170663 |
| cooke_defocused.json | 55.43902587890625 | 0.199877 | 0.9248467724 |
| singlet_35deg.json | 33 | 0.407 | 0.9992424260 |
| singlet_40deg.json | 33 | 0.407 | 0.9987598124 |

Run from the repository root:

```bash
python scripts/reproduce_psf.py --config configs/reproduction/cooke_focused.json
python scripts/reproduce_psf.py --config configs/reproduction/cooke_defocused.json
python scripts/reproduce_psf.py --config configs/reproduction/singlet_35deg.json
python scripts/reproduce_psf.py --config configs/reproduction/singlet_40deg.json
```

Each writes a unit-sum `psf.npy` and metadata. Packaged `reference/rerun_dwo_*.npz`
contain the CPU reruns; `reference/archived_dwo_*.npz` contain the unchanged saved
DWO arrays. Both use the `psf` key. `four_cases_validation.json` records the
rerun environment, ray counts and measured agreement. No resizing was used
for these comparisons. SSIM is calculated after independently dividing both
arrays by their maxima, with data_range=1.

## Recovery evidence and limitations

The Cooke lens uses the same six optical surfaces in both configurations.
The focused distance is supported by the archived DeepLens Cooke model's
explicit sensor position and a paraxial ray focus near z=60.467 mm. The DWO
text prescription places the sensor at z=55.43902587890625 mm, giving a 5 mm
shift toward the lens. Crop widths were recovered from development configs.
The 35-degree beam shift and sensor center were recovered from historical
single-PSF code constants; the 40-degree case comes from an archived config.

Three reruns closely match the saved development arrays. The defocused case
still has a meaningful residual difference: its exact historical sampling,
crop or rendering settings remain unresolved. These configurations and arrays
are recovered evidence, not certification of every published panel. Supplement
Figures 1–2 two-axis grids remain separate unresolved reproduction requests.
CPU reruns do not establish GPU timing or cross-method accuracy rankings.

The local archive also contains four-case outputs for Yang, Yang_10k, Wei,
Chen and Airy. Some current baseline scripts were overwritten or contain
inconsistent case labels/parameters, so their outputs and configurations are
not included as validated reproduction assets in this update. Zemax reference
PSFs and models are not included.
