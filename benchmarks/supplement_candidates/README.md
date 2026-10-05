# Recovered supplementary-grid candidates

These November 14–15, 2024 archives match the historical workflow of generating
many PSFs and then composing RGB crops. Their exact correspondence to
Supplement Figures 1–2 has not yet been established.

Two primary candidates are packaged:

| Directory | Original PSFs | RGB crops | Illumination radius (mm) | Sensor grid half-width (mm) |
|---|---:|---:|---:|---:|
| 1114_wave_single_wide | 81 positions × 3 wavelengths | 25 | 0.2 | 0.368 |
| 1114_wave_triplet_wide | 81 positions × 3 wavelengths | 25 | 0.1 | 0.75 |

`psfs_440.npz`, `psfs_510.npz`, and `psfs_650.npz` contain arrays under the
`psfs` key, indexed 0–80, plus `indices`. Original values and precision are
preserved without normalization or resizing. `manifest.json` records source
hashes, source timestamps and packaged hashes. `rgb/` contains archived PNGs.
`archived_config.json` preserves the original configuration, including stale
absolute development-machine paths; it is evidence, not a runnable release config.

Each original batch sampled a 9×9 sensor-position grid (dim=257, interval=32).
The RGB script selected rows 0–4 and columns 0–4, giving indices 0–4, 9–13,
18–22, 27–31 and 36–40. Each wavelength is independently scaled to peak 250;
OpenCV writes 440/510/650 nm as B/G/R. All 25 archived RGB images in each
primary candidate were exactly reproduced from the packaged source arrays.

```bash
python scripts/synthesize_archived_rgb.py benchmarks/supplement_candidates/1114_wave_single_wide --output results/supplement_rgb/singlet
python scripts/synthesize_archived_rgb.py benchmarks/supplement_candidates/1114_wave_triplet_wide --output results/supplement_rgb/cooke
```

## Lens and angle recovery

The singlet configuration has `load_surface=false`. It uses initialized surfaces,
not the `single_rms.pkl` filename in the config. Historical initialization uses
system_scale=0.1: surface radii of curvature 0.93506 and 8.750819 mm, thickness
0.496 mm, sensor z=1.496 mm, surface aperture radius 0.6 mm, and N-LAF2 glass.
The historical main/initialization code is included for review.

The primary Cooke configuration has `load_surface=true`: it loads the
wave-optimized `1016_triplet_recon_wave_proceed_2/lens_opt.pkl`. Its recovered
geometry is supplied as `lens_prescription.json` (surface d/r/c/k/ai, materials,
and sensor z). The pickle itself is not required to inspect this geometry.
Alternative Cooke configurations `1114_wave_pt_triplet_a1_wide` and
`1114_wave_pt_triplet_a3_wide` are included as evidence; they instead initialize
surfaces because `load_surface=false`. Their full PSF arrays are not packaged.
Matching the actual published panel is needed to select between these candidates.

Development commit bea760c (2024-11-15) selects sensor positions, obtains
incident directions and beam origins through `send_chief_ray`, and calculates
polar tilt/azimuth from those directions before rendering. Thus a uniform
sensor grid does not establish a uniform 0–20-degree incident-angle grid.
The exact supplementary angle-label table or subsequent angle calculation
has not been recovered. The archived code under `historical_code/` is an
inspection snapshot, not a standalone executable environment; its imports
refer to the historical development tree. RGB synthesis is runnable from
these assets, but full historical optical rerendering is not yet validated.

No Zemax PSFs/models, reconstruction networks, or unrelated image data are
included. These are supplementary-grid candidates, distinct from the four
monochromatic comparison cases described in ../four_cases.md.
