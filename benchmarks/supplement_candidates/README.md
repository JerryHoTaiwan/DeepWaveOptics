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

## Sensor lateral positions and generation chain

Each primary case now includes `lateral_positions.csv` with all 81 sensor-plane
centers. `psf_index` matches the NPZ leading index, original PTH suffix, and
`stack_img_<index>.png`. `in_archived_rgb_subset=1` identifies the 25 RGB crops.
Coordinates are physical mm in internal (y,x) order; z is the lens sensor position.
They are sensor crop centers, not illumination offsets, object-plane positions,
or measured PSF centroids.

The historical chain is:

1. `main.py` constructs or loads the lens and sets sensor z.
2. `render.py` calls `send_chief_ray` separately for each wavelength. It matches
   rays launched from the designated stop to a coarse sensor grid (chief_gap=64),
   interpolates direction tilt, and traces toward the front to obtain incident
   directions and beam origins.
3. `render.py` samples every 32nd row/column of a 257×257 sensor lattice, with
   row-major PSF indices 0–80. Directions/origins and sensor centers are rotated
   to align with that indexing.
4. `plotter.py::plot_zoomin_img` selects the center from
   `land_pos[0,:,256-(i%257),i//257]`, renders an endpoint-inclusive crop, and
   saves a unit-sum intensity PSF. The RGB stage then scales each color to peak 250.

For j=9*row+col, render row r=32*row and column c=32*col:

```
sensor_y_mm = center_y - width + 2*width*c/256
sensor_x_mm = center_x - width + 2*width*r/256
```

Thus image row varies sensor x; image column varies sensor y. The full grids
span -width to +width on both axes. The selected RGB subset spans -width to 0.
For singlet, the 5 axis positions are -0.368, -0.276, -0.184, -0.092, 0 mm;
for Cooke, -0.75, -0.5625, -0.375, -0.1875, 0 mm. Position tables apply to all
three wavelengths because the centers come from the same sensor lattice;
chief-ray directions and beam shifts can differ by wavelength.

Regenerate a table using:

```bash
python scripts/export_archived_lateral_positions.py benchmarks/supplement_candidates/1114_wave_single_wide
```

To recover angles, use these positions with the supplied lens geometry and
sensor z and the historical chief-ray aiming convention. A simple atan(position/EFL)
calculation is not the historical aiming procedure. This release does not supply
an independently validated angle table or archived beam-origin table.

The adjacent archived configs are incomplete execution snapshots: their zi_idx
lists only a few positions although the output batches contain all 81. Their
crop-ratio dictionaries are also incomplete. Consequently these config files
alone are insufficient to rerender every archived optical PSF. The lateral
position table follows the recovered historical grid/index code, independently
of which positions were selected for saving. Historical code files are retained
unaltered, including their original formatting.
