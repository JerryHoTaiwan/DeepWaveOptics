"""Export sensor-plane PSF centers using the historical render/plot index mapping."""
import argparse
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('case', type=Path)
args = parser.parse_args()
config = json.loads((args.case / 'archived_config.json').read_text())
dim, interval = config['dim'], config['interval']
cy, cx = config['center']
width = config['width']
positions = list(range(0, dim, interval))
output = args.case / 'lateral_positions.csv'
with output.open('w', newline='') as stream:
    fields = ['psf_index', 'grid_row', 'grid_col', 'render_flat_index',
              'sensor_grid_row', 'sensor_grid_col', 'sensor_y_mm', 'sensor_x_mm',
              'in_archived_rgb_subset']
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row, r in enumerate(positions):
        for col, c in enumerate(positions):
            # plot_zoomin_img: center = land_pos[0,:,dim-1-(i%dim),i//dim]
            writer.writerow(dict(psf_index=row*len(positions)+col,
                                 grid_row=row, grid_col=col,
                                 render_flat_index=r*dim+c,
                                 sensor_grid_row=dim-1-c, sensor_grid_col=r,
                                 sensor_y_mm=format(cy-width+2*width*c/(dim-1), '.12g'),
                                 sensor_x_mm=format(cx-width+2*width*r/(dim-1), '.12g'),
                                 in_archived_rgb_subset=int(row<=4 and col<=4)))
print(output)
