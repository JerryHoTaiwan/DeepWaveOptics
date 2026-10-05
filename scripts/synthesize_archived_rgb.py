"""Rebuild the historical 5x5 RGB subset from packaged 9x9 PSF arrays."""
import argparse
from pathlib import Path
import cv2
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('case', type=Path, help='A benchmarks/supplement_candidates case directory')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
arrays = [np.load(args.case / ('psfs_%d.npz' % w))['psfs'] for w in (440, 510, 650)]
args.output.mkdir(parents=True, exist_ok=True)
for j in range(81):
    if j // 9 <= 4 and j % 9 <= 4:
        # OpenCV BGR order: 440/510/650 nm become blue/green/red.
        channels = [a[j] * 250 / a[j].max() for a in arrays]
        image = np.rint(np.stack(channels, axis=2)).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(args.output / ('stack_img_%d.png' % j)), image)
