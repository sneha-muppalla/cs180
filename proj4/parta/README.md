# Project 4 · Part A

`part4a.py` implements least-squares homography estimation, nearest-neighbor
and bilinear inverse warping, and feathered mosaic blending.

Run from this folder with Python, NumPy, Pillow, and Matplotlib installed:

```sh
python part4a.py
```

The script opens the two images named in its main block. Click eight matching
features in the first image, then the same features in the same order in the
second. It prints H and saves `warp_nearest.png`, `warp_bilinear.png`, and
`mosaic.png` in the working directory. Change the image paths to process another
pair. The click coordinates are currently kept only in memory.

Both warping functions return `(image, mask, (xmin, ymin))`. The mosaic uses
these origins to place each warped image on a common canvas before blending.

`images/` contains the original photos. `previews/` contains smaller copies used
by the webpage. `results/` contains the earlier generated report images and
numerical records. Those earlier results used automatic correspondences and
have not yet been regenerated with the current manually selected points.

The webpage is `../index.html`. Its rectification examples and timing comparison
need fresh runs with the current implementation before the report is final.
