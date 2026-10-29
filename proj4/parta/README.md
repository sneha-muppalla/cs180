# Project 4 · Part A

`code/part4a.py` contains the homography, warping, and blending functions.
`code/part4a_cli.py` handles point selection and command-line output. Keep both
files together; the commands below still run through `part4a.py`.

Run commands from `proj4/parta`. Use Python with NumPy, Pillow, and Matplotlib.
The existing environment works: `../../proj3/.venv-partb/bin/python`.
Commands below use `python` for brevity; substitute that interpreter if needed.

## A.2 · Recover homographies

Choose one reference image per set (usually the middle view). Select 12 or more
matching landmarks spread across the overlap, avoiding moving objects,
reflections, repeated textures, and nearly collinear selections. Click matching
points in the same numbered order in both panels. Right-click undoes the last
click. Close the window after selecting the requested number in each panel.

```sh
python code/part4a.py pick images/set1_1.png images/set1_2.png points/set1_left.json
python code/part4a.py pick images/set1_3.png images/set1_2.png points/set1_right.json
python code/part4a.py pick images/set2_1.png images/set2_2.png points/set2_left.json
python code/part4a.py pick images/set2_3.png images/set2_2.png points/set2_right.json
python code/part4a.py pair points/set1_left.json results/set1_left
python code/part4a.py pair points/set1_right.json results/set1_right
python code/part4a.py pair points/set2_left.json results/set2_left
python code/part4a.py pair points/set2_right.json results/set2_right
```

Images are EXIF-oriented and resized to at most 1200 pixels on the longest side.
Clicks, H, and reprojection errors refer to these resized coordinates. The JSON
saves the resize setting; do not edit it after clicking. Use `--max-side 1600`
**before** the subcommand to change the working resolution.

For a match `(x,y) -> (u,v)`, the two rows of `Ah=b` are:

```text
[x y 1 0 0 0 -ux -uy] h = u
[0 0 0 x y 1 -vx -vy] h = v
h = [h11 h12 h13 h21 h22 h23 h31 h32]^T, h33 = 1
```

There are **2n equations** for 8 unknowns. `computeH` centers/scales both point
sets, solves the normalized overdetermined system with `numpy.linalg.lstsq`,
and converts H back into image coordinates. `system.npz` stores the equivalent
original-coordinate A and b for the report. `H.txt` stores H; `report.json`
stores per-point errors and RMSE. `correspondences.png` numbers the clicked
matches. A large error suggests a mismatched point, parallax, or lens distortion;
inspect the visualization before accepting the transform.

## A.3 · Inverse warping and rectification

For each destination pixel, multiply by `H^-1` and divide by the homogeneous
coordinate to find its source location. Nearest neighbor rounds it to a pixel;
bilinear computes a weighted sum of the four surrounding pixels. Both are
implemented directly with NumPy, with no interpolation library. Transformed
corners determine the output bounds. Masks distinguish missing pixels from
valid black pixels. `warp_image` exposes the mask and canvas origin; the two
assignment-named wrapper functions return just the warped image.

Each `pair` command above produces `nearest.png`, `bilinear.png`, masks, and
measured execution times. Compare the same diagonal edge or lettering at a
zoomed scale: nearest typically looks more jagged, while bilinear looks smoother
and may soften fine details. Bilinear performs more arithmetic; use the measured
times to describe the actual speed difference on your machine. Single timings
include allocation and should be treated as approximate.

For **two different photos** containing known rectangles, click corners in
top-left, top-right, bottom-right, bottom-left order:

```sh
python code/part4a.py rectify images/rect1.png results/rect1 --width 600 --height 400
python code/part4a.py rectify images/rect2.png results/rect2 --width 500 --height 500
```

Replace those filenames with your own photos. Choose width/height to reflect
the object's known aspect ratio (a square uses equal dimensions). Each command
saves the clicks, H, nearest/bilinear results, masks, and timings. The output is
cropped to the target rectangle. Four corners suffice here because their target
geometry is explicitly specified; use more points for the mosaic pairs.

## A.4 · Blend mosaics

```sh
python code/part4a.py mosaic results/mosaic1 points/set1_left.json points/set1_right.json
python code/part4a.py mosaic results/mosaic2 points/set2_left.json points/set2_right.json
```

Each source maps directly into its set's reference frame. Compute the union of
the transformed image corners, inverse-warp all images into that common canvas,
and blend with `sum(weight * color) / sum(weight)`. Source weights increase with
distance from the nearest image boundary; their warped masks remove invalid
pixels. This reduces abrupt seams but cannot fix misregistration, parallax,
moving objects, or exposure differences. Output includes `mosaic.png`, a valid
pixel mask, and all homographies.

The assignment requests **three mosaics with source images**. The helper below
produces two full scene mosaics and a third two-view kitchen mosaic. The webpage
shows their sources and explicitly identifies the reused scene.

## Generated report outputs

The webpage now includes four correspondence reports, two rectification
comparisons, and three feathered mosaics. Reproduce the generated artifacts with:

```sh
python code/generate_results.py
```

This helper additionally requires OpenCV (`opencv-python`). It uses SIFT,
a ratio test, and RANSAC to propose/filter matching points. Final homographies,
all interpolation, and mosaic blending use our implementation in `part4a.py`.
Saved correspondence JSON files disclose the selection method. These are
**automatically selected correspondences**, not user clicks.

The kitchen reference is image 2; the room reference is image 3. The manual
commands above remain examples; the generated files use names such as
`points/set2_1_to_3.json`.

Rectification corners were manually identified on the stove controls and box
front. Physical dimensions were unavailable, so target aspect ratios are
approximate. Each result includes the selected corners and median timings over
five runs. Three mosaic outputs use two scenes: full kitchen, full room, and a
two-view kitchen mosaic. A third distinct scene is not included.

All generated outputs live in `results/`, and the corresponding webpage embeds
them directly. Small alignment artifacts remain from parallax and imperfect
matches; blending does not correct the geometry.
