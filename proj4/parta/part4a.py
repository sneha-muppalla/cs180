"""Homographies, inverse warping, and a simple blended mosaic.
Points are (x, y). H maps points in the source image to the destination image.
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#some helper functions i added 
 
def load_image(path, max_side=800):
    im = Image.open(path).convert('RGB')
    im.thumbnail((max_side, max_side))
    return np.array(im) / 255.0
 
 
def save_image(path, im):
    im = np.clip(im, 0, 1)
    Image.fromarray((im * 255).astype(np.uint8)).save(path)
 
 
# this is 4.2 impolement computeH
def computeH(im1_pts,im2_pts):
 
    A, b = [], []

    # Ah = b, where A is 2nx8, h is 8 x 1, and b is 2nx1 
    # calculating the 8 unknowns in h, which is the homography matrix H
    for (x, y), (u, v) in zip(im1_pts, im2_pts): 
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.append(v)
 
    solution = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
    h = solution[0]

    H = np.zeros((3, 3))
    H[0] = h[0:3]
    H[1] = h[3:6]
    H[2] = [h[6], h[7], 1]
    return H
 
def warpImageNearestNeighbor(im, H):
    h, w = im.shape[:2]

    corners = [(0, 0), 
               (w-1, 0), 
               (w-1, h-1), 
               (0, h-1)]
    xs, ys = [], []
    for (x, y) in corners:
        p = H @ np.array([x, y, 1])
        xs.append(p[0] / p[2])    
        ys.append(p[1] / p[2])

    xmin, xmax = int(np.floor(min(xs))), int(np.ceil(max(xs)))
    ymin, ymax = int(np.floor(min(ys))), int(np.ceil(max(ys)))
    out_w = xmax - xmin + 1
    out_h = ymax - ymin + 1

    out = np.zeros((out_h, out_w, 3), dtype=im.dtype)
    alpha = np.zeros((out_h, out_w))
    H_inv = np.linalg.inv(H)

    for row in range(out_h):
        for col in range(out_w):
            x_out = col + xmin
            y_out = row + ymin

            p = H_inv @ np.array([x_out, y_out, 1])
            sx = p[0] / p[2]
            sy = p[1] / p[2]

            xi = int(round(sx))
            yi = int(round(sy))

            if 0 <= xi < w and 0 <= yi < h:
                out[row, col] = im[yi, xi]
                alpha[row, col] = 1

    return out, alpha, (xmin, ymin)

def warpImageBilinear(im, H):
    h, w = im.shape[:2]
    corners = [(0, 0), (w-1, 0), (w-1, h-1), (0, h-1)]
    xs, ys = [], []
    for (x, y) in corners:
        p = H @ np.array([x, y, 1])
        xs.append(p[0] / p[2])    
        ys.append(p[1] / p[2])

    xmin, xmax = int(np.floor(min(xs))), int(np.ceil(max(xs)))
    ymin, ymax = int(np.floor(min(ys))), int(np.ceil(max(ys)))
    out_w = xmax - xmin + 1
    out_h = ymax - ymin + 1

    # empty output + alpha mask
    out = np.zeros((out_h, out_w, 3), dtype=im.dtype)
    alpha = np.zeros((out_h, out_w))
    H_inv = np.linalg.inv(H)

    # loop over every output pixel
    for row in range(out_h):
        for col in range(out_w):
            x_out = col + xmin
            y_out = row + ymin

            p = H_inv @ np.array([x_out, y_out, 1])
            sx = p[0] / p[2]
            sy = p[1] / p[2]


            if 0 <= sx <= w-1 and 0 <= sy <= h-1:
                x0 = int(np.floor(sx))
                y0 = int(np.floor(sy))
                x1 = min(x0 + 1, w-1)    
                y1 = min(y0 + 1, h-1)
                dx = sx - x0          
                dy = sy - y0

                top = (1-dx) * im[y0, x0] + dx * im[y0, x1]
                bottom = (1-dx) * im[y1, x0] + dx * im[y1, x1]
                out[row, col] = (1-dy) * top + dy * bottom
                alpha[row, col] = 1

    return out, alpha, (xmin, ymin)

def mosaic(images, Hs):
    # more than two images in this set
    warped_images = []
 
    for im, H in zip(images, Hs):
        h, w = im.shape[:2]
        rows, cols = np.arange(h)[:, None], np.arange(w)[None, :]
        # distance to the nearest image edge
        dist = np.minimum(np.minimum(rows, h - 1 - rows), np.minimum(cols, w - 1 - cols)) + 1
        dist = dist / dist.max()

        weight_im = np.dstack([dist]*3)
 
        warped, mask, origin = warpImageBilinear(im, H)
        weights, ign, ore = warpImageBilinear(weight_im, H)
        weights = weights[..., 0] * mask
 
        warped_images.append((warped, weights, origin))
 
    # getting the bounds for the warped image 
    xmin = min(origin[0] for ign, ore, origin in warped_images)
    ymin = min(origin[1] for ign, ore, origin in warped_images)
    xmax = max(origin[0] + warped.shape[1] for warped, _, origin in warped_images)
    ymax = max(origin[1] + warped.shape[0] for warped, _, origin in warped_images)
    total = np.zeros((ymax - ymin, xmax - xmin, 3))

    total_weights = np.zeros((ymax - ymin, xmax - xmin))

    for warped, weights, origin in warped_images:
        row = origin[1] - ymin
        col = origin[0] - xmin
        h, w = warped.shape[:2]
        total[row:row + h, col:col + w] += warped * weights[..., None]
        total_weights[row:row + h, col:col + w] += weights

    result = np.zeros_like(total)
    mask = total_weights > 0
    result[mask] = total[mask] / total_weights[mask][..., None]
    return result
 
 
if __name__ == '__main__':

    # part 4.2   
    # Enter your manually selected point coordinates below.
    im1 = plt.imread("images/set2_1.png")
    im2 = plt.imread("images/set2_2.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.imshow(im1)
    plt.title("Click 8  points on image 1")
    im1_pnts = np.array(plt.ginput(8, timeout=0))
    plt.close()

    plt.imshow(im2)
    plt.title("Click the same 8 points in the same order")
    im2_pnts = np.array(plt.ginput(8, timeout=0))
    plt.close()

    H = computeH(im1_pnts, im2_pnts)
    print('H =\n', H)

    # 4.3 warp 
 
    save_image('warp_nearest.png', warpImageNearestNeighbor(im1, H)[0])
    save_image('warp_bilinear.png', warpImageBilinear(im1, H)[0]) 
    # 4.4 mosaic
    result = mosaic([im2, im1], [np.eye(3), H])
    save_image('mosaic.png', result)