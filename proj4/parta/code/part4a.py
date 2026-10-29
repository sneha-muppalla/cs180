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
 
 
def get_points(im1, im2, n):
    """Click n matching points on each image (in the same order)."""
    plt.imshow(im1)
    plt.title('Click %d points on image 1' % n)
    pts1 = np.array(plt.ginput(n, timeout=0))
    plt.close()
 
    plt.imshow(im2)
    plt.title('Click the same %d points on image 2' % n)
    pts2 = np.array(plt.ginput(n, timeout=0))
    plt.close()
    return pts1, pts2
 
 
def apply_H(pts, H):
    """Apply homography H to an (n, 2) array of points."""
    pts = np.asarray(pts, dtype=float)
    ones = np.ones((len(pts), 1))
    p = np.hstack([pts, ones]) @ H.T
    return p[:, :2] / p[:, 2:3]
 

# this is 4.2 impolement computeH
def computeH(pts1, pts2):
    """Solve for H (with H[2,2] = 1) so that pts1 -> pts2, using least squares."""
    pts1 = np.asarray(pts1, dtype=float)
    pts2 = np.asarray(pts2, dtype=float)
    assert len(pts1) >= 4 and pts1.shape == pts2.shape
 
    A = []
    b = []
    for (x, y), (u, v) in zip(pts1, pts2):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.append(v)
 
    h = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)[0]
    H = np.append(h, 1).reshape(3, 3)
    return H
 
 
def get_bounds(shapes, Hs):
    # Find the bounding box (xmin, ymin, xmax, ymax) of all warped images
    all_pts = []
    for shape, H in zip(shapes, Hs):
        h, w = shape[:2]
        corners = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
        all_pts.append(apply_H(corners, H))
    all_pts = np.vstack(all_pts)
    xmin, ymin = np.floor(all_pts.min(axis=0)).astype(int)
    xmax, ymax = np.ceil(all_pts.max(axis=0)).astype(int)
    return xmin, ymin, xmax, ymax
 
 
def warp(im, H, bounds=None, method='bilinear'):
    # Inverse warp im with H. Returns (warped image, mask, (xmin, ymin))
    if bounds is None:
        bounds = get_bounds([im.shape], [H])
    xmin, ymin, xmax, ymax = bounds
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    ih, iw = im.shape[:2]
 
    # every pixel in the output canvas
    xs, ys = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))
    out_pts = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
 
    # go backwards: where does each output pixel come from in the source?
    src = np.linalg.inv(H) @ out_pts
    sx = src[0] / src[2]
    sy = src[1] / src[2]
 
    inside = (sx >= 0) & (sx <= iw - 1) & (sy >= 0) & (sy <= ih - 1)
    sx = sx[inside]
    sy = sy[inside]
 
    if method == 'nearest':
        vals = im[np.round(sy).astype(int), np.round(sx).astype(int)]
    else:
        x0 = np.floor(sx).astype(int)
        y0 = np.floor(sy).astype(int)
        x1 = np.minimum(x0 + 1, iw - 1)
        y1 = np.minimum(y0 + 1, ih - 1)
        dx = (sx - x0)[:, None]
        dy = (sy - y0)[:, None]
        vals = ((1 - dx) * (1 - dy) * im[y0, x0] + dx * (1 - dy) * im[y0, x1]
                + (1 - dx) * dy * im[y1, x0] + dx * dy * im[y1, x1])
 
    out = np.zeros((height * width, im.shape[2]))
    out[inside] = vals
    mask = inside.reshape(height, width)
    return out.reshape(height, width, im.shape[2]), mask, (xmin, ymin)
 
 
def warpImageNearestNeighbor(im, H):
    return warp(im, H, method='nearest')[0]
 
 
def warpImageBilinear(im, H):
    return warp(im, H, method='bilinear')[0]
 
 
def mosaic(images, Hs):
    # Blend warped images using weights that are biggest in the image center.
    bounds = get_bounds([im.shape for im in images], Hs)
    total = None
    weight_sum = None
 
    for im, H in zip(images, Hs):
        h, w = im.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        # distance to the nearest image edge
        dist = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy)) + 1.0
        dist = dist / dist.max()
        weight_im = np.dstack([dist, dist, dist])
 
        warped, mask, _ = warp(im, H, bounds)
        weights, _, _ = warp(weight_im, H, bounds)
        weights = weights[..., 0] * mask
 
        if total is None:
            total = np.zeros_like(warped)
            weight_sum = np.zeros_like(weights)
        total += warped * weights[..., None]
        weight_sum += weights
 
    result = np.zeros_like(total)
    ok = weight_sum > 0
    result[ok] = total[ok] / weight_sum[ok][:, None]
    return result
 
 
if __name__ == '__main__':
    #  curr example is  warp image 1 into image 2's frame and blend them together
    im1 = load_image('left.jpg')
    im2 = load_image('right.jpg')
 
    pts1, pts2 = get_points(im1, im2, 8)
    H = computeH(pts1, pts2)
    print('H =\n', H)
 
    # reprojection error as a sanity check
    err = np.linalg.norm(apply_H(pts1, H) - pts2, axis=1)
    print('mean error (pixels):', err.mean())
 
    save_image('warp_nearest.png', warpImageNearestNeighbor(im1, H))
    save_image('warp_bilinear.png', warpImageBilinear(im1, H))
 
    # im2 is the reference, so its homography is the identity
    result = mosaic([im2, im1], [np.eye(3), H])
    save_image('mosaic.png', result)