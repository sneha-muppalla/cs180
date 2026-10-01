import numpy as np
from scipy.signal import convolve2d
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


def pad_image(img, filter, mode='same'):
    fh, fw = filter.shape

    ph, pw = fh - 1, fw - 1

    if mode == 'same':
        ph, pw = fh // 2, fw // 2


    return np.pad(img, ((ph, ph), (pw, pw)), mode='constant'), ph, pw

def conv_4loops(img, filter, mode='same'):
    flipped = np.flip(filter)

    padded, ph, pw = pad_image(img, filter, mode)

    out_h, out_w = padded.shape[0] - flipped.shape[0] + 1, padded.shape[1] - flipped.shape[1] + 1
    out = np.zeros((out_h, out_w))
    
    for i in range(out_h):
        for j in range(out_w):
            for fi in range(flipped.shape[0]):
                for fj in range(flipped.shape[1]):
                    out[i, j] += padded[i + fi, j + fj] * flipped[fi, fj]
    return out

def conv_2loops(img, filter, mode='same'):
    flipped = np.flip(filter)
    padded, ph, pw = pad_image(img, filter, mode)

    out_h, out_w = padded.shape[0] - flipped.shape[0] + 1, padded.shape[1] - flipped.shape[1] + 1
    out = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = np.sum(padded[i:i+flipped.shape[0], j:j+flipped.shape[1]] * flipped)
    return out



# TASK 1.1 compare  with scipy
test_img = np.random.rand(100, 100)
test_kernel = np.random.rand(5, 5)

res_scipy = convolve2d(test_img, test_kernel, mode='same')
res_4loops = conv_2loops(test_img, test_kernel, mode='same')
res_2loops = conv_4loops(test_img, test_kernel, mode='same')
print("4-loop implementation matches scipy:", np.allclose(res_4loops, res_scipy))
print("2-loop implementation matches scipy:", np.allclose(res_2loops, res_scipy))

img = Image.open('task_1.1.jpeg')
corrected_img = ImageOps.exif_transpose(img).convert('L')
print('done img')
img_gray = np.array(corrected_img, dtype=np.float32) / 255.0


box_filter = np.ones((9, 9)) / 81.0
Dx = np.array([[-1, 1]])
Dy = np.array([[-1], [ 1]])


print('doing blur')
blurred_image = conv_2loops(img_gray, box_filter, mode='same')
print('doing x')
grad_x = conv_2loops(img_gray, Dx, mode='same')
print('doing y')
grad_y = conv_2loops(img_gray, Dy, mode='same')

#saving the images
plt.imsave('output_original.png', img_gray, cmap='gray')
plt.imsave('output_box_blur.png', blurred_image, cmap='gray')
plt.imsave('output_grad_x.png', grad_x, cmap='gray')
plt.imsave('output_grad_y.png', grad_y, cmap='gray')














