import cv2
import numpy as np
from scipy.signal import convolve2d
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

img = Image.open('cameraman.png')
img_gray = np.array(img.convert('L'), dtype=np.float32) / 255.0
print(f"Image successfully loaded. Shape: {img_gray.shape}")


Dx = np.array([[-1, 1]])
Dy = np.array([[-1], [ 1]])


# 1D Gaussian kernel
filter = 15
g_1d = cv2.getGaussianKernel(filter)
gaussian_filter = np.outer(g_1d, g_1d.T)


# gaussian blue 2d 
img_blurred = convolve2d(img_gray, gaussian_filter, mode='same', boundary='fill', fillvalue=0)

grad_x_A = convolve2d(img_blurred, Dx, mode='same', boundary='fill', fillvalue=0)
grad_y_A = convolve2d(img_blurred, Dy, mode='same', boundary='fill', fillvalue=0)

magnitude_A = np.sqrt(grad_x_A**2 + grad_y_A**2)
#threshold is 0.18
edges_A = (magnitude_A > 0.18).astype(np.float32)

DoG_x = convolve2d(gaussian_filter, Dx, mode='same', boundary='fill', fillvalue=0)
DoG_y = convolve2d(gaussian_filter, Dy, mode='same', boundary='fill', fillvalue=0)


#derivative of gaussian filters 

grad_x_B = convolve2d(img_gray, DoG_x, mode='same', boundary='fill', fillvalue=0)
grad_y_B = convolve2d(img_gray, DoG_y, mode='same', boundary='fill', fillvalue=0)

magnitude_B = np.sqrt(grad_x_B**2 + grad_y_B**2)
edges_B = (magnitude_B > 0.18).astype(np.float32)

x_match = np.allclose(grad_x_A, grad_x_B, atol=1e-5)
y_match = np.allclose(grad_y_A, grad_y_B, atol=1e-5)


print(f"DoG X-Derivative Matches Two-Step Method: {x_match}")
print(f"DoG Y-Derivative Matches Two-Step Method: {y_match}")


plt.imsave('part1_3_gaussian_blur.png', img_blurred, cmap='gray')
plt.imsave('part1_3_dog_x_kernel.png', DoG_x, cmap='gray')
plt.imsave('part1_3_dog_y_kernel.png', DoG_y, cmap='gray')
plt.imsave('part1_3_dog_gradient_magnitude.png', magnitude_B, cmap='gray')
plt.imsave('part1_3_dog_binarized_edges.png', edges_B, cmap='gray')
