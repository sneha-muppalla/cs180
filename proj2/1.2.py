import numpy as np
from scipy.signal import convolve2d
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


img = Image.open('cameraman.png')
img_gray = np.array(img.convert('L'), dtype=np.float32) / 255.0
print(f"Image successfully loaded. Shape: {img_gray.shape}")


Dx = np.array([[-1, 1]])
Dy = np.array([[-1], [ 1]])

grad_x = convolve2d(img_gray, Dx, mode='same', boundary='fill')
grad_y = convolve2d(img_gray, Dy, mode='same', boundary='fill')

gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
THRESHOLD = 0.19
edges = (gradient_magnitude > THRESHOLD).astype(np.float32)

# Save partial derivatives
plt.imsave('part1_2_dx.png', grad_x, cmap='gray')

plt.imsave('part1_2_dy.png', grad_y, cmap='gray')

# Save full gradient magnitude
plt.imsave('part1_2_gradient_magnitude.png', gradient_magnitude, cmap='gray')

# Save final thresholded binary edge map
plt.imsave('part1_2_edges.png', edges, cmap='gray')
img = Image.open('cameraman.png')
img_gray = np.array(img.convert('L'), dtype=np.float32) / 255.0

