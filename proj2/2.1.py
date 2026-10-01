import cv2
import numpy as np
from scipy.signal import convolve2d
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

#deriving unsharp masking technique
'''
subtract blurred version from the original image
I_h = I - (I*G)
scaling factor is a

I_s = I + a * I_h
= I + (a * (I - (I*G)))

I_s =I*[(1+a )* e - a*G)] , where e is a matrix for the number one
so the filter is then filter = (1+a )* e - a*G
    '''

def apply_filter(img, filter):
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        out[:, :, c] = convolve2d(img[:, :, c], filter, mode='same', boundary='symm')
    return out


def unsharp_masking(filter_size, sigma, alpha):

    # single convolution unmasked filter 
    gaussian_1d = cv2.getGaussianKernel(filter_size, sigma)
    G = np.outer(gaussian_1d, gaussian_1d.T)

    impulse = np.zeros((filter_size, filter_size))
    impulse[filter_size//2, filter_size//2] = 1
    
    return  (1 + alpha) * impulse - alpha * G

def sharpen_image(img, filter_size, sigma, alpha):
    plt.imsave('2.1_original.png', img)
    print("Saved: 2.1_original.png")

    gaussian_1d = cv2.getGaussianKernel(filter_size, sigma)
    blur_filter = np.outer(gaussian_1d, gaussian_1d.T)

    blurred = apply_filter(img, blur_filter)
    blurred = np.clip(blurred, 0.0, 1.0)
    high_freq = (img - blurred + 0.5)  # +0.5 to center signed values for display
    sharpened = (apply_filter(img, unsharp_masking(filter_size, sigma, alpha)))
    return blurred, high_freq, sharpened

    
# i had to search this online because for some reason the images i selected would like rotate idk why
img = ImageOps.exif_transpose(Image.open('chair.jpg'))
img_array = np.array(img.convert('RGB'), dtype=np.float32) / 255.0

for alpha in [0.5, 1, 2, 4]:
    taj_blurr, taj_high_freq, taj_sharpen = sharpen_image(
        img_array, filter_size=15, sigma=3, alpha=alpha
    )

    percent_clipped = 100 * np.mean(
        (taj_sharpen < 0) | (taj_sharpen > 1)
    )
    print(f"alpha={alpha}: {percent_clipped:.2f}% of channel values clipped")

    plt.imsave( f'chair_sharpened_alpha_{alpha}.png', np.clip(taj_sharpen, 0, 1))


plt.imsave('chair_blurred_output.png', taj_blurr, cmap='gray')
print("Saved: chair_blurred_output.png")

plt.imsave('chair_high_freq_output.png', np.clip(taj_high_freq, 0, 1))


img = ImageOps.exif_transpose(Image.open('car.jpg'))

original = np.array(img.convert('RGB'), dtype=np.float32) / 255.0
plt.imsave('car_original.png', original)

# sharpen og
_, _, sharpened = sharpen_image(
    original, filter_size=15, sigma=3, alpha=0.5
)
sharpened = np.clip(sharpened, 0, 1)
plt.imsave('car_sharpened.png', sharpened)

# Blur that thang
gaussian_1d = cv2.getGaussianKernel(15, 3)
blur_filter = gaussian_1d @ gaussian_1d.T
blurred = apply_filter(sharpened, blur_filter)
plt.imsave('car_sharpened_then_blurred.png', np.clip(blurred, 0, 1))

_, _, resharpened = sharpen_image(
    blurred, filter_size=15, sigma=3, alpha=1
)
plt.imsave('car_resharpened.png', np.clip(resharpened, 0, 1))