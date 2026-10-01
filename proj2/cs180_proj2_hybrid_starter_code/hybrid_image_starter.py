import matplotlib.pyplot as plt
from align_image_code import align_images
import numpy as np
import cv2

# First load images

# low freq
im1 = plt.imread('./jack_black.jpeg') / 255.

# high freq
im2 = plt.imread('./jack_blacksmile.jpg') / 255.

# Next align images (this code is provided, but may be improved)
im1_aligned, im2_aligned = align_images(im1, im2)

## You will provide the code below. Sigma1 and sigma2 are arbitrary 
## cutoff values for the high and low frequencies
def gaussian_blur(image, sigma):
    ksize = int(np.ceil(6 * sigma)) 
    gaussian_1d = cv2.getGaussianKernel(ksize, sigma)
    G = gaussian_1d @ gaussian_1d.T

    out = np.zeros_like(image)
    for c in range(image.shape[2]):
        out[:,:,c] = cv2.filter2D(image[:,:,c], -1, G)

    return out

def compute_fft(img):
    img_gray = np.mean(img, axis=2)
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(img_gray))))

def hybrid_image(im_low, im_high, sigma1, sigma2):
    low_img = gaussian_blur(im_low, sigma1)
    high_img = im_high - gaussian_blur(im_high, sigma2)
    hybrid_img = np.clip(low_img + high_img, 0.0, 1.0)
    return hybrid_img, low_img, high_img

sigma1 = 8
sigma2 = 2
hybrid, low_freq, high_freq = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)

plt.imshow(hybrid)
plt.show()

'''
print("\n--- Exporting Required Images via plt.imsave ---")

# 1. Save aligned base inputs
plt.imsave('2.2_input1_aligned_wolf.png', im1_aligned)
plt.imsave('2.2_input2_aligned_husky.png', im2_aligned)

# 2. Save Filtered Versions (Intermediate steps for your favorite result documentation)
plt.imsave('2.2_low_pass_filtered_wolf.png', np.clip(low_freq, 0.0, 1.0))
# Center high frequency around 0.5 so it displays perfectly instead of clipping to black
plt.imsave('2.2_high_pass_filtered_husky.png', np.clip(high_freq + 0.5, 0.0, 1.0))

# 3. Save Final Hybrid Result
plt.imsave('2.2_hybrid_output_wolf_husky.png', hybrid)
print("Saved baseline image results successfully.")
'''

# 4. Save Fourier Transform Log Magnitudes as Grayscale Images
# (These match the formula explicitly stated in your project spec!)
plt.imsave('2.2_fft_input1.png', compute_fft(im1_aligned), cmap='gray')
plt.imsave('2.2_fft_input2.png', compute_fft(im2_aligned), cmap='gray')
plt.imsave('2.2_fft_high_pass.png', compute_fft(high_freq), cmap='gray')
plt.imsave('2.2_fft_low_pass.png', compute_fft(low_freq), cmap='gray')
plt.imsave('2.2_fft_hybrid.png', compute_fft(hybrid), cmap='gray')
print("Saved all 5 Fourier Transform frequency charts successfully.")

print("\nExecution complete! Check your project folder for all the new '.png' files.")
