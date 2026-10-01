# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.transform import rescale

# name of the input file
imname = './wharf.tif'

# read in the image as grayscale (the glass plate scan is stacked grayscale)
im = cv.imread(imname, cv.IMREAD_GRAYSCALE)

# convert to float in [0,1] (might want to do this later on to save memory)
im = im.astype(np.float32) / 255.0
    
# compute the height of each part (just 1/3 of total)
height = int(np.floor(im.shape[0] / 3.0))

# separate color channels
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]

# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)
def align(im1, im2):
    height, width = im2.shape

    # aligns image 1 onto image 2, if its small enough 

    if height < 900 or width < 900:
        #remove the borders and get interior like 15 to height - 15 to not affect the align score
        border_h = max(5, int(height * 0.1))
        border_w = max(5, int(width * 0.1))

        crop_im2 = im2[border_h: height - border_h, border_w: width - border_w]
        
        lowest_score = np.inf
        for dx in range(-50, 51):
            for dy in range (-50, 51):
                shifted_im1  = np.roll(im1, shift=(dy, dx), axis=(0,1))
                crop_im1 = shifted_im1[border_h: height - border_h, border_w: width - border_w]
                curr_score = np.sum((crop_im1 - crop_im2) ** 2)

                if curr_score < lowest_score:
                    lowest_score = curr_score
                    best_shift = (dy,dx)
        return best_shift

    # if image is too large, can size it down recursively
    resize_im1 = rescale(im1, 0.5, anti_aliasing=True, preserve_range=True)
    resize_im2 = rescale(im2, 0.5, anti_aliasing=True, preserve_range=True)

    next_shift = align(resize_im1, resize_im2)
    next_dy, next_dx = int(next_shift[0] * 2), int(next_shift[1] * 2)

    border_h = max(15, int(height * 0.1))
    border_w = max(15, int(width * 0.1))

    crop_im2 = im2[border_h: height - border_h, border_w: width - border_w]
    
    lowest_score = np.inf
    for dx in range(-5, 6):
        for dy in range (-5, 6):
            total_dy = next_dy + dy
            total_dx = next_dx + dx
            shifted_im1  = np.roll(im1, shift=(total_dy, total_dx), axis=(0,1))
            crop_im1 = shifted_im1[border_h: height - border_h, border_w: width - border_w]
            curr_score = np.sum((crop_im1 - crop_im2) ** 2)

            if curr_score < lowest_score:
                lowest_score = curr_score
                best_shift = (dy,dx)
    return (next_dy + best_shift[0], next_dx + best_shift[1])


ag = align(g, b)
offset = "Green offset:" + str(ag)
ag = np.roll(g, shift=ag, axis=(0,1))
ar = align(r, b)
offset += "Red offset:" + str(ar)
ar = np.roll(r, shift=ar, axis=(0,1))

# create a color image
im_out = np.dstack([ar, ag, b])

# display the image using matplotlib (expects RGB)
plt.figure(figsize=(8, 8))
plt.imshow(im_out)
plt.title('Colorized')
plt.axis('off')
plt.show()

print(offset)

# prepare for OpenCV saving/display (expects BGR uint8)
out_uint8 = np.clip(im_out * 255.0, 0, 255).astype(np.uint8)
out_bgr = cv.cvtColor(out_uint8, cv.COLOR_RGB2BGR)

# save the image
fname = './out_fname.jpg'
cv.imwrite(fname, out_bgr)

