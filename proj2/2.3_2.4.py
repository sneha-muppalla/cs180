import cv2
import numpy as np
import matplotlib.pyplot as plt
from align_image_code import align_images
from PIL import Image

def gaussian_stack(img, n = 6):
    stack = [img]
    for i in range(n-1):
        stack.append(cv2.GaussianBlur(stack[-1], (0,0), 2*2**i))
    return stack 

def laplacian_stack(gaussian_stack):
    stack = []
    for i in range(len(gaussian_stack)-1):
        stack.append(gaussian_stack[i] - gaussian_stack[i+1])
    stack.append(gaussian_stack[-1])
    return stack

def get_mask(name, h, w):
    mask = np.zeros((h, w), np.float32)
    if name == "vertical":
        mask[:, :int(w*0.5)] = 1
    elif name == "horizontal":
        mask[:int(h*0.5), :] = 1
    elif name == "irregular":
        cv2.ellipse(mask, (239,310), (120,148), 0, 0, 360, 1, -1)
    return mask
img1 = np.array(Image.open("./spline/rdj_smile.jpg").convert("RGB"), dtype=np.float32)/255
img2 = np.array(Image.open("./spline/jack_black.jpg").convert("RGB"), dtype=np.float32)/255

img1, img2 = align_images(img1, img2)

g1 = gaussian_stack(img1)
g2 = gaussian_stack(img2)
l1 = laplacian_stack(g1)
l2 = laplacian_stack(g2)

mask = get_mask("irregular", img2.shape[0], img2.shape[1])
gm = gaussian_stack(mask)
 
# blend each level with its own blurred mask, then add them all up
result = 0
for i in range(len(l1)):
    m = gm[i][..., None]
    result = result + m*l1[i] + (1-m)*l2[i]
 
hard = mask[..., None]*img1 + (1-mask[..., None])*img2
 
plt.imsave("mask.png", mask, cmap="gray")
plt.imsave("hard_seam.png", np.clip(hard,0,1))
plt.imsave("multiresolution_blend.png", np.clip(result,0,1))


# for submission i just included the output images not images at every single level