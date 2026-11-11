"""Redo only the church 2/3 manual points and comparison."""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent / 'parta'))
from part4a import computeH, mosaic, save_image

name = 'church_2_3'
output = root / 'results'
images = []
for filename in ['church_2.JPG', 'church_3.JPG']:
    image = ImageOps.exif_transpose(Image.open(root.parent / 'parta/images' / filename)).convert('RGB')
    image.thumbnail((1200, 1200))
    images.append(np.array(image) / 255.0)

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(images[0])
ax.set_title('Church image 2: click 8 shared features in a memorable order')
source = np.array(plt.ginput(8, timeout=0))
plt.close(fig)
if len(source) != 8:
    raise ValueError('Need exactly 8 points; previous results have not been changed.')

# Keep the numbered source points visible while selecting their matches.
fig, axes = plt.subplots(1, 2, figsize=(15, 8))
axes[0].imshow(images[0])
axes[0].scatter(source[:, 0], source[:, 1], c='lime')
for i, (x, y) in enumerate(source, 1):
    axes[0].text(x + 5, y - 5, str(i), color='yellow')
axes[0].set_title('Image 2: your numbered points')
axes[1].imshow(images[1])
axes[1].set_title('Image 3: click the same features in order 1–8')
plt.sca(axes[1])
print('Click only in the right image. Right-click undoes the last point.')
target = np.array(plt.ginput(8, timeout=0))
plt.close(fig)
if len(target) != 8:
    raise ValueError('Need exactly 8 points; previous results have not been changed.')

H = computeH(source, target)
# Save coordinates first so they survive an interrupted warp.
np.savez(output / f'{name}_manual_points.npz', source=source, target=target, H=H)
manual = mosaic([images[1], images[0]], [np.eye(3), H])
save_image(output / f'{name}_manual.png', manual)

automatic = plt.imread(output / f'{name}_automatic.png')
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, image, title in zip(axes, [manual, automatic],
                            ['Manual correspondences', 'Automatic correspondences + RANSAC']):
    ax.imshow(image)
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.savefig(output / f'{name}_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Updated church_2_3 manual points, manual mosaic, and comparison.')
