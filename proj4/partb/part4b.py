from harris import get_harris_corners
import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path
from PIL import Image, ImageOps  
from cv2 import GaussianBlur
from skimage.color import rgb2gray

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "parta"))

from part4a import computeH, mosaic, save_image


#part b.1 :)
def anms(coords, harris_response, c_robust=0.9, num_points=500):
    points = coords.T 
    strong = harris_response[points[:, 0], points[:, 1]]
    radii = np.full(len(points), np.inf)

    for i in range(len(points)):
        stronger = points[strong[i] < c_robust*strong]

        if len(stronger) > 0:
            distances = np.sum((stronger - points[i]) ** 2, axis=1)
            radii[i] = np.min(distances)
    order = sorted(
    range(len(radii)), key=lambda i: (radii[i], strong[i]), reverse=True
    )
    return points[order[:num_points]]

def feature_extractor(gray, points):
    blurred = GaussianBlur(gray, (0,0), sigmaX=2)

    features = []
    valid_points = []

    for y, x in points:
        if y < 20 or x < 20:
            continue 
        if y + 20 > gray.shape[0] or x + 20 > gray.shape[1]:
            continue 

        patch = blurred[y-20:y+20, x-20:x+20] #40x40
        descriptor = patch[2::5, 2::5] # at index 2 every fifth row + column
        mean = descriptor.mean()
        std = descriptor.std()

        if std < 1e-6: # dont want no uniform patches
            continue 
        #normalize
        descriptor = (descriptor - mean) / std
        features.append(descriptor)
        valid_points.append((y, x))
    # (N, 8,8) and (N,2) should be the shapes of the returned arrays
    return np.array(features).reshape(-1,8,8), np.array(valid_points).reshape(-1,2)

# so find similar descriptors between the images
def match_features(features1, features2, threshold=0.7):
    desc1 = features1.reshape(len(features1), 64) # flattens to 64
    desc2 = features2.reshape(len(features2), 64)
    matches = []

    if len(desc2) < 2:
        return np.empty((0, 2), dtype=int)

    for i, descriptor in enumerate(desc1): #euclidean distance -> sort smallest to largest-> closest and second closest kept
        distances = np.linalg.norm(desc2 - descriptor, axis=1)
        nearest = np.argsort(distances)
        best, second = nearest[:2]

        if distances[second] > 0:
            ratio = distances[best] / distances[second]

            if ratio < threshold:
                matches.append((i, best))

    return np.array(matches, dtype=int).reshape(-1, 2)

def ransac(matched1, matched2, iterations=2000, threshold=3):
    # switch (y, x) to (x, y), have the first image go onto the second image
    source = matched1[:, ::-1].astype(float)
    target = matched2[:, ::-1].astype(float)

    if len(source) < 4:
        return "less than four"

    rng = np.random.default_rng(0)
    best_inliers = np.zeros(len(source), dtype=bool) #boolean array false
    source_h = np.column_stack([source, np.ones(len(source))]) #[x,y,1] -> [x,y,1] for transformation

    for _ in range(iterations): #select four random
        indices = rng.choice(len(source), 4, replace=False)
        '''
        valid = True
        for points in [source[indices], target[indices]]:
            for a, b, c in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]:
                u = points[b] - points[a]
                v = points[c] - points[a]
                area = abs(u[0] * v[1] - u[1] * v[0])

                if area < 1e-6: #dont want three of the points to be  a straight line collinear
                    valid = False

        if not valid:
            continue
        commented out for now this is just to check less valid samples
        '''
        H = computeH(source[indices], target[indices])

        projected = source_h @ H.T # same as part a, transform using H
        finite = np.abs(projected[:, 2]) > 1e-10

        errors = np.full(len(source), np.inf)
        mapped = projected[finite, :2] / projected[finite, 2:3]
        errors[finite] = np.linalg.norm(mapped - target[finite], axis=1) #error

        inliers = errors < threshold

        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    if best_inliers.sum() < 4:
        return "no valid homography"

    # final H using all matches in the best consensus.
    H = computeH(source[best_inliers], target[best_inliers])
    return H, best_inliers

# all the testing and results
image = Image.open(Path(__file__).resolve().parent.parent / "parta/images/set1_1.png").convert("RGB")
image.thumbnail((1200, 1200))
image = np.array(image) / 255.0
gray = rgb2gray(image)

# b1
h, coords = get_harris_corners(gray)
points = anms(coords, h, num_points=500)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

axes[0].imshow(image)
axes[0].scatter(coords[1], coords[0], c="lime", s=5)
axes[0].set_title(f"Harris: {coords.shape[1]} corners")

axes[1].imshow(image)
axes[1].scatter(points[:, 1], points[:, 0], c="lime", s=8)
axes[1].set_title(f"After ANMS: {len(points)} corners")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig("harris_anms.png", dpi=150)
plt.show()

# b2
descriptors, descriptor_points = feature_extractor(gray, points)

fig, axes = plt.subplots(2, 6, figsize=(12, 4))

for ax in axes.flat:
    ax.axis("off")

for ax, descriptor, point in zip(
    axes.flat, descriptors[:12], descriptor_points[:12]
):
    ax.imshow(descriptor, cmap="gray",
              vmin=-2, vmax=2, interpolation="nearest")
    y, x = point
    ax.set_title(f"({x}, {y})", fontsize=9)

plt.tight_layout()
plt.savefig("descriptors.png", dpi=150)
plt.show()


# b3
folder = Path(__file__).resolve().parent.parent / "parta/images"

image1 = Image.open(folder / "set1_1.png").convert("RGB")
image2 = Image.open(folder / "set1_2.png").convert("RGB")

image1.thumbnail((1200, 1200))
image2.thumbnail((1200, 1200))

image1 = np.array(image1) / 255.0
image2 = np.array(image2) / 255.0

gray1 = rgb2gray(image1)
gray2 = rgb2gray(image2)

h1, coords1 = get_harris_corners(gray1)
h2, coords2 = get_harris_corners(gray2)

corners1 = anms(coords1, h1, num_points=500)
corners2 = anms(coords2, h2, num_points=500)

features1, points1 = feature_extractor(gray1, corners1)
features2, points2 = feature_extractor(gray2, corners2)

matches = match_features(features1, features2)
matched1 = points1[matches[:, 0]]
matched2 = points2[matches[:, 1]]

fig, ax = plt.subplots(figsize=(14, 7))

height = max(image1.shape[0], image2.shape[0])
offset = image1.shape[1]

canvas = np.zeros((height, offset + image2.shape[1], 3))
canvas[:image1.shape[0], :offset] = image1
canvas[:image2.shape[0], offset:] = image2
ax.imshow(canvas)

#  (y, x) order
for (y1, x1), (y2, x2) in zip(matched1, matched2):
    ax.plot([x1, x2 + offset], [y1, y2],
            color="lime", marker="o", markersize=3,
            linewidth=0.5, alpha=0.6)

ax.axis("off")
plt.savefig("feature_matches.png", dpi=150, bbox_inches="tight")
plt.show()

H, inliers = ransac(matched1, matched2)

print("Matches:", len(matched1))
print("RANSAC inliers:", inliers.sum())
print("H:\n", H)

result = mosaic([image2, image1], [np.eye(3), H])
save_image("automatic_mosaic.png", result)

folder = Path(__file__).resolve().parent.parent / "parta/images"
output = Path(__file__).resolve().parent / "results"
output.mkdir(exist_ok=True)

pairs = [
    ("kitchen_1_2", "set1_1.png", "set1_2.png"),
    ("church_2_3", "church_2.JPG", "church_3.JPG"),
    ("room_1_3", "set2_1.png", "set2_3.png"),
]

for name, file1, file2 in pairs:
    # Load the pair.
    images = []

    for filename in [file1, file2]:
        image = ImageOps.exif_transpose(Image.open(folder / filename)).convert("RGB")
        image.thumbnail((1200, 1200))
        images.append(np.array(image) / 255.0)

    image1, image2 = images

    features = []
    points = []

    for image in images:
        gray = rgb2gray(image)
        h, coords = get_harris_corners(gray)
        corners = anms(coords, h, num_points=500)
        descriptors, valid_points = feature_extractor(gray, corners)

        features.append(descriptors)
        points.append(valid_points)

    matches = match_features(features[0], features[1])
    matched1 = points[0][matches[:, 0]]
    matched2 = points[1][matches[:, 1]]

    H, inliers = ransac(matched1, matched2)

    print(name)
    print("Matches:", len(matches))
    print("Inliers:", inliers.sum())
    print("H:\n", H)

    # Show matches before RANSAC and its selected inliers.
    height = max(image1.shape[0], image2.shape[0])
    offset = image1.shape[1]

    canvas = np.zeros((height, offset + image2.shape[1], 3))
    canvas[:image1.shape[0], :offset] = image1
    canvas[:image2.shape[0], offset:] = image2

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    selections = [
        np.ones(len(matches), dtype=bool),
        inliers
    ]

    titles = [
        f"Descriptor matches: {len(matches)}",
        f"RANSAC inliers: {inliers.sum()}"
    ]

    for ax, selected, title in zip(axes, selections, titles):
        ax.imshow(canvas)

        for (y1, x1), (y2, x2) in zip(
            matched1[selected], matched2[selected]
        ):
            ax.plot(
                [x1, x2 + offset], [y1, y2],
                color="lime", marker="o",
                markersize=3, linewidth=0.5, alpha=0.6
            )

        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output / f"{name}_matches.png", dpi=150)
    plt.close(fig)

    result = mosaic([image2, image1], [np.eye(3), H])

    save_image(output / f"{name}_automatic.png", result)
    np.savetxt(output / f"{name}_H.txt", H)

    manual_points = []

    for i, image in enumerate(images):
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image)
        ax.set_title(
            f"{name}: click 8 matching points on image {i + 1}\n"
            "Use the same order in both images"
        )

        clicked = np.array(plt.ginput(8, timeout=0))
        plt.close(fig)

        if len(clicked) != 8:
            raise ValueError("Select exactly 8 points in each image.")

        manual_points.append(clicked)

    manual_H = computeH(manual_points[0], manual_points[1])

    np.savez(
        output / f"{name}_manual_points.npz",
        source=manual_points[0],
        target=manual_points[1],
        H=manual_H
    )

    manual_result = mosaic([image2, image1], [np.eye(3), manual_H])
    save_image(output / f"{name}_manual.png", manual_result)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(manual_result)
    axes[0].set_title("Manual correspondences")
    axes[1].imshow(result)
    axes[1].set_title("Automatic correspondences + RANSAC")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        output / f"{name}_comparison.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.close(fig)
