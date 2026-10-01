from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

OUTPUT = "part2_3_2_4_rdj_on_jackblack"
OUTPUT.mkdir(exist_ok=True)

def save_image(path, img):
    plt.imsave(path, np.clip(img, 0, 1), cmap="gray")


def blur(img, sigma):
    size = int(np.ceil(6 * sigma)) | 1
    g = cv2.getGaussianKernel(size, sigma)
    return cv2.sepFilter2D(img, -1, g, g, borderType=cv2.BORDER_REFLECT)


def gaussian_stack(img, levels=6, sigma=2):
    stack = [img.copy()]

    for i in range(levels - 1):
        stack.append(blur(stack[-1], sigma * (2 ** i)))

    return np.stack(stack)


def laplacian_stack(g):
    result = []

    for i in range(len(g) - 1):
        result.append(g[i] - g[i + 1])

    result.append(g[-1])
    return np.stack(result)


def make_mask(shape, kind="vertical", ellipse=None):
    h, w = shape[:2]
    y, x = np.mgrid[:h, :w]

    if kind == "vertical":
        mask = x < w // 2

    elif kind == "horizontal":
        mask = y < h // 2

    elif kind == "ellipse":
        cx, cy, rx, ry = ellipse
        mask = (((x - cx*w) / (rx*w)) ** 2 +
                ((y - cy*h) / (ry*h)) ** 2) <= 1

    return mask.astype(np.float32)


def blend(image1, image2, mask):
    g1 = gaussian_stack(image1)
    g2 = gaussian_stack(image2)
    gm = gaussian_stack(mask)

    l1 = laplacian_stack(g1)
    l2 = laplacian_stack(g2)

    mask = gm[..., None]

    a = mask * l1
    b = (1 - mask) * l2

    combined = a + b
    result = np.sum(combined, axis=0)

    return g1, g2, gm, l1, l2, a, b, combined, result


def save_stack(stack, folder, name, signed=False):
    fig, axes = plt.subplots(1, len(stack), figsize=(15, 3))

    for i in range(len(stack)):
        img = stack[i]

        if signed and i < len(stack) - 1:
            img = img + 0.5

        save_image(folder / f"{name}_{i}.png", img)

        axes[i].imshow(np.clip(img, 0, 1), cmap="gray")
        axes[i].set_title(f"Level {i}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(folder / f"{name}_overview.png")
    plt.close()


def save_process(a, b, combined, folder):
    levels = [0, 2, 4]

    fig, axes = plt.subplots(4, 3, figsize=(10, 12))

    for row in range(3):
        i = levels[row]

        panels = [a[i], b[i], combined[i]]

        for col in range(3):
            img = panels[col]
            scale = max(np.max(np.abs(img)), 1e-8)

            axes[row, col].imshow(
                np.clip(0.5 + img / (2 * scale), 0, 1)
            )
            axes[row, col].axis("off")

    imgs = [a.sum(axis=0), b.sum(axis=0), combined.sum(axis=0)]

    for i in range(3):
        axes[3, i].imshow(np.clip(imgs[i], 0, 1))
        axes[3, i].axis("off")

    plt.tight_layout()
    plt.savefig(folder / "blending_process.png")
    plt.close()


def run_example(path1, path2, name, mask_kind="vertical",
                ellipse=None, face_points=None):

    folder = OUTPUT / name
    folder.mkdir(exist_ok=True)

    image1 = np.array(Image.open(path1).convert("RGB"), dtype=np.float32) / 255
    image2 = np.array(Image.open(path2).convert("RGB"), dtype=np.float32) / 255

    # align the faces
    if face_points is not None:
        source, destination = face_points

        M = cv2.getAffineTransform(
            np.float32(source),
            np.float32(destination)
        )

        image1 = cv2.warpAffine(
            image1, M,
            (image2.shape[1], image2.shape[0]),
            borderMode=cv2.BORDER_REFLECT
        )

    if image1.shape != image2.shape:
        image2 = cv2.resize(
            image2,
            (image1.shape[1], image1.shape[0])
        )

    mask = make_mask(image1.shape, mask_kind, ellipse)

    g1, g2, gm, l1, l2, a, b, combined, result = blend(
        image1, image2, mask
    )

    save_image(folder / "input_a.png", image1)
    save_image(folder / "input_b.png", image2)
    save_image(folder / "mask.png", mask)

    hard = mask[..., None] * image1 + (1 - mask[..., None]) * image2

    save_image(folder / "hard_seam.png", hard)
    save_image(folder / "multiresolution_blend.png", result)

    save_stack(g1, folder, "gaussian_a")
    save_stack(g2, folder, "gaussian_b")
    save_stack(gm, folder, "gaussian_mask")
    save_stack(l1, folder, "laplacian_a", True)
    save_stack(l2, folder, "laplacian_b", True)
    save_stack(combined, folder, "blended_laplacian", True)

    save_process(a, b, combined, folder)

    print("Saved:", name)


run_example(
    BASE / "cs180_proj2_hybrid_starter_code" / "rdj_smile.jpg",
    BASE / "cs180_proj2_hybrid_starter_code" / "jack_black.jpeg",
    "rdj_smile_on_jack_black",
    mask_kind="ellipse",
    ellipse=(239/489, 310/628, 120/489, 148/628),
    face_points=(
        [(1049,1295), (1565,1284), (1300,1840)],
        [(175,274), (302,268), (240,395)]
    )
)