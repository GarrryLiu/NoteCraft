import os
from glob import glob

import numpy as np
import skimage.io as io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

from commonfunctions import show_images
from segmenter import Segmenter


def get_gray(img):
    if img.ndim == 2:  # If the image is already grayscale
        return img
    elif img.ndim == 3 and img.shape[2] == 3:  # If the image is RGB
        gray = rgb2gray(np.copy(img))
        return gray
    else:
        raise ValueError("Input image must be a 2D grayscale or 3D RGB image.")


def get_thresholded(img, thresh):
    return 1 * (img > thresh)


def normalize_image(img):
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max - img_min > 0:
        normalized_img = (img - img_min) / (img_max - img_min) * 255
        return normalized_img.astype(np.uint8)
    return img


# Read all PNG files from the specified directory
input_files = glob("data/package_aa/*/*.png")

for input_file in input_files:
    img = io.imread(input_file)
    original = img.copy()
    gray = get_gray(img)
    bin_img = get_thresholded(gray, threshold_otsu(gray))
    segmenter = Segmenter(bin_img)
    imgs_with_staff = segmenter.regions_with_staff
    imgs_without_staff = segmenter.regions_without_staff

    # Define the output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    for i, img in enumerate(imgs_without_staff):
        # Normalize and save the processed image without staff lines
        normalized_without_staff = normalize_image(img)
        output_without_staff_path = os.path.join(
            output_dir,
            f"processed_without_staff_{os.path.basename(input_file).split('.')[0]}_{i}.png",
        )
        io.imsave(output_without_staff_path, normalized_without_staff)

        # # Normalize and save the original image with staff lines
        # normalized_with_staff = normalize_image(imgs_with_staff[i])
        # output_with_staff_path = os.path.join(
        #     output_dir,
        #     f"original_with_staff_{os.path.basename(input_file).split('.')[0]}_{i}.png",
        # )
        # io.imsave(output_with_staff_path, normalized_with_staff)
