import os
from glob import glob

import cv2
import numpy as np
import skimage.io as io
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu

from segmenter import Segmenter

# part 1

# read all images

# segment image for each line

# remove staff lines

input_files = glob("data/package_aa/*/*.png")  # input path

for i, input_file in enumerate(input_files):
    img = io.imread(input_file)
    gray_img = rgb2gray(img)
    _, binary_img = cv2.threshold(
        gray_img, threshold_otsu(gray_img), 1, cv2.THRESH_BINARY
    )
    segmenter = Segmenter(binary_img)
    imgs_with_staff = segmenter.regions_with_staff
    imgs_without_staff = segmenter.regions_without_staff

    # Define the output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    normalized_region = (imgs_with_staff[0] * 255).astype(np.uint8)
    output_path = os.path.join(output_dir, f"region_with_staff_{i}.png")
    io.imsave(output_path, normalized_region)

    normalized_region = (imgs_without_staff[0] * 255).astype(np.uint8)
    output_path = os.path.join(output_dir, f"region_without_staff_{i}.png")
    io.imsave(output_path, normalized_region)


# part 2

# based on dataset, train a model to detect
# Clefs
# Notes
# Rests
# Key Signature
# Time Signature
# Dynamics
# Articulation Marks
