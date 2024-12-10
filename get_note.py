import cv2
import numpy as np
from skimage.color import rgb2gray

from extract_symbol.extract_symbol import split_symbol
from remove_staff_line.remove import remove
from utils.commonfunctions import read_data
from utils.pre_processing import IsHorizontal

# Read images from the specified data path
imgs = read_data("./data/package_aa/000100134-10_1_1.png")

# Process each image in the list
for i, img in enumerate(imgs):
    # Convert the image to grayscale
    gray_img = rgb2gray(img)

    # Remove staff lines and get processed images
    imgs_with_staff, imgs_without_staff, segmenter = remove(gray_img)

    # Prepare the first image for horizontal check
    img = (imgs_with_staff[0] * 255).astype(np.uint8)

    # Check if the image is horizontal
    horizontal = IsHorizontal(img)

    # Split the symbols from the processed images
    saved_imgs = split_symbol(
        imgs_with_staff, imgs_without_staff, segmenter, horizontal
    )

    # Save the split images to the output directory
    for j, row_images in enumerate(saved_imgs):
        for k, saved_img in enumerate(row_images):
            cv2.imwrite(f"./outputs/split_notes/processed_{i}_{j}_{k}.png", saved_img)
