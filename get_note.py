import cv2

from extract_symbol.extract_symbol import extract_symbol
from utils.commonfunctions import read_data

# Read images from the specified data path
imgs = read_data("./data/package_aa/000100134-10_1_1.png")

# Process each image in the list
for i, img in enumerate(imgs):
    saved_imgs = extract_symbol(img)

    # Save the split images to the output directory
    for j, row_images in enumerate(saved_imgs):
        for k, saved_img in enumerate(row_images):
            cv2.imwrite(f"./outputs/split_notes/processed_{i}_{j}_{k}.png", saved_img)
