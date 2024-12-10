from glob import glob

import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu


def read_data(data_path):
    imgs = []
    input_files = glob(data_path)  # input path
    for input_file in input_files:
        img = cv2.imread(input_file)
        if img is not None:
            imgs.append(img)
        else:
            print(f"Warning: Could not read image {input_file}")

    return imgs


def histogram(img, thresh):
    hist = (np.ones(img.shape) - img).sum(dtype=np.int32, axis=1)
    _max = np.amax(hist)
    hist[hist[:] < _max * thresh] = 0
    return hist


def get_line_indices(hist):
    indices = []
    prev = 0
    for index, val in enumerate(hist):
        if val > 0 and prev <= 0:
            indices.append(index)
        prev = val
    return indices


def otsu(img):
    """
    Otsu with gaussian
    img: gray image
    return: binary image, pixel values 0:1
    """
    blur = gaussian(img)
    otsu_bin = 255 * (blur > threshold_otsu(blur))
    return (otsu_bin / 255).astype(np.int32)


def get_thresholded(img, thresh):
    return 1 * (img > thresh)
