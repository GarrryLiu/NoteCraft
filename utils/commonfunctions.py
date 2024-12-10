from glob import glob

import cv2
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu


def read_data(data_path):
    """
    Read images from the specified path using glob.

    Parameters:
    - data_path: Path to the images.

    Returns:
    - imgs: List of loaded images.
    """
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
    """
    Compute the histogram of an image and apply a threshold.

    Parameters:
    - img: Input image.
    - thresh: Threshold value.

    Returns:
    - hist: Processed histogram.
    """
    hist = (np.ones(img.shape) - img).sum(dtype=np.int32, axis=1)
    _max = np.amax(hist)
    hist[hist[:] < _max * thresh] = 0
    return hist


def get_line_indices(hist):
    """
    Get indices of lines from the histogram.

    Parameters:
    - hist: Histogram of the image.

    Returns:
    - indices: List of indices where lines are detected.
    """
    indices = []
    prev = 0
    for index, val in enumerate(hist):
        if val > 0 and prev <= 0:
            indices.append(index)
        prev = val
    return indices


def otsu(img):
    """
    Apply Otsu's method with Gaussian filtering to binarize an image.

    Parameters:
    - img: Grayscale image.

    Returns:
    - Binary image after applying Otsu's thresholding.
    """
    blur = gaussian(img)
    otsu_bin = 255 * (blur > threshold_otsu(blur))
    return (otsu_bin / 255).astype(np.int32)


def get_thresholded(img, thresh):
    """
    Threshold an image based on a specified threshold.

    Parameters:
    - img: Input image.
    - thresh: Threshold value.

    Returns:
    - Binary image after thresholding.
    """
    return 1 * (img > thresh)
