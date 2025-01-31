import cv2
import numpy as np
from numpy import linalg
from config import *


def get_clef(image, staff, extra_down_factor=0.3):

    i = 0
    image_height, image_width = image.shape[:2]
    gray_image = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    )
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)

    window_height = staff.lines_location[-1] - staff.lines_location[0]
    up = max(0, staff.lines_location[0] - int(window_height * 0.2))
    down = min(
        image_height,
        staff.lines_location[-1] + int(window_height * (0.2 + extra_down_factor)),
    )

    key_width = int(window_height * 1.5)

    while i + key_width < image_width:
        window = binary_image[up:down, i : i + key_width]
        black_pixel_ratio = np.sum(window == 0) / window.size

        if black_pixel_ratio > 0.05:
            print("Clef detected!")
            break

        i += int(key_width / 8)

    if i + key_width >= image_width:
        print("No clef detected!")
        return None

    clef_window = binary_image[up:down, i : i + key_width]
    cv2.imwrite("output/3clef.png", clef_window)

    return clef_window


def hu_moments():

    violin_key = cv2.imread("clef_samples/violin_clef.png", 0)
    bass_key = cv2.imread("clef_samples/bass_clef2.png", 0)
    violin_moment = cv2.HuMoments(cv2.moments(violin_key)).flatten()
    bass_moment = cv2.HuMoments(cv2.moments(bass_key)).flatten()
    return log_transform_hu(violin_moment), log_transform_hu(bass_moment)


def log_transform_hu(hu_moment):
    return -np.sign(hu_moment) * np.log10(np.abs(hu_moment))


def classify_clef(image, staff):
    original_clef = get_clef(image, staff)

    v_moment, b_moment = hu_moments()
    v_moment = v_moment[:3]
    b_moment = b_moment[:3]

    original_moment = cv2.HuMoments(cv2.moments(original_clef)).flatten()
    original_moment = log_transform_hu(original_moment)
    original_moment = original_moment[:3]

    if linalg.norm(v_moment - original_moment) < linalg.norm(
        b_moment - original_moment
    ):
        return "violin"
    else:
        return "bass"
