from skimage.feature import canny, corner_harris
from skimage.transform import (
    hough_line,
    hough_line_peaks,
    probabilistic_hough_line,
    rotate,
)

from utils.commonfunctions import *


def deskew(image):
    edges = canny(image, low_threshold=50, high_threshold=150, sigma=2)
    harris = corner_harris(edges)
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    h, theta, d = hough_line(harris, theta=tested_angles)
    out, angles, d = hough_line_peaks(h, theta, d)
    rotation_number = np.average(np.degrees(angles))
    if rotation_number < 45 and rotation_number != 0:
        rotation_number += 90
    return rotation_number


def rotation(img, angle):
    """
    Rotate the image by the given angle.
    Args:
        img: Input image
        angle: Angle to rotate in degrees
    Returns:
        Rotated image
    """
    # Check if angle is valid (not NaN)
    if np.isnan(angle):
        print("Warning: Invalid rotation angle (NaN). Skipping rotation.")
        return img

    # Ensure angle is within reasonable bounds
    angle = float(angle)
    if abs(angle) > 45:  # If angle is too large, it's likely an error
        print(f"Warning: Large rotation angle ({angle}°) detected. Skipping rotation.")
        return img

    try:
        image = rotate(img, angle, resize=True, mode="edge")
        return image
    except ValueError as e:
        print(f"Error during rotation: {e}")
        return img


def get_closer(img):
    rows = []
    cols = []
    for x in range(16):
        no = 0
        for col in range(x * img.shape[0] // 16, (x + 1) * img.shape[0] // 16):
            for row in range(img.shape[1]):
                if img[col][row] == 0:
                    no += 1
        if no >= 0.01 * img.shape[1] * img.shape[0] // 16:
            rows.append(x * img.shape[0] // 16)
    for x in range(16):
        no = 0
        for row in range(x * img.shape[1] // 16, (x + 1) * img.shape[1] // 16):
            for col in range(img.shape[0]):
                if img[col][row] == 0:
                    no += 1
        if no >= 0.01 * img.shape[0] * img.shape[1] // 16:
            cols.append(x * img.shape[1] // 16)
    new_img = img[
        rows[0] : min(img.shape[0], rows[-1] + img.shape[0] // 16),
        cols[0] : min(img.shape[1], cols[-1] + img.shape[1] // 16),
    ]
    return new_img


def IsHorizontal(img):
    projected = []
    rows, cols = img.shape
    for i in range(rows):
        proj_sum = 0
        for j in range(cols):
            if img[i][j] == 0:
                proj_sum += 1
        projected.append([1] * proj_sum + [0] * (cols - proj_sum))
        if proj_sum >= 0.9 * cols:
            return True
    return False
