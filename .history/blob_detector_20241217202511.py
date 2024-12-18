import cv2
import numpy as np
from config import *


def detect_blobs(input_image, staffs):
    """
    Detects blobs (potential notes) in the given image and assigns them to the respective staff lines.
    
    :param input_image: The input music sheet image.
    :param staffs: Detected staff objects containing range information.
    :return: Sorted list of blobs with corresponding staff numbers.
    """
    if VERBOSE:
        print("Detecting blobs.")
        
    # Step 1: Preprocess image - invert and threshold
    im_inv = (255 - input_image.copy())
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY) if len(input_image.shape) == 3 else input_image
    _, im_inv = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Step 2: Remove horizontal lines using morphology
    kernel_height = max(1, int(im_inv.shape[0] / 20))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))
    horizontal_lines = cv2.morphologyEx(im_inv, cv2.MORPH_OPEN, kernel)
    horizontal_lines = (255 - horizontal_lines)

    if SAVING_IMAGES_STEPS:
        cv2.imwrite("output/8a_lines_horizontal_removed.png", horizontal_lines)

    # Step 3: Remove vertical lines using morphology
    kernel_width = max(1, int(im_inv.shape[1] / 100))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    vertical_lines = cv2.morphologyEx(255 - horizontal_lines, cv2.MORPH_OPEN, kernel)
    vertical_lines = (255 - vertical_lines)

    if SAVING_IMAGES_STEPS:
        cv2.imwrite("output/8a_lines_vertical_removed.png", vertical_lines)

    # Step 4: Blob detection configuration
    im_with_blobs = cv2.cvtColor(vertical_lines, cv2.COLOR_GRAY2BGR) if len(vertical_lines.shape) == 2 else vertical_lines
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 225
    params.maxArea = 1500
    params.filterByCircularity = True
    params.minCircularity = 0.2
    params.filterByConvexity = True
    params.minConvexity = 0.9
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Step 5: Detect blobs (keypoints)
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(im_with_blobs)

    # Draw detected blobs
    cv2.drawKeypoints(im_with_blobs, keypoints, im_with_blobs, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if SAVING_IMAGES_STEPS:
        cv2.imwrite("output/8b_with_blobs.jpg", im_with_blobs)

    # Step 6: Assign blobs to corresponding staffs
    staff_diff = 3 / 5 * (staffs[0].max_range - staffs[0].min_range)
    bins = [
