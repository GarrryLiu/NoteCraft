import cv2
import numpy as np
from config import *


def configure_blob_detector():

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

    return cv2.SimpleBlobDetector_create(params)


def detect_and_draw_blobs(image):

    detector = configure_blob_detector()
    keypoints = detector.detect(image)

    im_with_blobs = image.copy()
    cv2.drawKeypoints(
        im_with_blobs,
        keypoints,
        im_with_blobs,
        color=(0, 0, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return keypoints, im_with_blobs


def detect_blobs(input_image, staffs):
    """
    Detects blobs with given parameters.
    """
    if VERBOSE:
        print("Detecting blobs.")
    im_with_blobs = input_image.copy()

    im_inv = 255 - im_with_blobs
    gray = (
        cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        if len(input_image.shape) == 3
        else input_image
    )
    _, im_inv = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    kernel_height = max(1, int(im_inv.shape[0] / 20))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))

    horizontal_lines = cv2.morphologyEx(im_inv, cv2.MORPH_OPEN, kernel)
    horizontal_lines = 255 - horizontal_lines

    if SAVING_IMAGES_STEPS:
        cv2.imwrite("output/4lines_horizontal_removed.png", horizontal_lines)

    kernel_width = max(1, int(im_inv.shape[1] / 100))
    kernel = cv2.getStructuringElement(ksize=(kernel_width, 1), shape=cv2.MORPH_RECT)
    vertical_lines = cv2.morphologyEx(255 - horizontal_lines, cv2.MORPH_OPEN, kernel)
    vertical_lines = 255 - vertical_lines

    if SAVING_IMAGES_STEPS:
        cv2.imwrite("output/4lines_vertical_removed.png", vertical_lines)

    im_with_blobs = vertical_lines
    if len(im_with_blobs.shape) == 2:
        im_with_blobs = cv2.cvtColor(im_with_blobs, cv2.COLOR_GRAY2BGR)

    detector = configure_blob_detector()
    keypoints = detector.detect(im_with_blobs)

    cv2.drawKeypoints(
        im_with_blobs,
        keypoints=keypoints,
        outImage=im_with_blobs,
        color=(0, 0, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    if SAVING_IMAGES_STEPS:
        cv2.imwrite("output/5with_blobs.jpg", im_with_blobs)

    staff_diff = 3 / 5 * (staffs[0].max_range - staffs[0].min_range)
    bins = [
        x
        for sublist in [
            [staff.min_range - staff_diff, staff.max_range + staff_diff]
            for staff in staffs
        ]
        for x in sublist
    ]
    bins = np.sort(bins)
    keypoints_staff = np.digitize([key.pt[1] for key in keypoints], bins)
    sorted_notes = sorted(
        list(zip(keypoints, keypoints_staff)), key=lambda tup: (tup[1], tup[0].pt[0])
    )

    im_with_numbers = im_with_blobs.copy()

    for idx, tup in enumerate(sorted_notes):
        cv2.putText(
            im_with_numbers,
            str(idx),
            (int(tup[0].pt[0]), int(tup[0].pt[1])),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 0, 0),
        )
        cv2.putText(
            im_with_blobs,
            str(tup[1]),
            (int(tup[0].pt[0]), int(tup[0].pt[1])),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 0, 0),
        )
    if SAVING_IMAGES_STEPS:
        cv2.imwrite("output/8c_with_numbers.jpg", im_with_numbers)
        cv2.imwrite("output/8d_with_staff_numbers.jpg", im_with_blobs)

    if VERBOSE:
        print("Keypoints length : " + str(len(keypoints)))

    return sorted_notes
