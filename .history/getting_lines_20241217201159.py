import cv2
import numpy as np
from config import *
from staff import Staff


def preprocess_image(image):

    if VERBOSE:
        print("Preprocessing image.")
    gray = image.copy()
    _, thresholded = cv2.threshold(gray, THRESHOLD_MIN, THRESHOLD_MAX, cv2.THRESH_BINARY)
    element = np.ones((3, 3))
    thresholded = cv2.erode(thresholded, element)
    edges = cv2.Canny(thresholded, 10, 100, apertureSize=3)
    return edges, thresholded


def detect_lines(hough, image, nlines):

    if VERBOSE:
        print("Detecting lines.")
    all_lines = set()
    height, width = image.shape[:2]
    lines_image_color = image

    for result_arr in hough[:nlines]:
        rho = result_arr[0][0]
        theta = result_arr[0][1]
        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a * rho
        y0 = b * rho
        shape_sum = width + height
        x1 = int(x0 + shape_sum * (-b))
        y1 = int(y0 + shape_sum * a)
        x2 = int(x0 - shape_sum * (-b))
        y2 = int(y0 - shape_sum * a)

        start = (x1, y1)
        end = (x2, y2)
        diff = y2 - y1
        if abs(diff) < LINES_ENDPOINTS_DIFFERENCE:
            all_lines.add(int((start[1] + end[1]) / 2))
            cv2.line(lines_image_color, start, end, (0, 0, 255), 2)

    if SAVING_IMAGES_STEPS:
        cv2.imwrite("output/5lines.png", lines_image_color)

    return all_lines, lines_image_color


def detect_staffs(all_lines):

    if VERBOSE:
        print("Detecting staffs.")
    staffs = []
    lines = []
    all_lines = sorted(all_lines)
    for current_line in all_lines:
        if lines and abs(lines[-1] - current_line) > LINES_DISTANCE_THRESHOLD:
            if len(lines) >= 5:
                staffs.append((lines[0], lines[-1]))
            lines.clear()
        lines.append(current_line)

    if len(lines) >= 5:
        if abs(lines[-2] - lines[-1]) <= LINES_DISTANCE_THRESHOLD:
            staffs.append((lines[0], lines[-1]))
    return staffs


def draw_staffs(image, staffs):

    width = image.shape[0]
    for staff in staffs:
        cv2.line(image, (0, staff[0]), (width, staff[0]), (0, 255, 255), 2)
        cv2.line(image, (0, staff[1]), (width, staff[1]), (0, 255, 255), 2)
    if SAVING_IMAGES_STEPS:
        cv2.imwrite("output/6staffs.png", image)


def get_staffs(image):
    """
    Returns a list of Staff
    :param image: image to get staffs from
    :return: list of Staff
    """
    processed_image, thresholded = preprocess_image(image)
    hough = cv2.HoughLines(processed_image, 1, np.pi / 150, 200)
    all_lines, lines_image_color = detect_lines(hough, thresholded, 80)
    staffs = detect_staffs(all_lines)
    draw_staffs(lines_image_color, staffs)
    return [Staff(staff[0], staff[1]) for staff in staffs]
