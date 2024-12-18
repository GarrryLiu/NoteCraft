import cv2
import numpy as np
from config import SAVING_IMAGES_STEPS, VERBOSE


def preprocess_image(input_image):
    """
    Preprocess the input image: convert to grayscale, invert, and threshold.
    """
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY) if len(input_image.shape) == 3 else input_image
    _, im_inv = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    return im_inv


def remove_lines(image, line_orientation="horizontal"):
    """
    Remove horizontal or vertical lines from the binary image using morphological operations.
    :param image: Preprocessed binary image.
    :param line_orientation: "horizontal" or "vertical" to determine line removal direction.
    :return: Image with lines removed.
    """
    if line_orientation == "horizontal":
        kernel_height = max(1, int(image.shape[0] / 20))  # Kernel size based on image height
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))
        horizontal_lines = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        horizontal_lines = (255 - horizontal_lines)

        if SAVING_IMAGES_STEPS:
            cv2.imwrite("output/8a_lines_horizontal_removed.png", horizontal_lines)
        
        return horizontal_lines

    elif line_orientation == "vertical":
         kernel_width = max(1, int(image.shape[1] / 100))  # Kernel size based on image width
         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
         vertical_lines = cv2.morphologyEx(255 - horizontal_lines, cv2.MORPH_OPEN, kernel)
         vertical_lines = (255 - vertical_lines)
         
         return vertical_lines

         if SAVING_IMAGES_STEPS:
            cv2.imwrite("output/8a_lines_vertical_removed.png", vertical_lines)

    lines_removed = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    return 255 - lines_removed


def configure_blob_detector():
    """
    Configure the parameters for the blob detector.
    :return: A SimpleBlobDetector object.
    """
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


def assign_blobs_to_staffs(keypoints, staffs):
    """
    Assign blobs to the closest staff line using vertical bins.
    """
    staff_diff = 3 / 5 * (staffs[0].max_range - staffs[0].min_range)
    bins = [x for sublist in [[staff.min_range - staff_diff, staff.max_range + staff_diff] for staff in staffs]
            for x in sublist]
    bins = np.sort(bins)

    keypoints_staff = np.digitize([key.pt[1] for key in keypoints], bins)
    return sorted(list(zip(keypoints, keypoints_staff)), key=lambda tup: (tup[1], tup[0].pt[0]))


def annotate_and_save(image, notes, filename_prefix):
    """
    Annotate detected blobs with indices and staff numbers, then save the output images.
    """
    im_with_numbers = image.copy()
    for idx, (keypoint, staff_no) in enumerate(notes):
        position = (int(keypoint.pt[0]), int(keypoint.pt[1]))
        cv2.putText(im_with_numbers, str(idx), position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        cv2.putText(image, str(staff_no), position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

    if SAVING_IMAGES_STEPS:
        cv2.imwrite(f"output/{filename_prefix}_with_numbers.jpg", im_with_numbers)
        cv2.imwrite(f"output/{filename_prefix}_with_staff_numbers.jpg", image)


def detect_blobs(input_image, staffs):
    """
    Detect blobs (potential notes) in the given image and assign them to the respective staff lines.
    """
    if VERBOSE:
        print("Detecting blobs...")

    # Step 1: Preprocess image
    im_inv = preprocess_image(input_image)

    # Step 2: Remove horizontal and vertical lines
    horizontal_removed = remove_lines(im_inv, "horizontal")
    vertical_removed = remove_lines(horizontal_removed, "vertical")

    if SAVING_IMAGES_STEPS:
        cv2.imwrite("output/8a_lines_horizontal_removed.png", horizontal_removed)
        cv2.imwrite("output/8a_lines_vertical_removed.png", vertical_removed)

    # Step 3: Configure and detect blobs
    detector = configure_blob_detector()
    im_with_blobs = cv2.cvtColor(vertical_removed, cv2.COLOR_GRAY2BGR)
    keypoints = detector.detect(im_with_blobs)

    # Step 4: Draw blobs for debugging
    cv2.drawKeypoints(im_with_blobs, keypoints, im_with_blobs, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if SAVING_IMAGES_STEPS:
        cv2.imwrite("output/8b_with_blobs.jpg", im_with_blobs)

    # Step 5: Assign blobs to the closest staff
    sorted_notes = assign_blobs_to_staffs(keypoints, staffs)

    # Step 6: Annotate and save results
    annotate_and_save(im_with_blobs, sorted_notes, "8c")

    # Step 7: Print results
    if VERBOSE:
        print(f"Keypoints length: {len(keypoints)}")

    return sorted_notes
