import cv2
import numpy as np
from config import CONFIG

class BlobDetector:
    @staticmethod
    def detect(input_image, staffs):
        if CONFIG["verbose"]:
            print("Detecting blobs.")
        im_inv = (255 - input_image)
        gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY) if len(input_image.shape) == 3 else input_image
        _, im_inv = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

        # Remove horizontal and vertical lines
        horizontal_lines = BlobDetector._remove_lines(im_inv, axis='horizontal')
        vertical_lines = BlobDetector._remove_lines(255 - horizontal_lines, axis='vertical')

        keypoints = BlobDetector._detect_keypoints(vertical_lines)
        return BlobDetector._sort_notes_by_staff(keypoints, staffs)

    @staticmethod
    def _remove_lines(image, axis):
        kernel_size = max(1, image.shape[1] // 100) if axis == 'vertical' else max(1, image.shape[0] // 20)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 1) if axis == 'vertical' else (1, kernel_size))
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    @staticmethod
    def _detect_keypoints(image):
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea, params.minArea, params.maxArea = True, 225, 1500
        detector = cv2.SimpleBlobDetector_create(params)
        return detector.detect(image)

    @staticmethod
    def _sort_notes_by_staff(keypoints, staffs):
        staff_diff = 3 / 5 * (staffs[0].max_range - staffs[0].min_range)
        bins = [staff.min_range - staff_diff for staff in staffs] + [staff.max_range + staff_diff for staff in staffs]
        return sorted([(key, np.digitize(key.pt[1], bins)) for key in keypoints], key=lambda tup: (tup[1], tup[0].pt[0]))
