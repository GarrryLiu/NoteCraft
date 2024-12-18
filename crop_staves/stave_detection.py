import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os

def detect_staves_and_save(image_path, output_dir, vertical_expansion=20):
    """
    Detect staves from a sheet music image, save the output with bounding boxes, a JSON file, and cropped images.
    
    Parameters:
    - image_path: Path to the input sheet music image.
    - output_dir: Directory to save the output files.
    - vertical_expansion: Number of pixels to expand each bounding box vertically.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define paths for output files
    output_image_path = os.path.join(output_dir, "detected_staves.png")
    output_json_path = os.path.join(output_dir, "bounding_boxes.json")
    cropped_images_dir = os.path.join(output_dir, "cropped_staves")
    os.makedirs(cropped_images_dir, exist_ok=True)

    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not open or find the image.")

    # Apply binary thresholding
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    # Detect horizontal lines for staves
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # Detect vertical lines for left and right boundaries
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    detected_vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Find contours of the detected lines
    contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vertical_contours, _ = cv2.findContours(detected_vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract vertical boundaries (left and right) based on vertical contours
    left_boundary = min([cv2.boundingRect(c)[0] for c in vertical_contours])
    right_boundary = max([cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in vertical_contours])

    # Store the bounding boxes of individual lines
    line_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h < 10:  # Filter out anything not likely to be a stave line
            line_boxes.append((x, y, w, h))

    # Group lines into staves based on vertical proximity
    line_boxes.sort(key=lambda box: box[1])  # Sort by the y-coordinate
    stave_boxes = []
    current_stave = [line_boxes[0]]

    for i in range(1, len(line_boxes)):
        # Check the vertical distance between the current line and the previous one
        if line_boxes[i][1] - current_stave[-1][1] < 20:  # Adjust threshold if needed
            current_stave.append(line_boxes[i])
        else:
            # Compute the bounding box for the current stave
            y_min = min(line[1] for line in current_stave) - vertical_expansion
            y_max = max(line[1] + line[3] for line in current_stave) + vertical_expansion
            y_min = max(0, y_min)  # Ensure y_min is within image bounds
            y_max = min(image.shape[0], y_max)  # Ensure y_max is within image bounds

            stave_boxes.append({"left": left_boundary, "top": y_min, "width": right_boundary - left_boundary, "height": y_max - y_min})
            current_stave = [line_boxes[i]]

    # Add the last stave
    if current_stave:
        y_min = min(line[1] for line in current_stave) - vertical_expansion
        y_max = max(line[1] + line[3] for line in current_stave) + vertical_expansion
        y_min = max(0, y_min)  # Ensure y_min is within image bounds
        y_max = min(image.shape[0], y_max)  # Ensure y_max is within image bounds

        stave_boxes.append({"left": left_boundary, "top": y_min, "width": right_boundary - left_boundary, "height": y_max - y_min})

    # Draw the bounding boxes for staves on the original image
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for idx, box in enumerate(stave_boxes):
        x, y, w, h = box["left"], box["top"], box["width"], box["height"]
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the image for the current stave
        cropped_stave = image[y:y + h, x:x + w]
        cropped_stave_path = os.path.join(cropped_images_dir, f"stave_{idx + 1}.png")
        cv2.imwrite(cropped_stave_path, cropped_stave)

    # Save the output image
    cv2.imwrite(output_image_path, output_image)

    # Save the bounding boxes to a JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(stave_boxes, json_file, indent=4)

    print(f"Output image saved to {output_image_path}")
    print(f"Bounding boxes saved to {output_json_path}")
    print(f"Cropped stave images saved in {cropped_images_dir}")

if __name__ == "__main__":
    detect_staves_and_save(
        "data/AudioLabs_v2/Beethoven_Op026-01/img/Beethoven_Op026-01_000.png",
        "outputs"
    )