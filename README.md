# **Music Sheet Analysis Tool**

---

## **Overview**
This project is a Python-based tool for analyzing music sheets. It detects staff lines, notes, and their pitches using image processing and computer vision techniques. The tool processes input music sheet images and outputs annotated results with identified notes and their corresponding pitches.

---

## **Features**
- **Staff Line Detection**: Detects and visualizes staff lines in the music sheet.
- **Blob Detection**: Identifies note heads (blobs) on the sheet and maps them to the respective staff lines.
- **Pitch Recognition**: Classifies each detected note as a pitch (e.g., C4, G3) based on its position and clef.
- **Clef Detection**: Distinguishes between violin and bass clefs using Hu moments.
- **Visual Outputs**: Saves intermediate and final results, including processed images and text annotations.

---

## **File Structure**
### **1. `main.py`**
- The entry point of the project. It processes the input image, coordinates staff detection, blob detection, note extraction, and pitch annotation.

### **2. `config.py`**
- Contains configuration parameters like thresholds, kernel sizes, and flags for verbose output and saving intermediate results.

### **3. `getting_lines.py`**
- Implements staff line detection:
  - Preprocessing the input image.
  - Detecting lines using the Hough Transform.
  - Grouping lines into staff structures.

### **4. `blob_detector.py`**
- Detects blobs (note heads) using `cv2.SimpleBlobDetector` and assigns them to staff lines.

### **5. `hu.py`**
- Implements clef detection:
  - Extracts clef regions from staff lines.
  - Classifies clefs as violin or bass using Hu moments.

### **6. `note.py`**
- Defines the `Note` class, which:
  - Calculates a note's position relative to staff lines.
  - Determines its pitch based on clef and position.
  - Contains methods to extract and annotate notes in the image.

### **7. `staff.py`**
- Defines the `Staff` class, representing a staff structure:
  - Calculates line locations and spacing for a given staff.

### **8. `util.py`**
- Contains utility functions, such as calculating distances between points.

### **9. `requirements.txt`**
- Lists all required dependencies for the project:
  - `opencv-python`
  - `numpy`
  - `argparse`

---

## **How to Use**
### **1. Install Dependencies**
Install the required packages using the following command:
```bash
pip install -r requirements.txt
```

### **2. Run the Tool**
Execute the main script with an input image:
```bash
python main.py -i <path_to_image>
```
Replace `<path_to_image>` with the path to your input music sheet image (e.g., `data/sample.png`).

### **3. View Results**
The outputs will be saved in the `output/` directory:
- **Intermediate Results**:
  - Processed images with lines or blobs removed.
  - Images with detected blobs and annotated staff numbers.
- **Final Outputs**:
  - Annotated image with detected notes and their pitches.
  - A text file (`notes_pitch.txt`) listing the detected notes and their pitches.

---

## **Key Parameters**
- **`config.py`**:
  - `VERBOSE`: Enables detailed logging.
  - `SAVING_IMAGES_STEPS`: Saves intermediate image processing results.
  - `THRESHOLD_MIN` and `THRESHOLD_MAX`: Threshold values for binarizing the image.
  - `LINES_DISTANCE_THRESHOLD`: Minimum distance between staff lines for grouping.

---

## **File Outputs**
- **Annotated Images**:
  - `output/`: Includes images for horizontal/vertical lines removed, detected blobs, and annotated notes with pitches.
- **Text File**:
  - `output/notes_pitch.txt`: Lists all detected notes and their pitches.

