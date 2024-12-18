# **Music Sheet Analysis Tool**

---

## **Functions**
- **Crop Staves**: Crop an input sheet music image into a set of individual lines of staves.
- **Note Recognition**: Identifies types of notes and signatures present in the line.
- **Pitch Recognition**: Classifies each detected note as a pitch (e.g., C4, G3) based on its position and clef.
- **Midi Translation**: Translates the extracted note, signature, and pitch information into MIDI format.

---

## **Code Packages**
### **1. `crop_staves`**
- Code for cropping staves into individual lines of music for downstream recognition and classification.

### **2. `remove_staff_line`**
- Code for removing staff lines.

### **3. `pitch_classification`**
- Code for determining the pitch of notes.

### **4. `extract_symbol` and `recognition`**
- Code for note and signature classification.
---

## **Installing Dependencies**
Install the required packages using the following command:
```bash
pip install -r requirements.txt
```

## **Note and Signature Recognition**

Execute `get_results.py` after modifying the `image_path` in place in the code.
```bash
python get_results.py
```

## **Pitch Recognition**

### **1. Run the Tool**
Execute the main script with an input image:
```bash
python pitch_classification/main.py -i <path_to_image>
```
Replace `<path_to_image>` with the path to your input music sheet image (e.g., `data/sample.png`).

### **2. View Results**
The outputs will be saved in the `pitch_classification/output/` directory:
- **Intermediate Results**:
  - Processed images with lines or blobs removed.
  - Images with detected blobs and annotated staff numbers.
- **Final Outputs**:
  - Annotated image with detected notes and their pitches.
  - A text file (`notes_pitch.txt`) listing the detected notes and their pitches.

---

### **Key Parameters**
- **`config.py`**:
  - `VERBOSE`: Enables detailed logging.
  - `SAVING_IMAGES_STEPS`: Saves intermediate image processing results.
  - `THRESHOLD_MIN` and `THRESHOLD_MAX`: Threshold values for binarizing the image.
  - `LINES_DISTANCE_THRESHOLD`: Minimum distance between staff lines for grouping.

---

### **File Outputs**
- **Annotated Images**:
  - `output/`: Includes images for horizontal/vertical lines removed, detected blobs, and annotated notes with pitches.
- **Text File**:
  - `output/notes_pitch.txt`: Lists all detected notes and their pitches.

## **Info2MIDI Translation**

Execute `info2midi.py` for an example of translation. Format the information in the format of the example, and call `create_midi` to execute the code on a custom file.