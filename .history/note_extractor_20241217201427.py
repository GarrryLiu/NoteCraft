import cv2
import os
from config import NOTE_PITCH_DETECTION_MIDDLE_SNAPPING, VERBOSE
from hu import classify_clef
from util import distance
from note import Note


violin_key = {
    -6: 'C6',
    -5: 'D6',
    -4: 'C6',
    -3: 'H5',
    -2: 'A5',
    -1: 'G5',
    0:  'F5',
    1:  'E5',
    2:  'D5',
    3:  'C5',
    4:  'H4',
    5:  'A4',
    6:  'G4',
    7:  'F4',
    8:  'E4',
    9:  'D4',
    10: 'C4',
    11: 'H3',
    12: 'A3',
    13: 'G3',
    14: 'F3',
}

bass_key = {
    -6: 'G3',
    -5: 'F3',
    -4: 'E3',
    -3: 'D3',
    -2: 'C3',
    -1: 'H3',
    0:  'A3',
    1:  'G3',
    2:  'F3',
    3:  'E3',
    4:  'D3',
    5:  'C3',
    6:  'H2',
    7:  'A2',
    8:  'G2',
    9:  'F2',
    10: 'E2',
    11: 'D2',
    12: 'C2',
    13: 'H1',
    14: 'A1',
}


def extract_notes(blobs, staffs, image):
    clef = classify_clef(image, staffs[0])
    notes = []
    if VERBOSE:
        print('Detected clef: ' + clef)
        print('Extracting notes from blobs.')
    for blob in blobs:
        if blob[1] % 2 == 1:
            staff_no = int((blob[1] - 1) / 2)
            notes.append(Note(staff_no, staffs, blob[0], clef))
    if VERBOSE:
        print('Extracted ' + str(len(notes)) + ' notes.')
    return notes


def draw_notes_pitch(image, notes):
    im_with_pitch = image.copy()
    if len(im_with_pitch.shape) == 2: 
        im_with_pitch = cv2.cvtColor(im_with_pitch, cv2.COLOR_GRAY2BGR)
    elif len(im_with_pitch.shape) == 3 and im_with_pitch.shape[2] == 3: 
        print("Image is already BGR, skipping conversion.")
    for note in notes:
        cv2.putText(im_with_pitch, note.pitch, (int(note.center[0]) - 5, int(note.center[1]) + 35),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.5, color=(255, 0, 0))
    cv2.imwrite('output/9_with_pitch.png', im_with_pitch)
    output_dir = 'output'
    text_file_path = os.path.join(output_dir, 'notes_pitch.txt')
    with open(text_file_path, 'w') as text_file:
        for note in notes:
            text_file.write(f"{note.pitch}\n")


