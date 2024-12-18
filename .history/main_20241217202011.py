import argparse
from blob_detector import detect_blobs
from getting_lines import get_staffs
from note import Note
from photo_adjuster import adjust_photo
import cv2
import os


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        default='input/good/000100134-10_1_1.png'
    )

    return parser.parse_args()


def main():
    args = parse()
    image = cv2.imread(args.input)
    adjusted_photo = image
    staffs = get_staffs(adjusted_photo)
    blobs = detect_blobs(adjusted_photo, staffs)
    notes = Noteextract_notes(blobs, staffs, adjusted_photo)
    draw_notes_pitch(adjusted_photo, notes)


if __name__ == "__main__":
    main()
