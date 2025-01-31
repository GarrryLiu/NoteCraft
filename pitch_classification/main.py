import argparse
from blob_detector import detect_blobs
from getting_lines import get_staffs
from note import Note
import cv2
import os


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default="data/sample.jpg")

    return parser.parse_args()


def main():
    args = parse()
    image = cv2.imread(args.input)
    adjusted_photo = image
    staffs = get_staffs(adjusted_photo)
    blobs = detect_blobs(adjusted_photo, staffs)
    notes = Note.extract_notes(blobs, staffs, adjusted_photo)
    Note.draw_notes_pitch(adjusted_photo, notes)


if __name__ == "__main__":
    main()
