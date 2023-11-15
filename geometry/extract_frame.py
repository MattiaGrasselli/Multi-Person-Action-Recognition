# extract the first frame of a mp4 video and save it as jpg

import cv2
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, help="path to video file or camera id")
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        raise ValueError("Cannot open source file")
    
    ret, frame = cap.read()
    cv2.imwrite(args.output, frame)
    print(f"Frame saved to {args.output}")
    cap.release()
