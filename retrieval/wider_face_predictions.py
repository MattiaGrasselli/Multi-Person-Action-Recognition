"""
This script creates predictions for WIDER face evaluation on WIDER_val.
For the prediction files format, see the "Submissions" section of http://shuoyang1213.me/WIDERFACE/

To perform the evaluation, run the official matlab script (wider_eval.m), available at:
http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip
"""

import cv2
import dlib
import argparse
from tqdm import tqdm
from pathlib import Path

def convert_and_trim_bb(image, rect):
    """Convert the given bounding box from Dlib to OpenCV format"""

    start_x = rect.left()
    start_y = rect.top()
    end_x = rect.right()
    end_y = rect.bottom()

    # Ensure the bounding box coordinates fall within the image
    start_x = max(start_x, 0)
    start_y = max(start_y, 0)
    end_x = min(end_x, image.shape[1])
    end_y = min(end_y, image.shape[0])

    width = end_x - start_x
    height = end_y - start_y

    return (start_x, start_y, width, height)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wider-root', type=str, help='WIDER face root directory, e.g. WIDER_val/images')
    parser.add_argument('--dest-dir', type=str, default='./wider_predictions', help='Output directory')
    parser.add_argument('--use-bilateral', action='store_true', help='Use bilateral filter')
    parser.add_argument('--upsample', type=int, default=0, help='# of times to upsample')
    args = parser.parse_args()

    # Pre-trained HOG + Linear SVM face detector
    detector = dlib.get_frontal_face_detector()

    wider_root = Path(args.wider_root)
    dest_dir = Path(args.dest_dir)

    if not wider_root.exists():
        print(f"The directory {wider_root} does not exist")
        exit(1)

    if not dest_dir.exists():
        dest_dir.mkdir(parents=True)

    # Iteratively process all the jpeg files in the WIDER face root directory
    for img_path in tqdm(wider_root.glob('**/*.jpg')):
        img_rel_path = img_path.relative_to(wider_root)
        dest_subdir = dest_dir / img_rel_path.parent

        if not dest_subdir.exists():
            dest_subdir.mkdir(parents=True)

        print(f"Processing {img_path}...")
        img = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if args.use_bilateral:
            rgb = cv2.bilateralFilter(rgb, 5, 10, 10)

        dets, scores, w_indexes = detector.run(rgb, args.upsample)

        pred_file = dest_subdir / (img_path.stem + ".txt")
        with open(pred_file, "w") as f:
            # Prediction format: <image name without extension>NEWLINE<# of faces in the image>NEWLINE<face1 box>NW<face2 box>...
            # Box format: left, top, width, height, score
            f.write(img_path.stem + "\n")
            f.write(str(len(dets)) + "\n")

            for rect, score, index in zip(dets, scores, w_indexes):
                start_x, start_y, width, height = convert_and_trim_bb(rgb, rect)
                f.write(f"{start_x} {start_y} {width} {height} {score}\n")
