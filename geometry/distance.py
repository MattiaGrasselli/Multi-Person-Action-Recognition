import os
import cv2
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import time


CONFIDENCE_TH = 0.5

def pairwise_distance(world_points):
    dist = np.linalg.norm(
        world_points[:, np.newaxis, :] - world_points[np.newaxis, :, :], # (N, 1, 2), (1, N, 2) -> (N, N, 2)
        ord=2, axis=-1)

    return dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="path to video file or camera id")
    parser.add_argument("--homography", type=str, default="homography.npy", help="path to homography matrix")
    parser.add_argument("--model-dir", type=str, default="yolov7", help="path to model directory")
    parser.add_argument("--model-name", type=str, default="custom", help="name of the model (specified in hubconf.py)")
    parser.add_argument("--model-weights", type=str, default="weights/best.pt", help="path to model weights")
    parser.add_argument("--output", type=str, default=f"runs/output_{int(time.time())}.mp4", help="path to output file")
    args = parser.parse_args()

    # load homography matrix (img_points -> real_points)
    H = np.load(args.homography)

    # read frames from source
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)
        
    if not cap.isOpened():
        raise ValueError("Cannot open source file")
    
    # define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, 20.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # load yolov7 custom weights
    model = torch.hub.load(
        args.model_dir, args.model_name, path_or_model=args.model_weights, source="local"
    )
    model.eval()

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            warped = cv2.warpPerspective(frame, H, dsize=(frame.shape[1], frame.shape[0]))

            # detect people
            detections = model(frame)

            filtered_detections = detections.xyxy[0]
            filtered_detections = filtered_detections[(filtered_detections[:,4] > CONFIDENCE_TH) & (filtered_detections[:, 5] == 0)]

            N = len(filtered_detections)

            if N:
                # get N different colors from a colormap
                # colors_mp = mpl.colormaps["hsv"](np.linspace(0, 1, N))
                # colors_cv = [(int(c[2] * 255), int(c[1] * 255), int(c[0] * 255)) for c in colors_mp]

                img_points = np.zeros((N, 2), dtype=np.float32)

                for i, det in enumerate(filtered_detections):
                    x1, y1, x2, y2 = det[:4].to(torch.int32).tolist()
                    img_points[i, 0] = (x1 + x2) / 2
                    img_points[i, 1] = y2

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                world_points = cv2.perspectiveTransform(img_points.reshape(1, N, 2), H).reshape(N, 2)

                dist = pairwise_distance(world_points)

                for i in range(N):
                    for j in range(i):
                        cv2.line(
                            frame,
                            (int(img_points[i, 0]), int(img_points[i, 1])),
                            (int(img_points[j, 0]), int(img_points[j, 1])),
                            (0, 255, 0),
                            2,
                        )
                        # write in the middle of the line the distance
                        cv2.putText(
                            frame,
                            "{:.2f}".format(dist[i, j]),
                            (
                                int((img_points[i, 0] + img_points[j, 0]) // 2),
                                int((img_points[i, 1] + img_points[j, 1]) // 2),
                            ),
                            cv2.FONT_HERSHEY_DUPLEX,
                            1,
                            (255, 255, 255),
                            2,
                        )

            cat = np.hstack((
                cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2)),
                cv2.resize(warped, (warped.shape[1] // 2, warped.shape[0] // 2))
            ))
            cv2.imshow("curr", cat)

            out.write(frame)

            # check if the user wants to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()