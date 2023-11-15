import argparse
import torch
import cv2
import os
import dlib
import numpy as np
from torch.nn import functional as F
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm
import time


INPUT_SIZE = (160, 160)
EMBEDDING_SIZE = 512
VIS_SIZE = (100, 100)
VIDEO_SIZE = (2000, 800)


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


def prewhiten(x):
    mean = x.mean()
    std = x.std()
    std_adj = std.clamp(min=1.0 / (float(x.numel()) ** 0.5))
    y = (x - mean) / std_adj
    return y


def plot_similarity_grid(cos_similarity, input_size):
    H, W = cos_similarity.shape
    rows = []
    for i in range(H):
        row = []
        for j in range(W):
            # create small colorful image from value in distance matrix
            value = cos_similarity[i][j]
            cell = np.empty(input_size)
            cell.fill(value)
            cell = (cell * 255).astype(np.uint8)
            # color depends on value: blue is closer to 0, green is closer to 1
            img = cv2.applyColorMap(cell, cv2.COLORMAP_WINTER)

            # add distance value as text centered on image
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"{value:.2f}"
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            text_x = (img.shape[1] - textsize[0]) // 2
            text_y = (img.shape[0] + textsize[1]) // 2
            cv2.putText(
                img,
                text,
                (text_x, text_y),
                font,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            row.append(img)
        rows.append(np.concatenate(row, axis=1))
    grid = np.concatenate(rows)
    return grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="source")
    parser.add_argument("--blacklist-dir", type=str, help="blacklist directory")
    parser.add_argument(
        "--output",
        type=str,
        default=f"runs/output_{int(time.time())}.mp4",
        help="path to output file",
    )
    # Upsampling nededed for small bboxes. Set to 0 for no upsampling.
    parser.add_argument(
        "--upsample", type=int, default=1, help="n. of times to upsample"
    )
    parser.add_argument(
        "--similarity-th",
        type=float,
        default=0.5,
        help="threshold for face recognition",
    )

    args = parser.parse_args()

    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        raise ValueError("Cannot open source file")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load face detector
    detector = dlib.get_frontal_face_detector()

    model = InceptionResnetV1(pretrained="vggface2").to(device)
    model.eval()

    blacklist_raw_images = []
    blacklist_processed_images = []

    if not os.path.exists(args.blacklist_dir):
        print("No blacklist directory provided, starting with an empty blacklist")
    else:
        for filename in os.listdir(args.blacklist_dir):
            frame = cv2.imread(os.path.join(args.blacklist_dir, filename))
            bboxes = detector(frame, args.upsample)

            if not bboxes:
                continue

            bbox = bboxes[0]
            x, y, w, h = convert_and_trim_bb(frame, bbox)
            bl_img = frame[y : y + h, x : x + w]
            blacklist_raw_images.append(bl_img)

            blacklist_processed_images.append(
                prewhiten(
                    torch.from_numpy(cv2.resize(bl_img, INPUT_SIZE))
                    .permute(2, 0, 1)
                    .float()[[2, 1, 0], :, :]
                )
            )

        blacklist_input = torch.stack(blacklist_processed_images, dim=0)
        blacklist_embeddings = F.normalize(model(blacklist_input), dim=1)

    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    count = 0

    # define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        args.output,
        fourcc,
        20.0,
        VIDEO_SIZE
    )

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            bboxes = detector(frame, args.upsample)

            face_raw_images = []
            face_processed_images = []

            dest_frame = frame.copy()

            for bbox in bboxes:
                x, y, w, h = convert_and_trim_bb(frame, bbox)
                face_img = frame[y : y + h, x : x + w]
                face_raw_images.append(face_img)

                face_processed_images.append(
                    prewhiten(
                        torch.from_numpy(cv2.resize(face_img, INPUT_SIZE))
                        .permute(2, 0, 1)
                        .float()[[2, 1, 0], :, :]
                    )
                )
                cv2.rectangle(dest_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if len(face_processed_images) == 0:
                pbar.update(1)
                count += 1
                continue

            face_input = torch.stack(face_processed_images, dim=0)
            face_embeddings = F.normalize(model(face_input), dim=1)

            cos_similarity = torch.matmul(face_embeddings, blacklist_embeddings.T)

            # clip similarity values to [0, 1] range to plot with a non diverging colormap
            similarity_grid = plot_similarity_grid(
                cos_similarity.clip(min=0, max=1), VIS_SIZE
            )

            # pad similarity grid with images of bboxes
            horizontal_grid = np.hstack(
                [cv2.resize(t, VIS_SIZE) for t in blacklist_raw_images]
            )
            zeros = np.zeros((*VIS_SIZE, 3))
            vertical_grid = np.vstack(
                [zeros, *[cv2.resize(t, VIS_SIZE) for t in face_raw_images]]
            )
            result = np.vstack([horizontal_grid, similarity_grid])
            result = np.hstack([vertical_grid, result])

            # cv2.imshow("Similarity", result.astype(np.uint8))

            dest_frame = cv2.resize(dest_frame, (960, 540))
            # cv2.imshow("Detections", dest_frame)

            cat = np.empty((*VIDEO_SIZE[::-1], 3), dtype=np.uint8)
            cat[: dest_frame.shape[0], : dest_frame.shape[1], :] = dest_frame
            cat[
                : result.shape[0],
                dest_frame.shape[1] : (dest_frame.shape[1] + result.shape[1]),
                :,
            ] = result
            cv2.imshow("Retrieval", cat)
            out.write(cat)

            # quit if user presses 'q' key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            pbar.update(1)
            count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
