"""Utility to capture a frame from webcam."""
import cv2


if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

    img_counter = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break

        # frame = cv2.flip(frame, -1)
        cv2.imshow("frame", frame)

        k = cv2.waitKey(1)
        if k % 256 == 113:
            # q pressed
            print("Quitting...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = f"photo_{img_counter:02}.jpg"
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()
