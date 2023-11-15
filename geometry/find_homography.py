"""
Compute homography matrix from 9 points on the image and 9 corresponding points in the real world.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

"""
Select 9 points, from left to right, top to bottom, on the image.
"""

UNIT_LENGTH_M = 0.7 #0.50 #0.345

num_points = 9
points = []

# Callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", img_copy)

# Create a file dialog to choose an image
root = tk.Tk()
root.withdraw()  # Hide the main window
image_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])

if not image_path:
    print("No image selected. Exiting...")
    exit()

img = cv2.imread(image_path)
img_copy = img.copy()

# Create a window and set mouse callback
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

# Display the image and wait for points to be selected, then wait for ESC
cv2.imshow("Image", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_points = np.array(points)

print("Selected points:")
print(img_points)

world_points = np.array(
    [
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1],
        [0, 2], [1, 2], [2, 2]
    ]
) * UNIT_LENGTH_M

# Find homography matrix
H, _ = cv2.findHomography(img_points, world_points)
H = H / H[2, 2]
np.save("homography_01.npy", H)
