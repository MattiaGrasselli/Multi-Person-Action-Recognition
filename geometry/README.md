# Geometry Component

## Installation
```bash
conda create -n geometry_comp python=3.9
```
```
pip3 install -r requirements.txt
```

## Usage
1. `photo.py`: Capture an image of a squared pattern on the ground plane, pressing the `SPACE` button. Make sure that the pattern consists in at least 9 markers disposed in a 3x3 grid.
If you plan to use a pre-recorded video as input use `extract_frame.py` instead 
2. `find_homography.py`: Set `UNIT_LENGTH_M` to the real-world length of a side of a cell of the pattern, in meters. Then run the script and select the nine markers in the image (row major traversal order) to compute the homography matrix.![01](docs/markers_screen_01.png)![chess](docs/markers_screen_chess.png)
3. `distance.py`: Detect people in video frames and compute interpersonal distance. The `--source` argument allows to choose between a video file or a webcam as input.