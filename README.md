# Video Calibration Tool
Calibrate a video/image on a plane using homography.

## Requirements
- Python3.x
  - opencv-python
  - numpy

## Installation

If you are using conda:

```
conda create --name calibration python=3.8
conda activate calibration
```

otherwise, install your python venv (or use your system installation (r u sure tho?))

```
pip install opencv-python
```

## Run

Run with:

```
python3 calibrate_view.py /path/to/your/video.mp4 /path/to/your/image/plane.jpg
```

### Instructions
Press the following keys:
- "C" : calibrate the view, choose 4 points on the frame and those corresponding in the plane;
- "D" to draw up to 30 points at once to check the calibration. Press "Enter" if you want to stop before 30 points;

### Re-calibrate at every run?
No! If you already did your calibration, the calibration file will be loaded, which means you can visualize more points using the "D" command.

## Results
Results of the calibration will be saved in the data folder as:

```
data/h__cam_YOURCAMFILENAME.npy
```

aswell as the points used for the calibration:

```
data/pts1__view_YOURCAMFILENAME.npy
data/pts2__view_YOURCAMFILENAME.npy
```

