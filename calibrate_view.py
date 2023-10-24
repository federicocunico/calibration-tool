import json
import os
import argparse

import cv2
import numpy as np
from src.database import Database

from src.homography.homography_utils import get_h_from_images, transform
from src.opencv_utils import MousePointsClick, OpenCVWindow


def _to_filename(camera):
    return (
        camera.replace("\\", "")
        .replace(".", "")
        .replace("-", "")
        .replace("@", "")
        .replace(":", "")
        .replace("/", "")
    )


def calibrate(frame, pav_img, h_file, camera, max_num_pts=4):
    h, pts1, pts2 = get_h_from_images(frame, pav_img, num_rect_pts=max_num_pts)

    pts1_file = f"data/pts1__view_{_to_filename(camera)}.npy"
    pts2_file = f"data/pts2__view_{_to_filename(camera)}.npy"

    np.save(h_file, h)
    np.save(pts1_file, pts1)
    np.save(pts2_file, pts2)

    return h


def get_draw_pts(frame, num_pts=30):
    w = OpenCVWindow(f"Get max #{num_pts}")
    mpc = MousePointsClick(w, num_pts)
    mpc.get_points(w, frame, pts_size=20)
    draw_pts = np.asarray(mpc.points)
    return draw_pts


def transform(points, homography):
    if homography is not None:
        res = cv2.perspectiveTransform(
            np.asarray(points).reshape((-1, 1, 2)).astype(np.float32), homography
        )
        out_pts: np.ndarray = res.reshape(points.shape).astype(np.int)
        return out_pts
    return None


def main(args):

    # TODO

    camera = args.input
    floor = cv2.imread(args.plane)

    wname = "frame"
    wname_p = "floor"

    cap = cv2.VideoCapture(camera)

    assert cap.isOpened(), f'Unable to open the camera: "{camera}"!'

    ret, frame = cap.read()
    if not ret:
        raise NotImplementedError()

    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(wname, 512, 512)

    h_file = f"data/h__cam_{_to_filename(camera)}.npy"
    draw_pts = []

    if os.path.isfile(h_file):
        h = np.load(h_file)

    while True:
        cv2.imshow(wname, frame)
        cv2.imshow(wname_p, floor)
        key = cv2.waitKey(10)

        if key == ord("c"):
            h = calibrate(frame, floor, h_file, camera)
        elif key == ord("d"):
            draw_pts = get_draw_pts(frame)
            print(f"Points: {draw_pts}")
            _h, w = frame.shape[0], frame.shape[1]
            norm_pts = []
            for pt in draw_pts:
                norm_pts.append((pt[0] / _h, pt[1] / w))
        # Exit
        elif key == ord("q"):
            break

        # Draw in original image
        for pt in draw_pts:
            cv2.circle(
                frame,
                tuple(pt),
                20,
                (255, 0, 0),
                -1,
            )
        if len(draw_pts) > 0:
            center_out_pts = transform(draw_pts, h)
            # print(center_out_pts)
            for pt in center_out_pts:
                cv2.circle(
                    floor,
                    tuple(pt),
                    10,
                    (255, 0, 0),
                    -1,
                )

def read_db():
    with open(database_file, "r") as f:
        cameras = json.load(f)
    db = Database(**cameras)
    return db

def save_db(cameras):
    with open(database_file, "w") as f:
        json.dump(cameras, f, indent=4)

if __name__ == "__main__":
    database_file = "cameras.json"
    cameras = read_db()

    print("*" * 200)
    print("*")
    print(
        "Press C to calibrate, D to draw points in order to show the calibration accuracy"
    )
    print("*")
    print("*" * 200)

    main(cameras)
