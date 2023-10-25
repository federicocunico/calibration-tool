import json
import os
import argparse

import cv2
import numpy as np
from src.database_utils import save_db
from src.database import Camera, Database

from src.homography.homography_utils import get_h_from_images, transform
from src.opencv_utils import MousePointsClick, OpenCVWindow


def calibrate(frame, pav_img, camera: Camera, max_num_pts=4):
    h, pts1, pts2 = get_h_from_images(frame, pav_img, num_rect_pts=max_num_pts)

    camera.homography = h.tolist()
    camera.image_pts = pts1.astype(int).tolist()
    camera.pav_pts = pts2.astype(int).tolist()

    return h


def get_draw_pts(frame, num_pts=30):
    w = OpenCVWindow(f"Get max #{num_pts}")
    mpc = MousePointsClick(w, num_pts)
    mpc.get_points(w, frame, pts_size=20)
    draw_pts = np.asarray(mpc.points)
    return draw_pts


def transform(points, homography):
    if homography is not None:
        res = cv2.perspectiveTransform(np.asarray(points).reshape((-1, 1, 2)).astype(np.float32), homography)
        out_pts: np.ndarray = res.reshape(points.shape).astype(np.int32)
        return out_pts
    return None


def calibrate_camera(database: Database, camera_idx: int):
    camera = database.cameras[camera_idx]

    assert os.path.isfile(camera.pav_img), f'Unable to open the floor image: "{camera.pav_img}"!'

    camera_uri = camera.uri
    floor = cv2.imread(camera.pav_img)

    wname = "frame"
    wname_p = "floor"

    cap = cv2.VideoCapture(camera_uri)

    assert cap.isOpened(), f'Unable to open the camera: "{camera_uri}"!'

    ret, frame = cap.read()
    if not ret:
        raise NotImplementedError()

    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(wname, 512, 512)

    draw_pts = []

    h = np.asarray(camera.homography)
    orig_frame = frame.copy()
    orig_floor = floor.copy()

    while True:
        cv2.imshow(wname, frame)
        cv2.imshow(wname_p, floor)
        key = cv2.waitKey(10)

        if key == ord("c"):
            h = calibrate(orig_frame, orig_floor, camera)
        if key == ord("s"):
            # save!
            save_db(database)

        elif key == ord("d"):
            # reset frame for new points
            frame = orig_frame.copy()
            floor = orig_floor.copy()

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
        if len(draw_pts) > 0 and h is not None and h.shape[0] > 0:
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

    cv2.destroyAllWindows()
    cap.release()


# if __name__ == "__main__":
#     print("*" * 200)
#     print("*")
#     print("Press C to calibrate, D to draw points in order to show the calibration accuracy, S to save")
#     print("*")
#     print("*" * 200)

#     main(cameras)
