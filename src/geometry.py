import cv2
import numpy as np
import math as m


def rot_homography(origin, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    # use 2x3 affine transformation for free
    return np.r_[cv2.getRotationMatrix2D(origin, -angle, 1.0), [[0.0, 0.0, 1.0]]]


def trans_homograpy(shift):
    identity = np.eye(3)
    np.put(identity, [2, 5], np.asarray(shift, dtype=identity.dtype))
    return identity


def Rx(theta):
    return np.matrix([[1, 0, 0],
                     [0, m.cos(theta), -m.sin(theta)],
                     [0, m.sin(theta), m.cos(theta)]])


def Ry(theta):
    return np.matrix([[m.cos(theta), 0, m.sin(theta)],
                     [0, 1, 0],
                     [-m.sin(theta), 0, m.cos(theta)]])


def Rz(theta):
    return np.matrix([[m.cos(theta), -m.sin(theta), 0],
                     [m.sin(theta), m.cos(theta), 0],
                     [0, 0, 1]])
