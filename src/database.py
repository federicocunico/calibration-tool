from typing import List
from pydantic import BaseModel


class Camera(BaseModel):
    uri: str
    pav_img: str
    image_pts: List[List[int]] = []
    pav_pts: List[List[int]] = []
    homography: List[List[float]] = []


class Database(BaseModel):
    cameras: List[Camera] = []
