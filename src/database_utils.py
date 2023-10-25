import json
import os
from src.database import Camera, Database

database_file = "cameras.json"


def read_db():
    if not os.path.isfile(database_file):
        print("Database file not found! Create a file with the name 'cameras.json' in the root directory of the project.")
        print("\nExample of the content:")
        print(json.dumps(Database(cameras=[Camera(uri="rtsp://<ip>/stream", pav_img="plan.jpg")]).model_dump(), indent=4))
        return Database()
    with open(database_file, "r") as f:
        cameras = json.load(f)
    db = Database(**cameras)
    return db


def save_db(db: Database):
    with open(database_file, "w") as f:
        json.dump(db.model_dump(), f, indent=4)
