import tkinter as tk
from tkinter import Scrollbar
from src.database_utils import read_db
from src.database import Camera
from calibrate_view import calibrate_camera


database = read_db()
selected_camera_idx = None


def on_ok_button_click():
    selected_index = listbox.curselection()[0]  # Get the selected item's index
    selected_item: Camera = database.cameras[selected_index]  # Get the selected Camera object
    uri = selected_item.uri
    if uri:
        global selected_camera_idx
        selected_camera_idx = selected_index
        process_selected_uri(selected_index)
        # window.quit()


def process_selected_uri(camera_idx: int):
    # Replace this with the method to process the selected URI
    print("Selected URI:", database.cameras[camera_idx].uri)
    calibrate_camera(database, camera_idx)


print("Press C to calibrate, D to draw points in order to show the calibration accuracy, S to save")

# Create the main window
window = tk.Tk()
window.title("URI Selector")

# Create a Listbox with horizontal scrolling
listbox_frame = tk.Frame(window)
listbox_frame.pack()

listbox = tk.Listbox(listbox_frame, selectmode=tk.SINGLE)
listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# scrollbar = Scrollbar(listbox_frame, orient=tk.HORIZONTAL)
# listbox.config(xscrollcommand=scrollbar.set)
# scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

for camera in database.cameras:
    s = camera.uri
    s = s.split("@")[-1]
    # if "rtsp" not in s:
    #     s = "rtsp://" + s

    listbox.insert(tk.END, s)  # Add only the URI to the Listbox

# Create an "OK" button
ok_button = tk.Button(window, text="OK", command=on_ok_button_click)
ok_button.pack()

# Start the tkinter main loop
window.mainloop()

# if selected_camera_idx is not None:
#     process_selected_uri(selected_camera_idx)
# else:
#     print("No camera selected")
# print("Done")
