import time
import pickle
import sys

from bioviz import Viz

file_path = 0

if len(sys.argv) > 1:
    file_path = str(sys.argv[1])

if not isinstance(file_path, str):
    t = time.localtime(time.time())
    file_path = f"results/{t.tm_year}_{t.tm_mon}_{t.tm_mday}_upDown.bob"

with open(file_path, "rb") as file:
    data = pickle.load(file)

data_interpolate, _ = data["data"]

b = Viz(
    "../models/BrasViolon.bioMod",
    markers_size=0.0002,
    show_segments_center_of_mass=False,
    show_local_ref_frame=False,
    show_global_ref_frame=False,
    show_muscles=False,
)
b.load_movement(
    data_interpolate["q"].T,
)
b.exec()
