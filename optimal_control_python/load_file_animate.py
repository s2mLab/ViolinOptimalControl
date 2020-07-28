import time
import pickle
import sys

from BiorbdViz import BiorbdViz

file_path = "results/xia 5 phases/2020_7_25_upDown_single.bob"

if len(sys.argv) > 1:
    file_path = str(sys.argv[1])

if not isinstance(file_path, str):
    t = time.localtime(time.time())
    file_path = f"results/{t.tm_year}_{t.tm_mon}_{t.tm_mday}_upDown_interpolate.bob"

with open(file_path, "rb") as file:
    data = pickle.load(file)

data_interpolate, _ = data["data"]

b = BiorbdViz("../models/BrasViolon.bioMod", markers_size=0.001)
b.load_movement(data_interpolate["q"].T, )
b.exec()
