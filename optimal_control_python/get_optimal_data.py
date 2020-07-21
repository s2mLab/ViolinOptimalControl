import time
import pickle
import numpy as np
import sys

file_path = "results/xia2/2020_7_21_upDown.bob"

if len(sys.argv) > 1:
    file_path = str(sys.argv[1])

if not isinstance(file_path, str):
    t = time.localtime(time.time())
    file_path = f"results/{t.tm_year}_{t.tm_mon}_{t.tm_mday}_upDown.bob"

with open(file_path, "rb") as file:
    data = pickle.load(file)


states, controls = data["data"]

optimal_states = np.concatenate((states["q"], states["q_dot"],  states["muscles_active"],  states["muscles_fatigue"],  states["muscles_resting"]), 0)
optimal_controls = np.concatenate((controls["tau"], controls["muscles"]), 0)[:, :-1]
dict = {"states": optimal_states, "controls": optimal_controls}

with open("utils/optimal_init_15_nodes.bio", "wb") as file:
    pickle.dump(dict, file)

