import time
import pickle
import numpy as np

file_path = 0

if not isinstance(file_path, str):
    t = time.localtime(time.time())
    file_path = f"results/{t.tm_year}_{t.tm_mon}_{t.tm_mday}_upDown.bob"

with open(file_path, "rb") as file:
    data = pickle.load(file)

data_interpolate, _ = data["data"]

init = np.concatenate((data_interpolate["q"], data_interpolate["q_dot"]), 0)

# step = 80/49
# init = np.delete(init, [int(step * k) for k in range(80 - 31)], 1)


_, data_interpolate = data["data"]

init = np.concatenate((data_interpolate["tau"], data_interpolate["muscles"]), 0)

print(init[:, :-1])
