import time
import sys
import pickle

from bioptim import OptimalControlProgram, ShowResult, Data, Simulate
from up_and_down_bow import xia_model_dynamic, xia_model_configuration, xia_model_fibers, xia_initial_fatigue_at_zero

file_path = "results/xia 5 phases/2020_7_25_upDown.bo"


if len(sys.argv) > 1:
    file_path = str(sys.argv[1])

if not isinstance(file_path, str):
    t = time.localtime(time.time())
    file_path = f"results/{t.tm_year}_{t.tm_mon}_{t.tm_mday}_upDown.bo"

ocp, sol = OptimalControlProgram.load(file_path)

d = Data.get_data(ocp, Simulate.from_solve(ocp, sol, single_shoot=True))
dict = {"data": d}
with open(file_path[:-3] + "_single.bob", "wb") as file:
    pickle.dump(dict, file)
