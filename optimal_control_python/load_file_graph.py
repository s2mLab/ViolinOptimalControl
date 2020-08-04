import time
import sys

from biorbd_optim import OptimalControlProgram, ShowResult, Data
from up_and_down_bow import xia_model_dynamic, xia_model_configuration, xia_model_fibers, xia_initial_fatigue_at_zero

file_path = "/home/theophile/Documents/programmation/ViolinOptimalControl/optimal_control_python/results/2020_7_29_upDown.bo"


if len(sys.argv) > 1:
    file_path = str(sys.argv[1])

if not isinstance(file_path, str):
    t = time.localtime(time.time())
    file_path = f"results/{t.tm_year}_{t.tm_mon}_{t.tm_mday}_upDown.bo"

ocp, sol = OptimalControlProgram.load(file_path)

d = Data.get_data(ocp, sol)

result = ShowResult(ocp, sol)
result.graphs()
