import time
import sys
import pickle
from biorbd_optim import OptimalControlProgram, ShowResult, Data, Simulate, Objective
from up_and_down_bow import xia_model_dynamic, xia_model_configuration, xia_model_fibers, xia_initial_fatigue_at_zero

file_path = "2020_7_25_upDown.bo"

if len(sys.argv) > 1:
    file_path = str(sys.argv[1])

if not isinstance(file_path, str):
    t = time.localtime(time.time())
    file_path = f"results/{t.tm_year}_{t.tm_mon}_{t.tm_mday}_upDown.bo"

ocp, sol = OptimalControlProgram.load(file_path)

sol_obj = Objective.Analyse.get_objective_values(ocp, sol)
analyse = Objective.Analyse(ocp, sol_obj)

print(f"{analyse.mean()}\n{analyse.by_function()}\n{analyse.by_nodes()}")