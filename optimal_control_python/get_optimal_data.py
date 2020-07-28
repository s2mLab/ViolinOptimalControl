import time
import pickle
import numpy as np
import sys

from biorbd_optim import OptimalControlProgram, ShowResult, Data, Simulate
from up_and_down_bow import xia_model_dynamic, xia_model_configuration, xia_model_fibers, xia_initial_fatigue_at_zero

file_path = "results/xia 5 phases/2020_7_25_upDown.bo"

if len(sys.argv) > 1:
    file_path = str(sys.argv[1])

if not isinstance(file_path, str):
    t = time.localtime(time.time())
    file_path = f"results/{t.tm_year}_{t.tm_mon}_{t.tm_mday}_upDown.bob"

ocp, sol = OptimalControlProgram.load(file_path)

new_ns = 10

if new_ns == ocp.nlp[0]['ns']:
    states, controls = Data.get_data(ocp, sol, concatenate=False)
else:
    states, controls = Data.get_data(ocp, sol, interpolate_nb_frames=new_ns + 1, concatenate=False)



optimal_states = np.concatenate(
    (states["q"][0], states["q_dot"][0], states["muscles_active"][0], states["muscles_fatigue"][0], states["muscles_resting"][0]), 0
)
optimal_controls = np.concatenate((controls["tau"][0], controls["muscles"][0]), 0)[:, :-1]

dict = {"states": optimal_states, "controls": optimal_controls}

with open(f"utils/optimal_init_{new_ns}_nodes_first.bio", "wb") as file:
    pickle.dump(dict, file)

if ocp.nb_phases > 1:
    optimal_states = np.concatenate(
        (states["q"][1], states["q_dot"][1], states["muscles_active"][1], states["muscles_fatigue"][1],
         states["muscles_resting"][0]), 0
    )
    optimal_controls = np.concatenate((controls["tau"][1], controls["muscles"][1]), 0)[:, :-1]

    dict = {"states": optimal_states, "controls": optimal_controls}

    with open(f"utils/optimal_init_{new_ns}_nodes_others.bio", "wb") as file:
        pickle.dump(dict, file)