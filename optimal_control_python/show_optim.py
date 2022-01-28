import pickle
import bioviz
import biorbd_casadi as biorbd

model_path = "../models/WuViolin.bioMod"
m = biorbd.Model(model_path)
nq = m.nbQ()

solution_path = "results/2021_10_7_1_cycles_with_fatigue_out.bo"
with open(solution_path, "rb") as file:
    states, controls, parameters = pickle.load(file)

b = bioviz.Viz(
    model_path,
    markers_size=0.003,
    show_local_ref_frame=False,
    show_global_ref_frame=False,
    show_segments_center_of_mass=False,
)
b.load_movement(states["q"])
b.exec()
