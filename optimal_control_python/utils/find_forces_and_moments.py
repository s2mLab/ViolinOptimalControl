import biorbd
import numpy as np


if biorbd.currentLinearAlgebraBackend() != biorbd.EIGEN3:
    raise RuntimeError("Biorbd backend should be biorbd.EIGEN3")

force_target = biorbd.Vector3d(0, 0, 2)

biorbd_model = biorbd.Model("../../models/BrasViolon.bioMod")
bow_segment_idx = 9
violin_segment_idx = 17
rt_on_string = {"E": 3, "A": 2, "D": 1, "G": 0}

print(f"Converting F={force_target.to_array()} N for each strings")
for string_key in rt_on_string.keys():
    b = biorbd.Vector3d(force_target.to_array()[0], force_target.to_array()[1], force_target.to_array()[2])
    b.applyRT(biorbd.RotoTrans(biorbd_model.RT(np.zeros(biorbd_model.nbDof()), rt_on_string[string_key]).rot()))
    print(f"Moment/Force for {string_key} string = {np.concatenate((np.array([0, 0, 0]), b.to_array()))}")
