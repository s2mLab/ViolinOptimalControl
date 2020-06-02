import biorbd
import numpy as np

biorbd_model = biorbd.Model("../../models/BrasViolon.bioMod")
bow_segment_idx = 9
violin_segment_idx = 17

rt_on_string = {"E": 3, "A": 2, "D": 1, "G": 0}

forces_and_moments = []
for string_key in rt_on_string.keys():
    force = biorbd.Vector3d(0, 0, 2)

    b = biorbd.Vector3d(0, 0, 2)
    b.applyRT(biorbd.RotoTrans(biorbd_model.RT(np.zeros(biorbd_model.nbDof()), rt_on_string[string_key]).rot()))
    forces_and_moments.append(np.concatenate((np.array([0, 0, 0]), b.to_array())))
print(forces_and_moments)
