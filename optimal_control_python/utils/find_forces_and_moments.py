import biorbd
import numpy as np

biorbd_model = biorbd.Model("../../models/BrasViolon.bioMod")
bow_segment_idx = 9
violin_segment_idx = 17

bridge_marker = {"E": 35, "A": 37, "D": 39, "G": 41}
rt_on_string = {"E": 3, "A": 2, "D": 1, "G": 0}

forces_and_moments = []
for string_key in bridge_marker.keys():
    force = biorbd.Vector3d(0, 0, 2)

    jcs = biorbd_model.globalJCS(np.zeros(biorbd_model.nbDof()), bow_segment_idx)

    r_seg = biorbd_model.globalJCS(np.zeros(biorbd_model.nbDof()), bow_segment_idx).rot()

    seg_violin = biorbd_model.segment(violin_segment_idx)
    r_rt = biorbd_model.RT(np.zeros(biorbd_model.nbDof()), rt_on_string[string_key]).rot()

    force.applyRT(jcs)
    force = force.to_array()
    jcs = jcs.trans()
    jcs = jcs.to_array()
    moment = np.cross(force, jcs)
    forces_and_moments.append(np.concatenate((moment, force)))
print(forces_and_moments)