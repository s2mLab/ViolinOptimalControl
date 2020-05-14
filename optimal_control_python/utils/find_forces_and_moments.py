import biorbd
import numpy as np

biorbd_model = biorbd.Model("../../models/BrasViolon.bioMod")
bow_segment_idx = 9
violin_segment_idx = 17

forces_and_moments = []
for segment_idx in (bow_segment_idx):
    force = biorbd.Vector3d(0,0,2)
    jcs = biorbd_model.globalJCS(np.zeros(biorbd_model.nbDof()), segment_idx)

    seg_violin = biorbd_model.segment(violin_segment_idx)


    force.applyRT(jcs)
    force = force.to_array()
    jcs = jcs.trans()
    jcs = jcs.to_array()
    moment = np.cross(force, jcs)
    forces_and_moments.append(np.concatenate((moment, force)))
print(forces_and_moments)