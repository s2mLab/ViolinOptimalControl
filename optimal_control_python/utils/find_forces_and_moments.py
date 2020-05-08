import biorbd
import numpy as np

biorbd_model = biorbd.Model("../../models/BrasViolon.bioMod")
bow_segment_idx = 8
violin_segment_idx = 16

forces_and_moments = []
for segment_idx in (bow_segment_idx, violin_segment_idx):
    force = biorbd.Vector3d(0,2,0)
    jcs = biorbd_model.globalJCS(np.zeros(biorbd_model.nbDof()), segment_idx)
    force.applyRT(jcs)
    force = force.to_array()
    jcs = jcs.trans()
    jcs = jcs.to_array()
    moment = np.cross(force, jcs)
    forces_and_moments.append(np.concatenate((moment, force)))
print(forces_and_moments)