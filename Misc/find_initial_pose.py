import biorbd
import BiorbdViz
from scipy import optimize
import numpy as np

model_path = "../models/BrasViolon.bioMod"
bow_place = 'frog'  # 'frog', 'tip'
string_to_test = 'E'  # 'G', 'D', 'A', 'E'
show_live_optim = False  # Bool

idxSegmentBow = 8
idxSegmentViolin = 16
tagBowFrog = 16
tagBowTip = 18
tagViolinBString = 38
tagViolinEString = 34
tagViolinAString = 35
tagViolinDString = 36
tagViolinGString = 37
tagViolinCString = 39

if bow_place == 'frog':
    tabBowPosition = tagBowFrog
elif bow_place == 'tip':
    tabBowPosition = tagBowTip
else:
    raise ValueError("bow_place should be: 'frog' or 'tip'")

if string_to_test == 'G':
    tagViolin = tagViolinGString
    tagStringsAround = [tagViolinDString, tagViolinCString]
elif string_to_test == 'D':
    tagViolin = tagViolinDString
    tagStringsAround = [tagViolinAString, tagViolinGString]
elif string_to_test == 'A':
    tagViolin = tagViolinAString
    tagStringsAround = [tagViolinEString, tagViolinDString]
elif string_to_test == 'E':
    tagViolin = tagViolinEString
    tagStringsAround = [tagViolinBString, tagViolinAString]
else:
    raise ValueError("string_to_test should be: 'G', 'D', 'A' or 'E'")

m = biorbd.Model(model_path)
b = BiorbdViz.BiorbdViz(loaded_model=m)
bound_min = []
bound_max = []
for i in range(m.nbSegment()):
    seg = m.segment(i)
    for r in seg.ranges():
        bound_min.append(r.min())
        bound_max.append(r.max())
bounds = (bound_min, bound_max)


def objective_function(x, *args, **kwargs):
    # Update the model
    Q = biorbd.GeneralizedCoordinates(np.array(x))
    m.UpdateKinematicsCustom(Q)
    if show_live_optim:
        b.set_q(Q.to_array())
        b.refresh_window()

    # Get the tag to match
    bow = m.marker(Q, tabBowPosition, True, False).to_array()
    string = m.marker(Q, tagViolin, True, False).to_array()
    bow_position_on_violin = bow - string

    # Get the bow to align with the surrounding strings
    bow_frog = m.marker(Q, tagBowFrog, True, False).to_array()
    bow_tip = m.marker(Q, tagBowTip, True, False).to_array()
    string_behind = m.marker(Q, tagStringsAround[1], True, False).to_array()
    string_front = m.marker(Q, tagStringsAround[0], True, False).to_array()
    bow = bow_tip - bow_frog
    bow_expected = string_behind - string_front
    bow_direction = bow_expected.dot(bow)

    # horsehair on the string
    bow_z_axis = m.globalJCS(idxSegmentBow).to_array()[0:3, 2]
    violin_z_axis = -m.globalJCS(idxSegmentViolin).to_array()[0:3, 2]
    z_axes_alignment = violin_z_axis.dot(bow_z_axis)

    out = np.ndarray((5,))
    out[:3] = bow_position_on_violin
    out[3] = 1 - bow_direction
    out[4] = 1 - z_axes_alignment
    return out


pos = optimize.least_squares(objective_function, x0=np.zeros(m.nbDof(), ), bounds=bounds)
print(f"Optimal Q for the bow at {bow_place} on {string_to_test} string is:\n{pos.x}")
b.set_q(pos.x)
b.exec()
