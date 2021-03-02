import biorbd
import bioviz
from bioptim import ObjectiveFcn, BiMapping
from scipy import optimize
import numpy as np
from casadi import MX, Function


class DummyPenalty:
    class DummyNlp:
        def __init__(self, m):
            self.model = m
            self.q = MX.sym("q", m.nbQ(), 1)
            self.casadi_func = {}
            self.mapping = {"q": BiMapping(range(self.model.nbQ()), range(self.model.nbQ()))}

    class DummyPen:
        @staticmethod
        def get_type():
            return DummyPenalty

    def __init__(self, m):
        self.ocp = []
        self.nlp = DummyPenalty.DummyNlp(m)
        self.x = [self.nlp.q]
        self.type = DummyPenalty.DummyPen()
        self.val = None

    @staticmethod
    def add_to_penalty(ocp, nlp, val, penalty):
        penalty.val = val


model_path = "../models/BrasViolon.bioMod"
bow_place = 'frog'  # 'frog', 'tip'
string_to_test = 'E'  # 'G', 'D', 'A', 'E'
show_live_optim = False  # Bool

idx_segment_bow_hair = 9
tag_bow_contact = 19
tag_violin_e_string = 35
tag_violin_a_string = 37
tag_violin_d_string = 39
tag_violin_g_string = 41

if string_to_test == 'G':
    tag_violin = tag_violin_g_string
    rt_on_string = 0
elif string_to_test == 'D':
    tag_violin = tag_violin_d_string
    rt_on_string = 1
elif string_to_test == 'A':
    tag_violin = tag_violin_a_string
    rt_on_string = 2
elif string_to_test == 'E':
    tag_violin = tag_violin_e_string
    rt_on_string = 3
else:
    raise ValueError("string_to_test should be: 'G', 'D', 'A' or 'E'")

m = biorbd.Model(model_path)
bound_min = []
bound_max = []
for i in range(m.nbSegment()):
    seg = m.segment(i)
    for r in seg.QRanges():
        bound_min.append(r.min())
        bound_max.append(r.max())
bounds = (bound_min, bound_max)


pn = DummyPenalty(m)
ObjectiveFcn.Lagrange.TRACK_SEGMENT_WITH_CUSTOM_RT.value[0](pn, pn, idx_segment_bow_hair, rt_on_string)
custom_rt = Function("custom_rt", [pn.nlp.q], [pn.val]).expand()
ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS.value[0](pn, pn, tag_bow_contact, tag_violin)
superimpose = Function("superimpose", [pn.nlp.q], [pn.val]).expand()


def objective_function(x, *args, **kwargs):
    out = np.ndarray((6, ))
    out[:3] = np.array(custom_rt(x))[:, 0]
    out[3:] = np.array(superimpose(x))[:, 0]
    return out


b = bioviz.Viz(loaded_model=m, markers_size=0.003, show_markers=True, show_meshes=False)
x0 = np.zeros(m.nbDof(), )
if bow_place == "frog":
    x0[-1] = -0.07
    bounds[0][-1] = -0.0701
    bounds[1][-1] = -0.0699
else:
    x0[-1] = -0.55
    bounds[0][-1] = -0.551
    bounds[1][-1] = -0.549
pos = optimize.least_squares(objective_function, x0=x0, bounds=bounds)
print(f"Optimal Q for the bow at {bow_place} on {string_to_test} string is:\n{pos.x}")
b.set_q(pos.x)
b.exec()
