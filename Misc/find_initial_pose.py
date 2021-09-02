import biorbd_casadi as biorbd
import bioviz
from bioptim import ObjectiveFcn, BiMapping
from bioptim.optimization.optimization_variable import OptimizationVariableList
from scipy import optimize
import numpy as np
from casadi import MX, Function


class DummyPenalty:
    class DummyState:
        def __init__(self, mx):
            self.mx = mx
            self.cx = mx

    class DummyNlp:
        def __init__(self, m):
            self.model = m
            self.states = OptimizationVariableList()
            self.states.append(
                "q",
                [MX.sym("q", m.nbQ(), 1)],
                MX.sym("q", m.nbQ(), 1),
                BiMapping(range(self.model.nbQ()), range(self.model.nbQ())),
            )
            self.casadi_func = dict()

    class DummyPen:
        @staticmethod
        def get_type():
            return DummyPenalty

    def __init__(self, m):
        self.ocp = []
        self.nlp = DummyPenalty.DummyNlp(m)
        self.type = DummyPenalty.DummyPen()
        self.quadratic = True
        self.rows = None

    @staticmethod
    def add_to_penalty(ocp, nlp, val, penalty):
        penalty.val = val


bow_place = "tip"  # 'frog', 'tip'
string_to_test = "E"  # 'G', 'D', 'A', 'E'
show_live_optim = False  # Bool
model = "WuViolin"

if model == "BrasViolon":
    model_path = "../models/BrasViolon.bioMod"
    idx_segment_bow_hair = 9
    tag_bow_contact = 19
    tag_violin_e_string = 35
    tag_violin_a_string = 37
    tag_violin_d_string = 39
    tag_violin_g_string = 41
elif model == "WuViolin":
    model_path = "../models/WuViolin.bioMod"
    idx_segment_bow_hair = 14
    tag_bow_contact = 2
    tag_violin_e_string = 3
    tag_violin_a_string = 5
    tag_violin_d_string = 7
    tag_violin_g_string = 9
else:
    raise ValueError("Wrong model")

if string_to_test == "G":
    tag_violin = tag_violin_g_string
    rt_on_string = 0
elif string_to_test == "D":
    tag_violin = tag_violin_d_string
    rt_on_string = 1
elif string_to_test == "A":
    tag_violin = tag_violin_a_string
    rt_on_string = 2
elif string_to_test == "E":
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
val = ObjectiveFcn.Lagrange.TRACK_SEGMENT_WITH_CUSTOM_RT.value[0](pn, pn, idx_segment_bow_hair, rt_on_string)
custom_rt = Function("custom_rt", [pn.nlp.states["q"].cx], [val]).expand()
val = ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS.value[0](pn, pn, first_marker=tag_bow_contact, second_marker=tag_violin)
superimpose = Function("superimpose", [pn.nlp.states["q"].cx], [val]).expand()


def objective_function(x, *args, **kwargs):
    out = np.ndarray((6,))
    out[:3] = np.array(custom_rt(x))[:, 0]
    out[3:] = np.array(superimpose(x))[:, 0]
    return out


b = bioviz.Viz(m.path().absolutePath().to_string(), markers_size=0.003, show_markers=True)
x0 = np.zeros(
    m.nbDof(),
)
if bow_place == "frog":
    bounds[0][-1] = -0.0701
    bounds[1][-1] = -0.0699
else:
    bounds[0][-1] = -0.551
    bounds[1][-1] = -0.549
x0 = np.mean(bounds, axis=0)
pos = optimize.least_squares(objective_function, x0=x0, bounds=bounds)
print(
    f"Optimal Q for the bow at {bow_place} on {string_to_test} string is:\n{pos.x}\n"
    f"with cost function = {objective_function(pos.x)}"
)
b.set_q(pos.x)
b.exec()
