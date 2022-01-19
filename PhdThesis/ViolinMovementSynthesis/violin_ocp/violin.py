from enum import Enum

from scipy import optimize
from bioptim import MichaudFatigue, MichaudTauFatigue, EffortPerception, TauEffortPerception, ObjectiveFcn, BiMapping
from bioptim.optimization.optimization_variable import OptimizationVariableList
import numpy as np
from casadi import MX, Function
import biorbd_casadi as biorbd

from .enums import FatigueType, StructureType
from .bow import Bow, BowPosition


class ViolinString(Enum):
    E = "E"
    A = "A"
    D = "D"
    G = "G"


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


class Violin:
    """
    Contains initial values and references from useful markers and segments.
    """

    def __init__(self, model: str, string: ViolinString):
        self.model = model
        self.string = string
        if self.model == "BrasViolon":
            self.residual_tau = range(0, 7)  # To be verified
            self.segment_idx = 17
        else:
            self.residual_tau = range(2, 7)
            self.segment_idx = 17
            # Arbitrarily set all dof from tau in [-30; 30] and then adjust some dof
            self.tau_min = [-30] * 13
            self.tau_max = [30] * 13
            self.tau_init = [0] * 13
            self.tau_min[1], self.tau_max[1] = -20, 20  # Clavicle elevation
            self.tau_min[2], self.tau_max[2] = -15, 15  # Scapula elevation

    def q(self, biorbd_model: biorbd.Model, bow: Bow, bow_position: BowPosition):
        # Get some values
        idx_segment_bow_hair = bow.hair_idx
        tag_bow_contact = bow.contact_marker
        tag_violin = self.bridge_marker
        rt_on_string = self.rt_on_string

        bound_min = []
        bound_max = []
        for i in range(biorbd_model.nbSegment()):
            seg = biorbd_model.segment(i)
            for r in seg.QRanges():
                bound_min.append(r.min())
                bound_max.append(r.max())
        bounds = (bound_min, bound_max)

        pn = DummyPenalty(biorbd_model)
        val = ObjectiveFcn.Lagrange.TRACK_SEGMENT_WITH_CUSTOM_RT.value[0](pn, pn, idx_segment_bow_hair, rt_on_string)
        custom_rt = Function("custom_rt", [pn.nlp.states["q"].cx], [val]).expand()
        val = ObjectiveFcn.Mayer.SUPERIMPOSE_MARKERS.value[0](
            pn, pn, first_marker=tag_bow_contact, second_marker=tag_violin
        )
        superimpose = Function("superimpose", [pn.nlp.states["q"].cx], [val]).expand()

        def objective_function(x, *args, **kwargs):
            out = np.ndarray((6,))
            out[:3] = np.array(custom_rt(x))[:, 0]
            out[3:] = np.array(superimpose(x))[:, 0]
            return out

        if bow_position == BowPosition.FROG:
            bounds[0][-1] = -0.0701
            bounds[1][-1] = -0.0699
        else:
            bounds[0][-1] = -0.551
            bounds[1][-1] = -0.549
        x0 = np.mean(bounds, axis=0)
        return optimize.least_squares(objective_function, x0=x0, bounds=bounds).x

    @property
    def bridge_marker(self):
        """
        :return: Marker number on the bridge, associate to the string.
        """
        if self.model == "BrasViolon":
            return {
                "E": 35,
                "A": 37,
                "D": 39,
                "G": 41,
            }[self.string.value]
        elif self.model == "WuViolin":
            return {
                "E": 3,
                "A": 5,
                "D": 7,
                "G": 9,
            }[self.string.value]
        else:
            raise ValueError("Wrong model")

    @property
    def neck_marker(self):
        """
        :return: Marker number on the neck of the violin, associate to the string.
        """
        if self.model == "BrasViolon":
            return {
                "E": 36,
                "A": 38,
                "D": 40,
                "G": 42,
            }[self.string.value]
        elif self.model == "WuViolin":
            return {
                "E": 4,
                "A": 6,
                "D": 8,
                "G": 10,
            }[self.string.value]
        else:
            raise ValueError("Wrong model")

    @property
    def rt_on_string(self):
        """
        :return: RT number according to the string.
        """
        return {
            "E": 3,
            "A": 2,
            "D": 1,
            "G": 0,
        }[self.string.value]

    @property
    def external_force(self):
        # This was obtained from "violin_ocp.find_forces_and_moments"
        return {
            "E": np.array([0.0, 0.0, 0.0, 0.40989355, 1.84413989, 0.65660896]),
            "A": np.array([0.0, 0.0, 0.0, 0.30881124, 1.65124622, 1.08536701]),
            "D": np.array([0.0, 0.0, 0.0, 0.16081784, 1.30189937, 1.50970052]),
            "G": np.array([0.0, 0.0, 0.0, 0.05865013, 1.05013794, 1.7011086]),
        }[self.string.value]

    def fatigue_model(
        self,
        fatigue_type: FatigueType,
        structure_type: StructureType,
        state_only: bool = True,
        index: int = None,
        split_tau: bool = False,
    ):
        if fatigue_type == FatigueType.QCC:
            if structure_type == StructureType.MUSCLE:
                return MichaudFatigue(**self.fatigue_parameters(fatigue_type, structure_type))

            elif structure_type == StructureType.TAU:
                return MichaudTauFatigue(
                    MichaudFatigue(**self.fatigue_parameters(fatigue_type, structure_type, -1, index)),
                    MichaudFatigue(**self.fatigue_parameters(fatigue_type, structure_type, 1, index)),
                    state_only=state_only,
                    split_controls=split_tau,
                )
            else:
                raise NotImplementedError("Structure type not implemented yet")

        elif fatigue_type == FatigueType.EFFORT_PERCEPTION:
            if structure_type == StructureType.MUSCLE:
                return EffortPerception(**self.fatigue_parameters(fatigue_type, structure_type))

            elif structure_type == StructureType.TAU:
                return TauEffortPerception(
                    EffortPerception(**self.fatigue_parameters(fatigue_type, structure_type, -1, index)),
                    EffortPerception(**self.fatigue_parameters(fatigue_type, structure_type, 1, index)),
                    split_controls=split_tau,
                )
            else:
                raise NotImplementedError("Structure type not implemented yet")

        else:
            raise NotImplementedError("Fatigue type not implemented yet")

    def fatigue_parameters(
        self,
        fatigue_type: FatigueType,
        structure_type: StructureType,
        direction: int = 0,
        index: int = None,
        LD: float = 100,
        LR: float = 100,
        F: float = 0.005,
        R: float = 0.005,
        scaling: float = None,  # Default 1
        stabilization_factor: float = 10,
        effort_factor: float = 0.1,
        effort_threshold: float = None,  # Default 0.2 * scaling
    ):
        if scaling is None:
            scaling = 1
            if structure_type == StructureType.TAU:
                if not (direction < 0 or direction > 0):
                    raise ValueError("direction should be < 0 or > 0")
                scaling = self.tau_min[index] if direction < 0 else self.tau_max[index]

        if effort_threshold is None:
            effort_threshold = scaling / 5

        if fatigue_type == FatigueType.QCC:
            if structure_type == StructureType.MUSCLE:
                return {
                    "LD": LD,
                    "LR": LR,
                    "F": F,
                    "R": R,
                    "effort_factor": effort_factor,
                    "stabilization_factor": stabilization_factor,
                    "effort_threshold": effort_threshold,
                }

            elif structure_type == StructureType.TAU:
                return {
                    "LD": LD,
                    "LR": LR,
                    "F": F,
                    "R": R,
                    "effort_factor": effort_factor,
                    "stabilization_factor": stabilization_factor,
                    "effort_threshold": effort_threshold,
                    "scaling": scaling,
                }
            else:
                raise NotImplementedError("Structure type not implemented yet")

        elif fatigue_type == FatigueType.EFFORT_PERCEPTION:
            if structure_type == StructureType.MUSCLE:
                return {"effort_factor": effort_factor, "effort_threshold": effort_threshold}

            elif structure_type == StructureType.TAU:
                return {"effort_factor": effort_factor, "effort_threshold": effort_threshold, "scaling": scaling}
            else:
                raise NotImplementedError("Structure type not implemented yet")

        else:
            raise NotImplementedError("Fatigue type not implemented yet")
