from enum import Enum

from bioptim import XiaFatigue, XiaTauFatigue, MichaudFatigue, MichaudTauFatigue
import numpy as np

from .bow import BowPosition


class ViolinString(Enum):
    E = "E"
    A = "A"
    D = "D"
    G = "G"


class Violin:
    """
    Contains initial values and references from useful markers and segments.
    """

    # TODO Get these values from a better method
    tau_min, tau_max, tau_init = -30, 30, 0
    segment_idx = 17

    def __init__(self, model: str, string: ViolinString):
        self.model = model
        self.string = string
        if self.model == "BrasViolon":
            self.residual_tau = range(0, 7)  # To be verified
        else:
            self.residual_tau = range(2, 7)

    def q(self, bow_position: BowPosition):
        if self.model == "BrasViolon":
            return {
                "E": {
                    "frog": [
                        -0.08296722,
                        0.09690602,
                        0.79205348,
                        0.6544504,
                        1.48280029,
                        0.08853452,
                        0.53858613,
                        -0.39647921,
                        -0.57508712,
                        -0.0699,
                    ],
                    "tip": [
                        0.05540307,
                        -0.29949352,
                        0.45207956,
                        0.47107735,
                        0.34250652,
                        0.43996516,
                        0.32375985,
                        0.26179933,
                        0.36437326,
                        -0.54957409,
                    ],
                },
                "A": {"frog": [], "tip": []},
                "D": {"frog": [], "tip": []},
                "G": {"frog": [], "tip": []},
            }[self.string.value][bow_position.value]
        elif self.model == "WuViolin":
            return {
                "E": {
                    "tip": [
                        0.03912959,
                        0.17247435,
                        -0.19338927,
                        0.20425233,
                        -0.57008224,
                        0.43337458,
                        0.57221809,
                        1.12542974,
                        -0.07691151,
                        -0.01791363,
                        -0.0434384,
                        0.41112526,
                        -0.54910199,
                    ]
                },
                "A": {"frog": [], "tip": []},
                "D": {"frog": [], "tip": []},
                "G": {"frog": [], "tip": []},
            }[self.string.value][bow_position.value]
        else:
            raise ValueError("Wrong model")

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

    def fatigue_parameters(self, fatigue_type, direction: int = 0):
        if fatigue_type == XiaFatigue:
            return {"LD": 100, "LR": 100, "F": 0.008, "R": 0.002}

        elif fatigue_type == XiaTauFatigue:
            if not (direction < 0 or direction > 0):
                raise ValueError("direction should be < 0 or > 0")
            scale = self.tau_min if direction < 0 else self.tau_max
            out = {"LD": 100, "LR": 100, "F": 0.008, "R": 0.002, "scale": scale}
            return out

        elif fatigue_type == MichaudFatigue:
            return {"LD": 100, "LR": 100, "F": 0.005, "R": 0.005, "L": 0.001, "S": 10, "fatigue_threshold": 0.2}

        elif fatigue_type == MichaudTauFatigue:
            if not (direction < 0 or direction > 0):
                raise ValueError("direction should be < 0 or > 0")
            scale = self.tau_min if direction < 0 else self.tau_max
            out = {"LD": 100, "LR": 100, "F": 0.005, "R": 0.005, "L": 0.001, "S": 10, "fatigue_threshold": 0.2, "scale": scale}
            return out

        elif fatigue_type == MichaudFatigueSimple:
            return {"L": 0.001, "fatigue_threshold": 0.2}

        elif fatigue_type == MichaudTauFatigueSimple:
            if not (direction < 0 or direction > 0):
                raise ValueError("direction should be < 0 or > 0")
            scale = self.tau_min if direction < 0 else self.tau_max
            out = {"L": 0.001, "fatigue_threshold": 0.2, "scale": scale}
            return out

        else:
            raise NotImplementedError(
                "Implemented fatigue_type are XiaFatigue, XiaTauFatigue, " "MichaudFatigue and MichaudTauFatigue"
            )
