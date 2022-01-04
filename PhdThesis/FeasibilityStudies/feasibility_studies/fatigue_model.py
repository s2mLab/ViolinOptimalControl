from bioptim import XiaFatigue, XiaFatigueStabilized, MichaudFatigue, EffortPerception as EP
from bioptim.dynamics.fatigue.muscle_fatigue import MuscleFatigue
import numpy as np

from .enums import Integrator, CustomAnalysis


class FatigueParameters:
    def __init__(
        self,
        LD: float = 100,
        LR: float = 100,
        F: float = 0.005,
        R: float = 0.005,
        scaling: float = 1,
        stabilization_factor: float = 10,
        effort_factor: float = 0.1,
        effort_threshold: float = 0.2
    ):
        self.LD = LD
        self.LR = LR
        self.F = F
        self.R = R
        self.scaling = scaling
        self.effort_factor = effort_factor
        self.stabilization_factor = stabilization_factor
        self.effort_threshold = effort_threshold


class FatigueModel:
    def __init__(
        self,
        model: MuscleFatigue,
        integrator: Integrator,
        x0: tuple[float, ...] = None,
        rms_indices: tuple[int, ...] = None,
        custom_analyses: tuple[CustomAnalysis, ...] = None,
    ):
        self.model = model
        self.integrator = integrator
        self.initial_guess = x0
        self.rms_indices = rms_indices
        self.custom_analyses = custom_analyses

    def apply_dynamics(self, target_load: float, states: np.ndarray) -> np.ndarray:
        return np.array(self.model.apply_dynamics(target_load, *states))[:, 0]

    @property
    def scaling(self) -> float:
        return self.model.scaling


class Xia(FatigueModel):
    def __init__(self, fatigue_params: FatigueParameters, *args, x0: tuple[float, ...] = None, **kwargs):
        model = XiaFatigue(
            LD=fatigue_params.LD,
            LR=fatigue_params.LR,
            F=fatigue_params.F,
            R=fatigue_params.R,
            scaling=fatigue_params.scaling,
        )

        if x0 is None:
            x0 = self.model.default_initial_guess()
        super(Xia, self).__init__(model, *args, x0=x0, **kwargs)


class XiaStabilized(FatigueModel):
    def __init__(self, fatigue_params: FatigueParameters, *args, x0: tuple[float, ...] = None, **kwargs):
        model = XiaFatigueStabilized(
            LD=fatigue_params.LD,
            LR=fatigue_params.LR,
            F=fatigue_params.F,
            R=fatigue_params.R,
            stabilization_factor=fatigue_params.stabilization_factor,
            scaling=fatigue_params.scaling,
        )
        if x0 is None:
            x0 = self.model.default_initial_guess()
        super(XiaStabilized, self).__init__(model, *args, x0=x0, **kwargs)


class Michaud(FatigueModel):
    def __init__(self, fatigue_params: FatigueParameters, *args, x0: tuple[float, ...] = None, **kwargs):
        model = MichaudFatigue(
            LD=fatigue_params.LD,
            LR=fatigue_params.LR,
            F=fatigue_params.F,
            R=fatigue_params.R,
            effort_threshold=fatigue_params.effort_threshold,
            stabilization_factor=fatigue_params.stabilization_factor,
            effort_factor=fatigue_params.effort_factor,
            scaling=fatigue_params.scaling,
        )
        if x0 is None:
            x0 = self.model.default_initial_guess()
        super(Michaud, self).__init__(model, *args, x0=x0, **kwargs)


class EffortPerception(FatigueModel):
    def __init__(self, fatigue_params: FatigueParameters, *args, x0: tuple[float, ...] = None, **kwargs):
        model = EP(
            effort_threshold=fatigue_params.effort_threshold,
            effort_factor=fatigue_params.effort_factor,
            scaling=fatigue_params.scaling,
        )
        if x0 is None:
            x0 = self.model.default_initial_guess()
        super(EffortPerception, self).__init__(model, *args, x0=x0, **kwargs)


class FatigueModels:
    XIA = Xia
    XIA_STABILIZED = XiaStabilized
    MICHAUD = Michaud
    EFFORT_PERCEPTION = EffortPerception
