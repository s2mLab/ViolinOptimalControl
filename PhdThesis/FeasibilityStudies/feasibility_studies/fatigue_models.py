from enum import Enum

from bioptim import XiaFatigue, XiaFatigueStabilized, MichaudFatigue, EffortPerception

from .fatigue_parameters import FatigueParameters


class FatigueModels(Enum):
    XIA = "xia"
    XIA_STABILIZED = "xia_stabilized"
    MICHAUD = "michaud"
    EFFORT_PERCEPTION = "effort_perception"


class FatigueModelsInternal:
    def __init__(self, fatigue_params: FatigueParameters, chosen_models: tuple[FatigueModels]):
        self.fatigue_params = fatigue_params
        self.models = tuple([getattr(self, chosen_model.value)() for chosen_model in chosen_models])

    def xia(self):
        return XiaFatigue(
            LD=self.fatigue_params.LD,
            LR=self.fatigue_params.LR,
            F=self.fatigue_params.F,
            R=self.fatigue_params.R,
            scaling=self.fatigue_params.scaling,
        )

    def xia_stabilized(self):
        return XiaFatigueStabilized(
            LD=self.fatigue_params.LD,
            LR=self.fatigue_params.LR,
            F=self.fatigue_params.F,
            R=self.fatigue_params.R,
            stabilization_factor=self.fatigue_params.stabilization_factor,
            scaling=self.fatigue_params.scaling,
        )

    def michaud(self):
        return MichaudFatigue(
            LD=self.fatigue_params.LD,
            LR=self.fatigue_params.LR,
            F=self.fatigue_params.F,
            R=self.fatigue_params.R,
            effort_threshold=self.fatigue_params.effort_threshold,
            stabilization_factor=self.fatigue_params.stabilization_factor,
            effort_factor=self.fatigue_params.effort_factor,
            scaling=self.fatigue_params.scaling,
        )

    def effort_perception(self):
        return EffortPerception(
            effort_threshold=self.fatigue_params.effort_threshold,
            effort_factor=self.fatigue_params.effort_factor,
            scaling=self.fatigue_params.scaling,
        )
