from bioptim import MichaudTauFatigue, MichaudFatigue, EffortPerception as EP, TauEffortPerception
from bioptim.dynamics.fatigue.muscle_fatigue import MuscleFatigue

from .enums import FatigableStructure


class FatigueParameters:
    def __init__(
        self,
        LD: float = 100,
        LR: float = 100,
        F: float = 0.005,
        R: float = 0.005,
        scaling: float = 1,
        stabilization_factor: float = 10,
        effort_factor: float = 0.0075,
        effort_threshold: float = 0.2,
        split_controls: bool = True,
    ):
        self.LD = LD
        self.LR = LR
        self.F = F
        self.R = R
        self.scaling = scaling
        self.effort_factor = effort_factor
        self.stabilization_factor = stabilization_factor
        self.effort_threshold = effort_threshold
        self.split_controls = split_controls


class FatigueModel:
    def __init__(
        self,
        model: MuscleFatigue,
    ):
        self.model = model


class Michaud(FatigueModel):
    def __init__(self, fatigable_structure: FatigableStructure, fatigue_params: FatigueParameters):
        if fatigable_structure == FatigableStructure.JOINTS:
            model = MichaudTauFatigue(
                MichaudFatigue(
                    LD=fatigue_params.LD,
                    LR=fatigue_params.LR,
                    F=fatigue_params.F,
                    R=fatigue_params.R,
                    effort_threshold=fatigue_params.effort_threshold,
                    stabilization_factor=fatigue_params.stabilization_factor,
                    effort_factor=fatigue_params.effort_factor,
                    scaling=-fatigue_params.scaling,
                ),
                MichaudFatigue(
                    LD=fatigue_params.LD,
                    LR=fatigue_params.LR,
                    F=fatigue_params.F,
                    R=fatigue_params.R,
                    effort_threshold=fatigue_params.effort_threshold,
                    stabilization_factor=fatigue_params.stabilization_factor,
                    effort_factor=fatigue_params.effort_factor,
                    scaling=fatigue_params.scaling,
                ),
                split_controls=fatigue_params.split_controls,
            )
        elif fatigable_structure == FatigableStructure.MUSCLES:
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
        else:
            raise NotImplementedError("Fatigue structure model not implemented")

        super(Michaud, self).__init__(model)


class EffortPerception(FatigueModel):
    def __init__(self, fatigable_structure: FatigableStructure, fatigue_params: FatigueParameters):
        if fatigable_structure == FatigableStructure.JOINTS:
            model = TauEffortPerception(
                EP(
                    effort_threshold=fatigue_params.effort_threshold,
                    effort_factor=fatigue_params.effort_factor,
                    scaling=-fatigue_params.scaling,
                ),
                EP(
                    effort_threshold=fatigue_params.effort_threshold,
                    effort_factor=fatigue_params.effort_factor,
                    scaling=fatigue_params.scaling,
                ),
                split_controls=fatigue_params.split_controls,
            )
        elif fatigable_structure == FatigableStructure.MUSCLES:
            model = EP(
                effort_threshold=fatigue_params.effort_threshold,
                effort_factor=fatigue_params.effort_factor,
                scaling=fatigue_params.scaling,
            )
        else:
            raise NotImplementedError("Fatigue structure model not implemented")

        super(EffortPerception, self).__init__(model)


class FatigueModels:
    NONE = None
    MICHAUD = Michaud
    EFFORT_PERCEPTION = EffortPerception
