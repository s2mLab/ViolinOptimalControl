import numpy as np

from .target_function import TargetFunctions, TargetFunctionInternal
from .fatigue_models import FatigueModels, FatigueModelsInternal
from .fatigue_parameters import FatigueParameters


class StudyConfiguration:
    def __init__(
        self,
        fatigue_parameters: FatigueParameters,
        t_end: float,
        fixed_target: float,
        target_function: TargetFunctions,
        n_points: int,
        fatigue_models: tuple[FatigueModels, ...],
        linestyles: tuple[str] = ("-", "--", "-.", ".-"),
    ):
        self.fatigue_params = fatigue_parameters
        self.t_end = t_end
        self.fixed_target = fixed_target
        self.n_points = n_points

        self.t = np.linspace(0, self.t_end, self.n_points)
        self.target_function = TargetFunctionInternal(self.t_end, self.fixed_target, target_function)
        self.fatigue_models = FatigueModelsInternal(self.fatigue_params, fatigue_models)
        self.linestyles = linestyles
        if len(self.linestyles) < len(self.fatigue_models.models):
            raise ValueError("len(linestyles) must be >= len(fatigue_models)")