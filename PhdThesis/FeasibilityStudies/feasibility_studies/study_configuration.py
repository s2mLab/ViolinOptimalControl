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
        x0: tuple[tuple[float, ...], ...] = None,
        rms_indices: tuple[tuple[int, ...], ...] = None,
        plot_options: tuple[dict, ...] = ({"linestyle": "-"}, {"linestyle": "--"}, {"linestyle": "-."}),
    ):
        self.fatigue_params = fatigue_parameters
        self.t_end = t_end
        self.fixed_target = fixed_target
        self.n_points = n_points

        self.t = np.linspace(0, self.t_end, self.n_points)
        self.target_function = TargetFunctionInternal(self.t_end, self.n_points, self.fixed_target, target_function)
        self.fatigue_models = FatigueModelsInternal(self.fatigue_params, fatigue_models)
        self.x0 = tuple([f.default_initial_guess() for f in self.fatigue_models.models]) if x0 is None else x0
        if len(self.x0) < len(self.fatigue_models.models):
            raise ValueError("len(x0) must be >= len(fatigue_models)")

        self.rms_indices = rms_indices
        if self.rms_indices is not None and len(self.rms_indices) < len(self.fatigue_models.models):
            raise ValueError("len(rms_indices) must be >= len(fatigue_models)")

        self.plot_options = plot_options
        if len(self.plot_options) < len(self.fatigue_models.models):
            raise ValueError("len(plot_options) must be >= len(fatigue_models)")
