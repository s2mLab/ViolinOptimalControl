import numpy as np

from .enums import CustomAnalysis, PlotOptions
from .target_function import TargetFunctions, TargetFunctionInternal
from .fatigue_model import FatigueModel


class StudyConfiguration:
    def __init__(
        self,
        fatigue_models: tuple[FatigueModel, ...],
        t_end: float,
        fixed_target: float,
        target_function: TargetFunctions,
        n_points: int,
        repeat: int = 1,
        plot_options: PlotOptions = PlotOptions(),
    ):
        self.t_end = t_end
        self.fixed_target = fixed_target
        self.n_points = n_points
        self.repeat = repeat

        self.t = np.linspace(0, self.t_end, self.n_points)
        self.target_function = TargetFunctionInternal(self.t_end, self.n_points, self.fixed_target, target_function)
        self.fatigue_models = fatigue_models

        self.plot_options = plot_options
        if len(self.plot_options.options) < len(self.fatigue_models):
            raise ValueError("len(plot_options) must be >= len(fatigue_models)")
