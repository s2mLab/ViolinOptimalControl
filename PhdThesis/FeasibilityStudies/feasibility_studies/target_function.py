from enum import Enum
from typing import Union

import numpy as np


class TargetFunctions(Enum):
    TARGET_UP_TO_END = "target_up_to_end"
    TARGET_UP_TO_MID_THEN_ZERO = "target_up_to_mid_then_zero"
    TARGET_RANDOM_PER_FRAME = "target_random_per_frame"
    TARGET_RANDOM_PER_SECOND = "target_random_per_second"
    TARGET_RANDOM_PER_10SECONDS = "target_random_per_10seconds"
    TARGET_ON_AND_OFF = "target_on_and_off"


class TargetFunctionInternal:
    def __init__(
        self,
        t_end: float,
        n_points: int,
        fixed_target: Union[int, float, tuple, list],
        chosen_function: TargetFunctions
    ):
        self.t_end = t_end
        self.fixed_target = fixed_target
        self.n_points = n_points
        self.function = getattr(self, chosen_function.value)

        np.random.seed(42)
        self.random_target = np.random.rand((int(self.t_end) + 1) * self.n_points + 1)

    def target_up_to_end(self, t):
        return self.fixed_target

    def target_up_to_mid_then_zero(self, t):
        return self.fixed_target if t < self.t_end / 2 else 0

    def target_random_per_frame(self, t):
        return self.random_target[int(t / self.t_end * self.n_points)]

    def target_random_per_second(self, t):
        return self.random_target[int(t)]

    def target_random_per_10seconds(self, t):
        return self.random_target[int(t / 10)]

    def target_on_and_off(self, t):
        return int(t / self.t_end * self.n_points) % 2
