from enum import Enum
from typing import Union


class TargetFunctions(Enum):
    TARGET_UP_TO_END = "target_up_to_end"
    TARGET_UP_TO_MID_THEN_ZERO = "target_up_to_mid_then_zero"


class TargetFunctionInternal:
    def __init__(self, t_end: float, fixed_target: Union[int, float, tuple, list], chosen_function: TargetFunctions):
        self.t_end = t_end
        self.fixed_target = fixed_target
        self.function = getattr(self, chosen_function.value)

    def target_up_to_end(self, t):
        return self.fixed_target

    def target_up_to_mid_then_zero(self, t):
        return self.fixed_target if t < self.t_end / 2 else 0