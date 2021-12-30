from typing import Union


class TargetFunctions:
    def __init__(self, t_end: float, target: Union[int, float, tuple, list]):
        self.TARGET_UP_TO_END = lambda t: target
        self.TARGET_UP_TO_MID_THEN_ZERO = lambda t: target if t < t_end / 2 else 0
