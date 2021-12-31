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
