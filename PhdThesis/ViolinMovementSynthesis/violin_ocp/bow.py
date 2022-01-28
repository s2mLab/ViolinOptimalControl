from enum import Enum

from scipy import optimize
import numpy as np
import scipy.integrate as integrate


class BowPosition(Enum):
    FROG = "frog"
    MID = "mid"
    TIP = "tip"


class Bow:
    """
    Contains references from useful markers.
    """

    def __init__(self, model):
        """
        Contains the side of the bow.
        """

        if model == "BrasViolon":
            self.segment_idx: int = 8
            self.hair_idx: int = 9
            self.hair_limits: list = [-0.07, -0.55]
            self.contact_marker: int = 19
            self.frog_marker: int = 16
            self.tip_marker: int = 18
        elif model == "WuViolin":
            self.segment_idx: int = 12
            self.hair_idx: int = 12
            self.hair_limits: list = [-0.07, -0.55]
            self.contact_marker: str = "bow_contact"
            self.frog_marker: int = 0
            self.tip_marker: int = 1
        else:
            raise ValueError("Wrong model")


class BowTrajectory:
    # TODO Allow to start from FROG

    def __init__(self, bow_range: list, n_points: int, bow_speed: float = 10, bow_acceleration: float = 0.5):
        # x[0] = amplitude
        # x[1]= 2 * pi / period
        def objective_function(x, t, final_t):
            integ = self.curve_integral(x, t)
            moy = (integ[int(n_points / 2) - 2] + integ[int(n_points / 2) + 2]) / 2
            return np.array((bow_acceleration - x[0], bow_speed - x[1], (moy - self.bow_limits[1]) * 1000))

        self.final_t = 2
        self.bow_limits = bow_range
        self.t = np.linspace(0, self.final_t, n_points)
        self.optimized_values = optimize.least_squares(lambda x: objective_function(x, self.t, self.final_t), x0=np.array((1, 8))).x
        self.target = self.curve_integral(self.optimized_values, self.t)[np.newaxis, :]

    @staticmethod
    def curve(x, t, final_t):
        amplitude = x[0]
        period = final_t * np.pi / x[1]
        mid_time = final_t / 2
        if isinstance(t, float):
            t = np.array((t,))

        y = np.ndarray((t.shape[0],))
        for i in range(t.shape[0]):
            if t[i] < (period / 4):
                y[i] = amplitude * np.sin(t[i] * x[1])
            elif (period / 4) < t[i] < (mid_time - (period / 4)):
                y[i] = amplitude
            elif (mid_time - (period / 4)) < t[i] < (mid_time + (period / 4)):
                y[i] = amplitude * np.sin((t[i] - mid_time - period / 2) * x[1])
            elif (mid_time + (period / 4)) < t[i] < (final_t - (period / 4)):
                y[i] = -amplitude
            else:
                y[i] = -amplitude * np.sin((final_t - t[i]) * x[1])
        return y

    def curve_integral(self, x, t):
        integ = np.ndarray((t.shape[0],))
        integ[0] = 0
        for i in range(t.shape[0] - 1):
            integ[i + 1] = integrate.quad(lambda t: self.curve(x, t, self.final_t), 0, t[i + 1])[0]
        return integ + self.bow_limits[0]
