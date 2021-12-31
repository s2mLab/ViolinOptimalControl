from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter
from scipy import integrate

from .study_configuration import StudyConfiguration


class FatigueIntegrator:
    def __init__(self, study_configuration: StudyConfiguration):
        self.study = study_configuration
        self._has_run: bool = False
        self._performing_time: list[float] = []

    def perform(self):
        """
        Perform the integration for all the fatigue_models
        """

        for fatigue, linestyle in zip(self.study.fatigue_models.models, self.study.linestyles):
            x0 = fatigue.default_initial_guess()

            starting_time = perf_counter()
            out: Any = integrate.solve_ivp(
                lambda t, x: self._dynamics(t, x, fatigue), [self.study.t[0], self.study.t[-1]], x0, t_eval=self.study.t
            )
            self._performing_time.append(perf_counter() - starting_time)

            self._add_result_to_plot(out.t, out.y, linestyle)
        self._has_run = True

    def plot_results(self):
        if not self._has_run:
            raise RuntimeError("run() must be called before plotting the results")
        plt.show()

    def print_integration_time(self):
        if not self._has_run:
            raise RuntimeError("run() must be called before printing the results")

        print(f"Individual integration time:")
        for model, t in zip(self.study.fatigue_models.models, self._performing_time):
            print(f"\t{type(model).__name__}: {t:1.3f} seconds")
        print(f"Total integration time: {sum(self._performing_time):1.3f} seconds")

    def _dynamics(self, t, x, fatigue):
        return np.array(fatigue.apply_dynamics(self.study.target_function.function(t) / fatigue.scaling, *x))[:, 0]

    def _add_result_to_plot(self, t: np.ndarray, out: np.ndarray, linestyle: Any):
        plt.plot(t, out[0, :], "tab:green", linestyle=linestyle)
        if out.shape[0] > 1:
            plt.plot(t, out[1, :], "tab:orange", linestyle=linestyle)
            plt.plot(t, out[2, :], "tab:red", linestyle=linestyle)
        if out.shape[0] > 3:
            plt.plot(t, out[3, :], "tab:gray", linestyle=linestyle)
        plt.plot(t, [self.study.target_function.function(_t) for _t in t], "tab:blue", alpha=0.5)
        plt.plot(t, np.sum(out[:4, :],  axis=0), "k")
