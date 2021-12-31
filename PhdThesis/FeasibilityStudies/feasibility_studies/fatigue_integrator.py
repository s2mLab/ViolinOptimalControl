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
        self._results: list[np.ndarray, ...] = []
        self._performing_time: list[float, ...] = []
        self.axes = plt.axes()

    def perform(self):
        """
        Perform the integration for all the fatigue_models
        """

        t_span = (self.study.t[0], self.study.t[-1])
        t_eval = self.study.t
        for fatigue, x0, plot_options in zip(self.study.fatigue_models.models, self.study.x0, self.study.plot_options):
            starting_time = perf_counter()
            out: Any = integrate.solve_ivp(lambda t, x: self._dynamics(t, x, fatigue), t_span, x0, t_eval=t_eval)
            self._performing_time.append(perf_counter() - starting_time)

            self._results.append(out.y)
            self._add_result_to_plot(out.t, out.y, plot_options)

        self._has_run = True

    def plot_results(self):
        if not self._has_run:
            raise RuntimeError("run() must be called before plotting the results")
        self.axes.set_ylim((0, 1))
        plt.show()

    def print_final_sum(self):
        if not self._has_run:
            raise RuntimeError("run() must be called before printing the results")

        print("Sum of components at the final index")
        for model, results in zip(self.study.fatigue_models.models, self._results):
            print(f"\t{type(model).__name__}: {np.sum(results, axis=0)[0]}")

    def print_integration_time(self):
        if not self._has_run:
            raise RuntimeError("run() must be called before printing the results")

        print(f"Individual integration time:")
        for model, t in zip(self.study.fatigue_models.models, self._performing_time):
            print(f"\t{type(model).__name__}: {t:1.3f} seconds")
        print(f"Total integration time: {sum(self._performing_time):1.3f} seconds")

    def print_rmse(self):
        if not self._has_run:
            raise RuntimeError("run() must be called before printing the results")

        if len(self.study.fatigue_models.models) != 2:
            raise RuntimeError("rmse must have exactly 2 models to be called")

        if self.study.rms_indices is None:
            raise ValueError("rms_indices were not provided in the study configuration")

        # Get aliases
        models = self.study.fatigue_models.models
        idx = self.study.rms_indices

        e = self._results[0][idx[0], :] - self._results[1][idx[0], :]
        se = e**2
        mse = np.sum(se, axis=1) / self.study.n_points
        rmse = np.sqrt(mse)

        print(f"The RMSE between {type(models[0]).__name__} and {type(models[1]).__name__} is {rmse}")

    def _dynamics(self, t, x, fatigue):
        return np.array(fatigue.apply_dynamics(self.study.target_function.function(t) / fatigue.scaling, *x))[:, 0]

    def _add_result_to_plot(self, t: np.ndarray, out: np.ndarray, plot_options: Any):
        plt.plot(t, out[0, :], color="tab:green", **plot_options)
        if out.shape[0] > 1:
            plt.plot(t, out[1, :], color="tab:orange", **plot_options)
            plt.plot(t, out[2, :], color="tab:red", **plot_options)
        if out.shape[0] > 3:
            plt.plot(t, out[3, :], "tab:gray", **plot_options)
        plt.plot(t, [self.study.target_function.function(_t) for _t in t], color="tab:blue", alpha=0.5, **plot_options)
        plt.plot(t, np.sum(out[:4, :],  axis=0), color="black", **plot_options)
