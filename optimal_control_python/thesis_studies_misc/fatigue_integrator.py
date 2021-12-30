from typing import Any, Callable
import numpy as np
from matplotlib import pyplot as plt
from time import perf_counter

from scipy import integrate


class FatigueIntegrator:
    def __init__(
        self,
        t: np.ndarray,
        target: Callable,
        fatigue_models: tuple,
        linestyles: tuple = ("-", "--", "-.", ".-")
    ):
        """
        Defining an integrator for the fatigue in bioptim

        Parameters
        ----------
        t: np.ndarray
            The time vector to evaluate to
        target: Callable
            The function(t, target_load) that returns the current target load
        fatigue_models: tuple
            A tuple of the models to integrate
        linestyles: tuple
            The linestyles for the matplotlib, len(linestyles) must be >= len(fatigue_models)
        """

        self._t: np.ndarray = t
        self._has_run: bool = False
        self._target: Callable = target
        self._fatigue_models: tuple = fatigue_models
        self._linestyles: tuple = linestyles
        self._performing_time: list = []

    def perform(self):
        """
        Perform the integration for all the fatigue_models
        """

        for i, fatigue in enumerate(self._fatigue_models):
            x0 = fatigue.default_initial_guess()

            starting_time = perf_counter()
            out: Any = integrate.solve_ivp(
                lambda t, x: self._dynamics(t, x, fatigue), [self._t[0], self._t[-1]], x0, t_eval=self._t
            )
            self._performing_time.append(perf_counter() - starting_time)

            self._add_result_to_plot(out.t, out.y, self._linestyles[i])
        self._has_run = True

    def plot_results(self):
        if not self._has_run:
            raise RuntimeError("run() must be called before plotting the results")
        plt.show()

    def print_integration_time(self):
        if not self._has_run:
            raise RuntimeError("run() must be called before printing the results")

        print(f"Individual integration time:")
        for model, t in zip(self._fatigue_models, self._performing_time):
            print(f"\t{type(model).__name__}: {t:1.3f} seconds")
        print(f"Total integration time: {sum(self._performing_time):1.3f} seconds")

    def _dynamics(self, t, x, fatigue):
        return np.array(fatigue.apply_dynamics(self._target(t) / fatigue.scaling, *x))[:, 0]

    def _add_result_to_plot(self, t: np.ndarray, out: np.ndarray, linestyle: Any):
        """
        Add a plot to the result

        Parameters
        ----------
        t
        out
        linestyle

        Returns
        -------

        """
        plt.plot(t, out[0, :], "tab:green", linestyle=linestyle)
        if out.shape[0] > 1:
            plt.plot(t, out[1, :], "tab:orange", linestyle=linestyle)
            plt.plot(t, out[2, :], "tab:red", linestyle=linestyle)
        if out.shape[0] > 3:
            plt.plot(t, out[3, :], "tab:gray", linestyle=linestyle)
        plt.plot(t, [self._target(_t) for _t in t], "tab:blue", alpha=0.5)
        plt.plot(t, np.sum(out[:4, :],  axis=0), "k")
