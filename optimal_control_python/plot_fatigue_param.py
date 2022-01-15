# This is a debug script for testing and plotting fatigue

import time
import pickle

from scipy import integrate
import numpy as np
from matplotlib import pyplot as plt

from violin_ocp.violin import Violin, ViolinString
from bioptim import XiaFatigue, MichaudFatigue, EffortPerception, XiaTauFatigue, MichaudTauFatigue, TauEffortPerception


def target_load(t, target):
    if isinstance(target, (np.ndarray, tuple, list)):
        return target[int(t)]
    else:
        return target if t < 4000 else 0  # if t < 200 else target if t < 300 else 0


def dynamics(t, x, fatigue, target):
    return np.array(fatigue.apply_dynamics(target_load(t, target) / fatigue.scaling, *x))[:, 0]


def plot_result(t, out, target, linestyle):
    plt.plot(t, out[0, :], "tab:green", linestyle=linestyle)
    if out.shape[0] > 1:
        plt.plot(t, out[1, :], "tab:orange", linestyle=linestyle)
        plt.plot(t, out[2, :], "tab:red", linestyle=linestyle)
    if out.shape[0] > 3:
        plt.plot(t, out[3, :], "tab:gray", linestyle=linestyle)
    # plt.plot(t, [target_load(_t, target) for _t in t], "tab:blue", alpha=0.5)
    plt.plot(t, np.sum(out[:4, :], axis=0), "k")


def main():
    # file_path = "./results/900_cycles/900_cycles_non_fatigue_out.bo"
    # with open(file_path, "rb") as file:
    #     data = pickle.load(file)
    # target = data[1]["tau"][1, :]

    # target = np.random.rand(t_end + 1) / 2.5 + 0.1
    target = [5, 5]
    t_end = 600

    starting_time = time.time()
    n_points = 100000
    t = np.linspace(0, t_end, n_points)
    violin = Violin("WuViolin", ViolinString.E)

    fatigue_models = [
        MichaudFatigue(
            **violin.fatigue_parameters(
                MichaudTauFatigue, 1, index=0, effort_threshold=0, stabilization_factor=0, effort_factor=0
            )
        ),  # Behaves like Xia original
        MichaudFatigue(
            **violin.fatigue_parameters(MichaudTauFatigue, 1, index=0, effort_threshold=0, effort_factor=0)
        ),  # Behaves like Xia stabilized
        # MichaudFatigue(**violin.fatigue_parameters(MichaudTauFatigue, 1, index=0)),
        # EffortPerception(**violin.fatigue_parameters(TauEffortPerception, -1, index=0)),
        # EffortPerception(**violin.fatigue_parameters(TauEffortPerception, 1, index=0)),
    ]
    linestyles = ["-", "--"]

    for i, fatigue in enumerate(fatigue_models):
        x0 = fatigue.default_initial_guess()
        out = integrate.solve_ivp(lambda t, x: dynamics(t, x, fatigue, target[i]), [t[0], t[-1]], x0, t_eval=t)
        plot_result(out.t, out.y, target[i], linestyles[i])
    print(f"time: {time.time() - starting_time}")
    plt.show()


if __name__ == "__main__":
    main()
