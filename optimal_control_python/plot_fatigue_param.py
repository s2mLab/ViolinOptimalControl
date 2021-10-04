import time

from scipy import integrate
import numpy as np
from matplotlib import pyplot as plt

from violin_ocp.violin import Violin, ViolinString
from bioptim import XiaFatigue, MichaudFatigue, EffortPerception, XiaTauFatigue, MichaudTauFatigue, TauEffortPerception


def target_load(t, target):
    if isinstance(target, (np.ndarray, tuple, list)):
        return target[int(t)]
    else:
        return target if t < 1500 else 0 # if t < 200 else target if t < 300 else 0


def dynamics(t, x, fatigue, target):
    return np.array(fatigue.apply_dynamics(target_load(t, target), *x))[:, 0]


def plot_result(t, out, target, linestyle):
    plt.plot(t, out[0, :], "tab:green", linestyle=linestyle)
    if out.shape[0] > 1:
        plt.plot(t, out[1, :], "tab:orange", linestyle=linestyle)
        plt.plot(t, out[2, :], "tab:red", linestyle=linestyle)
    if out.shape[0] > 3:
        plt.plot(t, out[3, :], "tab:gray", linestyle=linestyle)
    plt.plot(t, [target_load(_t, target) for _t in t], "tab:blue", alpha=0.5)
    plt.plot(t, np.sum(out[:4, :],  axis=0), "k")


def main():
    # target = np.random.rand(t_end + 1) / 2.5 + 0.1
    target = 1
    t_end = 3000

    starting_time = time.time()
    n_points = 10000
    t = np.linspace(0, t_end, n_points)
    violin = Violin("WuViolin", ViolinString.E)
    fatigue_models = [
        # XiaFatigue(**violin.fatigue_parameters(XiaTauFatigue, 1)),
        # MichaudFatigue(**violin.fatigue_parameters(MichaudTauFatigue, 1)),
        EffortPerception(**violin.fatigue_parameters(TauEffortPerception, 1)),
    ]
    linestyles = ["-", "--"]

    for i, fatigue in enumerate(fatigue_models):
        x0 = fatigue.default_initial_guess()
        out = integrate.solve_ivp(lambda t, x: dynamics(t, x, fatigue, target), [t[0], t[-1]], x0, t_eval=t)
        plot_result(out.t, out.y, target, linestyles[i])
    print(f"time: {time.time() - starting_time}")
    plt.show()


if __name__ == "__main__":
    main()
