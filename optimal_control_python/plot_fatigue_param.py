from scipy import integrate
import numpy as np
from matplotlib import pyplot as plt

from violin_ocp.violin import Violin, ViolinString
from bioptim import XiaFatigue, MichaudFatigue, XiaTauFatigue, MichaudTauFatigue


def target_load(t):
    return 0.8 if t < 150 else 0.16


def dynamics(t, x, fatigue):
    return np.array(fatigue.apply_dynamics(target_load(t), *x))[:, 0]


def plot_result(t, out, linestyle):
    plt.plot(t, out[0, :], "tab:green", linestyle=linestyle)
    plt.plot(t, out[1, :], "tab:orange", linestyle=linestyle)
    plt.plot(t, out[2, :], "tab:red", linestyle=linestyle)
    plt.plot(t, [target_load(_t) for _t in t], "k", alpha=0.5)


def main():
    t_end = 600
    n_points = 10000
    t = np.linspace(0, t_end, n_points)
    violin = Violin("WuViolin", ViolinString.E)
    fatigue_models = [
        XiaFatigue(**violin.fatigue_parameters(1, XiaTauFatigue)),
        MichaudFatigue(**violin.fatigue_parameters(1, MichaudTauFatigue)),
    ]
    linestyles = ["-", "--"]

    for i, fatigue in enumerate(fatigue_models):
        x0 = fatigue.default_initial_guess()
        out = integrate.solve_ivp(lambda t, x: dynamics(t, x, fatigue), [t[0], t[-1]], x0, t_eval=t).y
        plot_result(t, out, linestyles[i])
    plt.show()


if __name__ == "__main__":
    main()
