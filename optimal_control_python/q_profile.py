from scipy import optimize
import numpy as np
from matplotlib import pyplot as plt


def curve(x, t, n_points):
    y = np.ndarray((n_points,))
    period = 2 * np.pi / x[1]
    for i in range(int(n_points)):
        if t[i] < period / 4:
            y[i] = x[0] * np.sin(t[i] * x[1])
        elif t[i] > period / 4 and t[i] < (1 - (period / 4)):
            y[i] = x[0]
        elif t[i] > (1 - (period / 4)) and t[i] < (1 + (period / 4)):
            y[i] = x[0] * np.sin((t[i] - 1 - period / 2) * x[1])
        elif t[i] > (1 + (period / 4)) and t[i] < (2 - (period / 4)):
            y[i] = - x[0]
        elif t[i] > 2 - period / 4:
            y[i] = x[0] * np.sin((t[i] - 2) * x[1])
    return y


def generate_up_and_down_bow_target(ns, bow_speed=10, bow_acceleration=0.5):
    # plt.plot(t, curve(np.array([0.5, 0.5])))
    # plt.show()

    def bound_computation(x):
        y = curve(x)
        return np.array([y[-1], y[0]])

    def objective_function(x, t, n_points):
        y = curve(x, t, n_points)
        return np.array((bow_acceleration-x[0], bow_speed-x[1]))

    n_points = 200
    t = np.linspace(0, 2, n_points)
    x_opt = optimize.least_squares(objective_function, x0=np.array((0, 0)), args=(t, n_points)) # x[0] = amplitude et x[1]= 2 pi/ period
    # y_out = curve_integral(x_opt.x)
    # decouper y_out en ns
    return t, x_opt.x


if __name__ == "__main__":
    t, x = generate_up_and_down_bow_target(0)
    n_points = 200
    t = np.linspace(0, 2, n_points)
    plt.plot(t, curve(x, t, n_points))
    n_points = 50
    t = np.linspace(0, 2, n_points)
    plt.plot(t, curve(x, t, n_points), 'k.')
    plt.show()