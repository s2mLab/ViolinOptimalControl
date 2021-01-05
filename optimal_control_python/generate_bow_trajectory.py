from scipy import optimize
import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as integrate

from utils import Violin

violin = Violin("E")
min_bow, max_bow = violin.initial_position()["frog"][-1], violin.initial_position()["tip"][-1]


def curve(x, t):
    period = 2 * np.pi / x[1]
    if isinstance(t, float):
        t = np.array((t,))

    y = np.ndarray((t.shape[0],))
    for i in range(t.shape[0]):
        if t[i] < (period / 4):
            y[i] = x[0] * np.sin(t[i] * x[1])
        elif (period / 4) < t[i] < (1 - (period / 4)):
            y[i] = x[0]
        elif (1 - (period / 4)) < t[i] < (1 + (period / 4)):
            y[i] = x[0] * np.sin((t[i] - 1 - period / 2) * x[1])
        elif (1 + (period / 4)) < t[i] < (2 - (period / 4)):
            y[i] = - x[0]
        else:
            y[i] = x[0] * np.sin((t[i] - 2) * x[1])
    return y


def curve_integral(x, t):
    integ = np.ndarray((t.shape[0],))
    integ[0] = 0
    for i in range(t.shape[0]-1):
        integ[i+1] = integrate.quad(lambda t: curve(x, t), 0, t[i+1])[0]
    return integ + min_bow


def generate_bow_trajectory(n_points, bow_speed=10, bow_acceleration=0.5):
    # x[0] = amplitude
    # x[1]= 2 * pi / period
    def objective_function(x, t):
        y = curve(x, t)
        integ = curve_integral(x, t)
        moy = (integ[int(n_points/2)-2] + integ[int(n_points/2)+2])/2
        return np.array((bow_acceleration - x[0], bow_speed - x[1], y[-1], y[0], (moy - max_bow) * 1000))

    t = np.linspace(0, 2, n_points)
    x_opt = optimize.least_squares(lambda x: objective_function(x, t), x0=np.array((1, 8)))
    return x_opt.x


if __name__ == "__main__":
    n_points = 200
    x = generate_bow_trajectory(n_points=n_points)

    t = np.linspace(0, 2, n_points)
    plt.title("Speed and position of the virtual contact on the bow during the up and down movement")
    plt.plot(t, curve(x, t))
    plt.plot(t, curve_integral(x, t), color="red")

    n_points = 50
    t = np.linspace(0, 2, n_points)
    plt.plot(t, curve(x, t), 'k.')

    plt.show()
