from scipy import optimize
import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as integrate


def curve(x, t):
    period = 2 * np.pi / x[1]
    if isinstance(t, float):
        t = np.array((t,))

    y = np.ndarray((t.shape[0],))
    for i in range(t.shape[0]):
        if t[i] < (period / 4):
            y[i] = x[0] * np.sin(t[i] * x[1])
        elif t[i] > (period / 4) and t[i] < (1 - (period / 4)):
            y[i] = x[0]
        elif t[i] > (1 - (period / 4)) and t[i] < (1 + (period / 4)):
            y[i] = x[0] * np.sin((t[i] - 1 - period / 2) * x[1])
        elif t[i] > (1 + (period / 4)) and t[i] < (2 - (period / 4)):
            y[i] = - x[0]
        else:
            y[i] = x[0] * np.sin((t[i] - 2) * x[1])
    return y

def curve_integral(x, t):
    integ = np.ndarray((t.shape[0]-1,))
    for i in range(t.shape[0]-1):
        integ[i] = integrate.quad(lambda t: curve(x, t), 0, t[i+1])[0]
    return integ

def generate_up_and_down_bow_target(ns, bow_speed=10, bow_acceleration=0.5):


    def objective_function(x, t):
        y = curve(x, t)
        integ = curve_integral(x,t)
        moy = (integ[int(n_points/2)-2] + integ[int(n_points/2)+2])/2
        return np.array((bow_acceleration-x[0], bow_speed-x[1], y[-1], y[0], (moy-0.48)*1000))

    n_points = 200
    t = np.linspace(0, 2, n_points)
    x_opt = optimize.least_squares(lambda x: objective_function(x, t), x0=np.array((1, 8))) # x[0] = amplitude et x[1]= 2 pi/ period
    return t, x_opt.x


if __name__ == "__main__":
    t, x = generate_up_and_down_bow_target(0)
    n_points = 200
    t = np.linspace(0, 2, n_points)
    plt.plot(t, curve(x, t))
    plt.plot(t[:-1], curve_integral(x, t), color="red")
    n_points = 50
    t = np.linspace(0, 2, n_points)
    plt.plot(t, curve(x, t), 'k.')
    plt.plot(t[:-1], curve_integral(x, t), '.m')
    plt.show()