from scipy import optimize
import numpy as np
from matplotlib import pyplot as plt

n_shoot = 200
t = np.linspace(0, 2, n_shoot)


def curve(x):
    y = np.ndarray((n_shoot, ))

    y[:int(n_shoot / 6)] = x[0] * np.sin(t[:int(n_shoot / 6)] * x[1]) # partie 1 de la courbe, sinus
    y[int(n_shoot / 6):int(n_shoot / 3)] = x[0]  # partie 2 de la courbe, constante
    y[int(n_shoot / 3):int((2 * n_shoot / 3)-4)] = x[0] * np.sin((t[:int((n_shoot / 3)-3)]+(3.14/2)) * x[1])
    y[int((2* n_shoot / 3)-4):int((5*n_shoot/6)+1)] = -x[0]
    y[int((5*n_shoot/6)+1):] = x[0] * np.sin((t[int((5* n_shoot / 6)+1):]+3) * x[1])

    # y[:int(n_shoot / 4)] = x[0] * np.sin(t[:int(n_shoot / 4)] * x[1] + x[3]) + x[2]  # version 2

    # for i in range(int(n_shoot / 4)):  # version 1
    #     y[i] = x[0] * np.sin(t[i] * x[1] + x[3]) + x[2]

    # y[:int(n_shoot / 4)] = x[0] * np.sin(t[:int(n_shoot / 4)] * x[1] + x[3]) + x[2] # version 2

    return y

# plt.plot(t, curve(np.array([0.5, 0.5])))
# plt.show()


def bound_computation(x):
    y = curve(x)
    return y[-1]


def objective_function(x, *args, **kwargs):
    y = curve(x)
    return np.array((0.75-x[0], y[-1]))
    # return np.array((0.5-x[0], y[-1])) # formule de base


x_opt = optimize.least_squares(objective_function, x0=np.array((0.70, 5, 3, 4)))
plt.plot(t, curve(x_opt.x))
plt.show()
