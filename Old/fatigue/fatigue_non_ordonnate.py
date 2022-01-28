from pyoviz.BiorbdViz import BiorbdViz

import numpy as np

import biorbd_casadi as biorbd

import scipy.integrate as integrate

import scipy.interpolate as interpolate

import matplotlib.pyplot as plt


### Muscle parameters ###

## Slow fibers ##

S_Percent = 50  # percent of slow fibers in muscle
S_Specific_Tension = 1.0
F_S = 0.01  # fatigue rate
R_S = 0.002  # recovery rate
LD_S = 10  # development factor
LR_S = 10  # recovery factor

## Fast Fatigue Resistant fibers ##

FR_Percent = 25
FR_Specific_Tension = 2.0
F_FR = 0.05
R_FR = 0.01
LD_FR = 10
LR_FR = 10

## Fast Fatigable fibers ##

FF_Percent = 25
FF_Specific_Tension = 3.0
F_FF = 0.1
R_FF = 0.02
LD_FF = 10
LR_FF = 10

### Load ###

TL = 30  # percent of Maximal Voluntary Contraction
t_Max = 100

### Initial States ###

state_init_S0 = (0, 100, 0)
state_init_FR0 = (0, 100, 0)
state_init_FF0 = (0, 100, 0)


def defdyn(R, F, LD, LR):
    def dyn(t, X):
        (ma, mr, mf) = X
        if ma < TL:
            if mr > TL - ma:
                c = LD * (TL - ma)
            else:
                c = LD * mr
        else:
            c = LR * (TL - ma)

        madot = c - F * ma
        mrdot = -c + R * mf
        mfdot = F * ma - R * mf

        result = (madot, mrdot, mfdot)

        return result

    return dyn


dyn_S = defdyn(R_S, F_S, LD_S, LR_S)
dyn_FR = defdyn(R_FR, F_FR, LD_FR, LR_FR)
dyn_FF = defdyn(R_FF, F_FF, LD_FF, LR_FF)

X_S = integrate.solve_ivp(dyn_S, (0, t_Max), state_init_S0)
X_FR = integrate.solve_ivp(dyn_FR, (0, t_Max), state_init_FR0)
X_FF = integrate.solve_ivp(dyn_FF, (0, t_Max), state_init_FF0)


### Plot Activation
plt.figure(1)

plt.subplot(3, 1, 1)
plt.plot(X_S.t, X_S.y[0, :], label="Activated")
plt.plot(X_S.t, X_S.y[1, :], label="Resting")
plt.plot(X_S.t, X_S.y[2, :], label="Fatigued")
plt.title("Slow fibers")
plt.xlabel("time")
plt.ylabel("%MVC")

plt.subplot(3, 1, 2)
plt.plot(X_FR.t, X_FR.y[0, :], label="Activated")
plt.plot(X_FR.t, X_FR.y[1, :], label="Resting")
plt.plot(X_FR.t, X_FR.y[2, :], label="Fatigued")
plt.title("Fast Fatigue Resistant fibers")
plt.xlabel("time")
plt.ylabel("%MVC")

plt.subplot(3, 1, 3)
plt.plot(X_FF.t, X_FF.y[0, :], label="Activated")
plt.plot(X_FF.t, X_FF.y[1, :], label="Resting")
plt.plot(X_FF.t, X_FF.y[2, :], label="Fatigued")
plt.title("Fast Fatigable fibers")
plt.xlabel("time")
plt.ylabel("%MVC")

plt.legend()
plt.show()
