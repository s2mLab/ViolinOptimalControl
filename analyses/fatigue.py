
import numpy as np

import biorbd

import scipy.integrate as integrate

import scipy.interpolate as interpolate

import matplotlib.pyplot as plt


### Muscle parameters ###

## Slow fibers ##
# TODO give better name for variables
S_Percent = 0.50 # percent of slow fibers in muscle
S_Specific_Tension = 1.0
F_S = 0.01 # fatigue rate
R_S = 0.002 # recovery rate
LD_S = 10 # development factor
LR_S = 10 # recovery factor

## Fast Fatigue Resistant fibers ##

FR_Percent = 0.25
FR_Specific_Tension = 2.0
F_FR = 0.05
R_FR = 0.01
LD_FR = 10
LR_FR = 10

## Fast Fatigable fibers ##

FF_Percent = 0.25
FF_Specific_Tension = 3.0
F_FF = 0.1
R_FF = 0.02
LD_FF = 10
LR_FF = 10

### Load ###

TL = 10 # percent of Maximal Voluntary Contraction
t_Max = 5000

### Initial States ###

state_init_S0 = (0, 100, 0) # ma, mr, mf
state_init_FR0 = (0, 100, 0)
state_init_FF0 = (0, 100, 0)
state_init_activation0 = (1, 0, 0)
state_init_0 = (0, 100, 0, 0, 100, 0, 0, 100, 0, 1, 0, 0)


def fatigue(T):

    def dyn(t, X):
        [ma_S, mr_S, mf_S, ma_FR, mr_FR, mf_FR, ma_FF, mr_FF, mf_FF, activ_S, activ_FR, activ_FF] = X

        # Residual capacity
        rc_S = S_Percent * (ma_S + mr_S)
        rc_FR = FR_Percent * (ma_FR)
        rc_FF = FF_Percent * (ma_FF)

        # Recruitment order
        activ_S = 1
        if TL > rc_S:
            activ_FR = 1
            if TL > rc_S + rc_FR:
                activ_FF = 1
            else:
                activ_FF = 0
        else:
            activ_FR = 0

        # Conditions of evolution
        def defdyn(R, F, LD, LR, ma, mr, mf, activ, T, percent):
            if ma < T:
                if mr > T - ma:
                    c = LD * (T - ma)  # development & not fatigued
                else:
                    c = LD * mr  # development & fatigued
            else:
                c = LR * (T - ma)  # recovery
            c = percent * c

            madot = c * activ - F * ma
            mrdot = -c * activ + R * mf
            mfdot = F * ma - R * mf

            return madot, mrdot, mfdot

        (madot_S, mrdot_S, mfdot_S) = defdyn(R_S, F_S, LD_S, LR_S, ma_S, mr_S, mf_S, activ_S, T/S_Percent, S_Percent)
        (madot_FR, mrdot_FR, mfdot_FR) = defdyn(R_FR, F_FR, LD_FR, LR_FR, ma_FR, mr_FR, mf_FR, activ_FR, (T - S_Percent*ma_S)/FR_Percent, FR_Percent)
        (madot_FF, mrdot_FF, mfdot_FF) = defdyn(R_FF, F_FF, LD_FF, LR_FF, ma_FF, mr_FF, mf_FF, activ_FF, (T - S_Percent*ma_S - FR_Percent*ma_FR)/FF_Percent, FF_Percent)

        return madot_S, mrdot_S, mfdot_S, madot_FR, mrdot_FR, mfdot_FR, madot_FF, mrdot_FF, mfdot_FF, activ_S, activ_FR, activ_FF
    return dyn


X = integrate.solve_ivp(fatigue(TL), (0, t_Max), state_init_0)
t = X.t
X_S = (X.y[0, :], X.y[1, :], X.y[2, :])
X_FR = (X.y[3, :], X.y[4, :], X.y[5, :])
X_FF = (X.y[6, :], X.y[7, :], X.y[8, :])
activation =(X.y[9, :], X.y[10, :], X.y[11, :])

## Brain effort
BE_S = TL/(X.y[0, :] + X.y[1, :])
for i in range(len(BE_S)):
    if BE_S[i] >= 1:
        BE_S[i] = 1

BE_FR = TL/(X.y[3, :] + X.y[4, :])
for i in range(len(BE_FR)):
    if BE_FR[i] >= 1:
        BE_FR[i] = 1

BE_FF = TL/(X.y[6, :] + X.y[7, :])
for i in range(len(BE_FF)):
    if BE_FF[i] >= 1:
        BE_FF[i] = 1

## Total activity
ma_total = S_Percent*X.y[0, :] + FR_Percent*X.y[3, :] + FF_Percent*X.y[6, :]

## Endurance Time


# TODO make sure when changing TL, endur_time is not affected (function outside of the file)
# TODO verify why under 10% there is no fatiguability
def endur_time(T, tmax, state_init):
    x = integrate.solve_ivp(fatigue(T), (0, tmax), state_init)
    t = x.t

    ma_t = S_Percent * x.y[0, :] + FR_Percent * x.y[3, :] + FF_Percent * x.y[6, :]

    ET = 0
    for i in range(len(ma_t)):
        if abs((ma_t[i] - T) / T) < 0.05:
            ET = t[i]

    return ET


Target_Load = np.linspace(1, 100, 20)
ET = np.ndarray(len(Target_Load))
for i in range(len(Target_Load)):
    ET[i] = endur_time(Target_Load[i], 2000, state_init_0)
    print(ET[i])

### Plot ###
plt.figure(1)

plt.subplot(4, 1, 1)
plt.plot(t, X_S[0]*S_Percent, label='Force developed by active')
plt.plot(t, X_S[1]*S_Percent, label='Potential force from resting')
plt.plot(t, X_S[2]*S_Percent, label='Force lost from fatigue')
plt.title("Slow fibers")
plt.xlabel('time')
plt.ylabel('%MVC')

plt.subplot(4, 1, 2)
plt.plot(t, X_FR[0]*FR_Percent, label = 'Force developed by active')
plt.plot(t, X_FR[1]*FR_Percent, label = 'Potential force from resting')
plt.plot(t, X_FR[2]*FR_Percent, label = 'Force lost from fatigue')
plt.title("Fast Fatigue Resistant fibers")
plt.xlabel('time')
plt.ylabel('%MVC')

plt.subplot(4, 1, 3)
plt.plot(t, X_FF[0]*FF_Percent, label = 'Force developed by active')
plt.plot(t, X_FF[1]*FF_Percent, label = 'Potential force from resting')
plt.plot(t, X_FF[2]*FF_Percent, label = 'Force lost from fatigue')
plt.title("Fast Fatigable fibers")
plt.xlabel('time')
plt.ylabel('%MVC')
plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0)

plt.subplot(4, 1, 4)
plt.plot(t, ma_total)
plt.xlabel('time')
plt.ylabel('%MVC')


plt.figure(2)

plt.subplot(3, 1, 1)
plt.plot(t, BE_S)
plt.title("Slow fibers")
plt.xlabel('time')
plt.ylabel('Brain effort')

plt.subplot(3, 1, 2)
plt.plot(t, BE_FR)
plt.title("Fast Fatigue Resistant fibers")
plt.xlabel('time')
plt.ylabel('Brain effort')

plt.subplot(3, 1, 3)
plt.plot(t, BE_FF)
plt.title("Fast Fatigable fibers")
plt.xlabel('time')
plt.ylabel('Brain effort')


plt.figure(3)

plt.plot(Target_Load, ET)
plt.title("Endurance Time")
plt.xlabel('Target Load (%MVC)')
plt.ylabel('Time')

plt.show()

