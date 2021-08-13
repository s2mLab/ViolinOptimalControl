import numpy as np

import biorbd_casadi as biorbd

import scipy.integrate as integrate

import scipy.interpolate as interpolate

import matplotlib.pyplot as plt

# Parameters
t_Max = 1500
EMG_target = 1
state_init = (0.5, 0, 0.5)
n_frames = 300

# Load model for biorbd
model = biorbd.Model("../models/conv-arm26.bioMod")
muscle = model.muscleGroup(0).muscle(0)


# Muscle parameters for slow fibers
S_Percent = 0.50  # percent of slow fibers in muscle
S_Specific_Tension = 1.0  # each muscle fiber type can develop its own unit force
fatigue_rate_s = 0.01  # fatigue rate
recovery_rate_s = 0.002  # recovery rate
develop_factor_s = 10  # development factor
recovery_factor_s = 10  # recovery factor

# Parameters for EMG
target_load = 0.3
delta_t = 0.01
delta_amp = 0.15
freq_load = 10  # Hz

ma_recup = list()
mf_recup = list()
mr_recup = list()


# Define activation command
def var_load(_t, _target_load=target_load, _delta_t=delta_t):
    if _t <= delta_t:
        return _target_load/_delta_t*_t
    else:
        return _target_load


def var_sin_load(_t, freq=freq_load, _delta_amp=delta_amp, _target_load=target_load):
    return _target_load*(1 + _delta_amp*np.math.sin(freq*_t))


# Define model of fatigue
def def_dyn(fun_load, recovery_rate, fatigue_rate, develop_factor, recovery_factor):

    def dyn(t, x):
        _load = fun_load(t)
        (ma, mf, mr) = x
        if ma < _load:
            if mr > _load - ma:
                command = develop_factor*(_load-ma)
            else:
                command = develop_factor*mr
        else:
            command = recovery_factor*(_load-ma)

        ma_dot = command - fatigue_rate*ma
        mr_dot = -command + recovery_rate*mf
        mf_dot = fatigue_rate * ma - recovery_rate * mf

        result = (ma_dot, mf_dot, mr_dot)

        return result

    return dyn


# Get values from biorbd
def fatigue_dyn_biorbd(_model, _muscle, _q, _q_dot, fun_load, is_state, is_muscle_updated=True):
    _fatigue_model = biorbd.HillThelenTypeFatigable_DeepCopy(_muscle)
    _fatigue_state = biorbd.s2mMuscleFatigueDynamicStateXia_getRef(_fatigue_model.fatigueState()) # needs correct function
    if is_state and type(fun_load) != biorbd.s2mMuscleStateActual:
        print("Warning: command function is not of type s2mMuscleStateActual")
        return 1
    if type(fun_load) == biorbd.s2mMuscleStateActual:
        is_state = True

    def dyn(t, x):
        if not is_state:
            _load = fun_load(t)
            _emg = biorbd.s2mMuscleStateActual(0, _load)
        else:
            _emg = fun_load

        (ma, mf, mr) = x
        _fatigue_state.setState(ma, mf, mr)
        _model.updateMuscles(_model, _q, _q_dot, is_muscle_updated)
        _fatigue_model.computeFlCE(_emg)
        _fatigue_model.computeTimeDerivativeState(_emg)
        ma_dot = _fatigue_state.activeFibersDot()
        mf_dot = _fatigue_state.fatiguedFibersDot()
        mr_dot = _fatigue_state.restingFibersDot()
        result = (ma_dot, mf_dot, mr_dot)

        return result

    return dyn


# Create functions to integrate
dyn_S = def_dyn(var_sin_load, recovery_rate_s, fatigue_rate_s, develop_factor_s, recovery_factor_s)
dyn_biorbd = fatigue_dyn_biorbd(model, muscle, biorbd.s2mGenCoord(model), biorbd.s2mGenCoord(model), biorbd.s2mMuscleStateActual(0, 0.5), True, True)

# Integration
X_S = integrate.solve_ivp(dyn_S, (0, t_Max), state_init)
X_biorbd = integrate.solve_ivp(dyn_biorbd, (0, t_Max), state_init)

# Interpolation for error
Y = list()
Y_biorbd = list()
time = X_S.t
time_biorbd = X_biorbd.t
T = np.linspace(0, time[len(time)-1], n_frames)
for i in range(3):
    tck = interpolate.splrep(time, X_S.y[i, :], s=0)
    Y.append(interpolate.splev(T, tck, der=0))

    tck_biorbd = interpolate.splrep(time_biorbd, X_biorbd.y[i, :], s=0)
    Y_biorbd.append(interpolate.splev(T, tck_biorbd, der=0))

error_ma_absolute = (Y_biorbd[0][:] - Y[0][:])
error_mf_absolute = (Y_biorbd[1][:] - Y[1][:])
error_mr_absolute = (Y_biorbd[2][:] - Y[2][:])

error_ma_relative = (Y_biorbd[0][:] - Y[0][:])/Y[0][:]*100
error_mf_relative = (Y_biorbd[1][:] - Y[1][:])/Y[1][:]*100
error_mr_relative = (Y_biorbd[2][:] - Y[2][:])/Y[2][:]*100

# Error of integration and processing
error_total_biorbd = X_biorbd.y[0, :] + X_biorbd.y[1, :] + X_biorbd.y[2, :] - np.ones(X_biorbd.y[0, :].size)
error_total_biorbd_2 = list()
for i in range(len(ma_recup)):
    error_total_biorbd_2.append(ma_recup[i] + mr_recup[i] + mf_recup[i] - 1)
max_error = max(max(abs(error_ma_relative)), max(abs(error_mf_relative)), max(abs(error_mr_relative)))
print(max_error)

# Plot results
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(X_S.t, X_S.y[0, :], label='Activated')
plt.plot(X_S.t, X_S.y[1, :], label='Fatigued')
plt.plot(X_S.t, X_S.y[2, :], label='Resting')
plt.title("Slow fibers fatigue")
plt.xlabel('time')
plt.ylabel('%MVC')

plt.subplot(2, 1, 2)
plt.plot(X_biorbd.t, X_biorbd.y[0, :], label='Activated')
plt.plot(X_biorbd.t, X_biorbd.y[1, :], label='Fatigued')
plt.plot(X_biorbd.t, X_biorbd.y[2, :], label='Resting')
plt.title("Slow fibers fatigue from biorbd")
plt.xlabel('time')
plt.ylabel('%MVC')

plt.legend()

plt.figure(2)
plt.plot(X_biorbd.t, error_total_biorbd)
plt.title("Total error of integration and processing")
plt.xlabel('time')
plt.ylabel('%MVC')

plt.figure(3)
plt.plot(T, error_ma_absolute, label = 'error_Activated')
plt.plot(T, error_mf_absolute, label = 'error_Fatigued')
plt.plot(T, error_mr_absolute, label = 'error_Resting')
plt.title("Absolute Error from biorbd")
plt.xlabel('time')
plt.ylabel('%error')

plt.legend()

plt.figure(4)
plt.plot(T, error_ma_relative, label = 'error_relative_Activated')
plt.plot(T, error_mf_relative, label = 'error_relative_Fatigued')
plt.plot(T, error_mr_relative, label = 'error_relative_Resting')
plt.title("Relative Error from biorbd")
plt.xlabel('time')
plt.ylabel('%error')
plt.legend()

plt.figure(5)
plt.plot(error_total_biorbd_2)
plt.title("Total error of integration and processing 2")
plt.xlabel('time')
plt.ylabel('%MVC')

plt.show()
