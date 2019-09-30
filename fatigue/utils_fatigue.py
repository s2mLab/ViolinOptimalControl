import numpy as np
import biorbd


# Define activation command through time
def fun_const(_target_load=0.3, _delta_t=0.1):
    def var_load(_t):
        if _t <= _delta_t:
            return _target_load/_delta_t*_t
        else:
            return _target_load
    return var_load


def fun_sin(freq, _delta_amp, _target_load):
    def var_load(_t):
        return _target_load*(1 + _delta_amp*np.math.sin(freq*_t))
    return var_load


# Define model of fatigue
def def_dyn(fun_load=fun_const(), recovery_rate=0.002, fatigue_rate=0.01, develop_factor=10, recovery_factor=10):

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
def fatigue_dyn_biorbd(_model, _muscle, _q, _q_dot, fun_load, is_s2m_muscle_state_actual=False, is_muscle_updated=True, is_flce_computed=True):
    _fatigue_model = biorbd.s2mMuscleHillTypeThelenFatigable_getRef(_muscle)
    _fatigue_state = biorbd.s2mMuscleFatigueDynamicStateXia_getRef(_fatigue_model.fatigueState())
    if is_s2m_muscle_state_actual and type(fun_load) != biorbd.s2mMuscleStateActual:
        print("Warning: command function is not of type s2mMuscleStateActual")
        return 1
    if type(fun_load) == biorbd.s2mMuscleStateActual:
        is_s2m_muscle_state_actual = True

    def dyn(t, x):
        if not is_s2m_muscle_state_actual:
            _load = fun_load(t)
            _emg = biorbd.s2mMuscleStateActual(0, _load)
        else:
            _emg = fun_load

        (ma, mf, mr) = x
        _fatigue_state.setState(ma, mf, mr)
        _model.updateMuscles(_model, _q, _q_dot, is_muscle_updated)
        if is_flce_computed:
            _fatigue_model.computeFlCE(_emg)
        _fatigue_model.computeTimeDerivativeState(_emg)
        ma_dot = _fatigue_state.activeFibersDot()
        mf_dot = _fatigue_state.fatiguedFibersDot()
        mr_dot = _fatigue_state.restingFibersDot()
        result = (ma_dot, mf_dot, mr_dot)

        return result

    return dyn



