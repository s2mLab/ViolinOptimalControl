from types import SimpleNamespace

import numpy as np
import biorbd
from scipy import integrate, interpolate
from matplotlib import pyplot as plt


def read_acado_output_states(file_path, biorbd_model, nb_nodes, nb_phases):
    # Get some values from the model
    nb_dof_total = biorbd_model.nbQ() + biorbd_model.nbQdot()

    # Create some aliases
    nb_points = (nb_phases * nb_nodes) + 1

    i = 0
    t = np.ndarray(nb_points)  # initialization of the time

    # initialization of the derived states
    all_q = np.ndarray((biorbd_model.nbQ(), nb_points))  # initialization of the states nbQ lines and nbP columns
    all_qdot = np.ndarray((biorbd_model.nbQdot(), nb_points))
    with open(file_path, "r") as data:
        # Nodes first lines
        for l in range(nb_nodes):
            line = data.readline()
            lin = line.split('\t')  # separation of the line in element
            lin[:1] = []  # remove the first element ( [ )
            lin[(nb_phases * biorbd_model.nbQ()) + (nb_phases * biorbd_model.nbQdot()) + 1:] = []  # remove the last ]

            t[i] = float(lin[0])  # complete the time with the first column
            for p in range(nb_phases):
                all_q[:, i + p * nb_nodes] = [float(j) for j in lin[
                                                               1 + p * nb_dof_total:biorbd_model.nbQ() + p * nb_dof_total + 1]]  # complete the states with the nQ next columns
                all_qdot[:, i + p * nb_nodes] = [float(k) for k in lin[biorbd_model.nbQ() + 1 + p * nb_dof_total:nb_dof_total * (
                            p + 1) + 1]]  # complete the states with the nQdot next columns
            i += 1

        # Last line
        line = data.readline()
        lin = line.split('\t')  # separation of the line in element
        lin[:1] = []  # remove the first element ( [ )
        lin[(nb_phases * biorbd_model.nbQ()) + (nb_phases * biorbd_model.nbQdot()) + 1:] = []  # remove the last ( ] )
        t[i] = float(lin[0])
        all_q[:, -1] = [float(j) for j in lin[1 + (nb_phases - 1) * nb_dof_total:biorbd_model.nbQ() + (
                    nb_phases - 1) * nb_dof_total + 1]]  # complete the states with the nQ next columns
        all_qdot[:, -1] = [float(k) for k in lin[biorbd_model.nbQ() + 1 + (nb_phases - 1) * nb_dof_total:nb_dof_total * nb_phases + 1]]

    return t, all_q, all_qdot


def read_acado_output_controls(file_path, nb_nodes, nb_phases, nb_controls):
    i = 0

    # Create some aliases
    nb_points = (nb_phases * nb_nodes) + 1

    all_u = np.ndarray((nb_controls, nb_points))
    with open(file_path, "r") as fichier_u:

        for l in range(nb_nodes):
            line = fichier_u.readline()
            lin = line.split('\t')
            lin[:1] = []
            lin[(nb_phases*nb_controls) + 1:] = []

            for p in range(nb_phases):
                all_u[:, i+p*nb_nodes] = [float(i) for i in lin[1+p*nb_controls:nb_controls*(p+1)+1]]

            i += 1

        line = fichier_u.readline()
        lin = line.split('\t')
        lin[:1] = []
        lin[(nb_phases*nb_controls) + 1:] = []

        all_u[:, -1] = [float(i) for i in lin[1+(nb_phases-1)*nb_controls:nb_controls*nb_phases+1]]
    return all_u


def organize_time(file_path, t, nb_phases, nb_nodes, parameter=True):
    if parameter:
        with open(file_path, "r") as fichier_p:
            line = fichier_p.readline()
            lin = line.split('\t')
            time_parameter = [float(i) for i in lin[2:nb_phases+2]]
        t_final = t*time_parameter[0]

        for p in range(1, nb_phases):
            for j in range(nb_nodes+1):
                t_final[(nb_nodes*p)+j] = t_final[(nb_nodes*p)-1] + (t[j + 1]*time_parameter[p])
    else:
        t_final = t
        for p in range(1, nb_phases):
            for j in range(nb_nodes+1):
                t_final[(nb_nodes*p)+j] = t_final[(nb_nodes*p)-1] + t[j + 1]
    return t_final


def dynamics_from_muscles(t_int, states, biorbd_model, u):
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_muscle = biorbd_model.nbMuscleTotal()

    states_dynamics = biorbd.VecBiorbdMuscleStateDynamics(nb_muscle)
    for i in range(len(states_dynamics)):
        states_dynamics[i] = biorbd.StateDynamics(0, u[i])

    biorbd_model.updateMuscles(biorbd_model, states[:nb_q], states[nb_q:], True)
    tau = biorbd.Model.muscularJointTorque(biorbd_model, states_dynamics, states[:nb_q], states[nb_q:])
    qddot = biorbd.Model.ForwardDynamics(biorbd_model, states[:nb_q], states[nb_q:], tau).get_array()

    rsh = np.ndarray(nb_q + nb_qdot)
    for i in range(nb_q):
        rsh[i] = states[nb_q+i]
        rsh[i + nb_q] = qddot[i]

    return rsh


def dynamics_from_muscles_and_torques(t_int, states, biorbd_model, u):
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_tau = biorbd_model.nbGeneralizedTorque()
    nb_muscle = biorbd_model.nbMuscleTotal()

    states_dynamics = biorbd.VecBiorbdMuscleStateDynamics(nb_muscle)
    for i in range(len(states_dynamics)):
        states_dynamics[i] = biorbd.StateDynamics(0, u[i])

    biorbd_model.updateMuscles(states[:nb_q], states[nb_q:], True)
    tau = biorbd.Model.muscularJointTorque(biorbd_model, states_dynamics, states[:nb_q], states[nb_q:])

    tau_final = tau.get_array() + u[nb_muscle:nb_muscle+nb_tau]

    qddot = biorbd.Model.ForwardDynamics(biorbd_model, states[:nb_q], states[nb_q:], tau_final).get_array()

    rsh = np.ndarray(nb_q + nb_qdot)
    for i in range(nb_q):
        rsh[i] = states[nb_q+i]
        rsh[i + nb_q] = qddot[i]
    return rsh


def dynamics_from_muscles_and_torques_and_contact(t_int, states, biorbd_model, u):
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    nb_tau = biorbd_model.nbGeneralizedTorque()
    nb_muscle = biorbd_model.nbMuscleTotal()

    states_dynamics = biorbd.VecBiorbdMuscleStateDynamics(nb_muscle)
    for i in range(len(states_dynamics)):
        states_dynamics[i] = biorbd.StateDynamics(0, u[i])

    biorbd_model.updateMuscles(states[:nb_q], states[nb_q:], True)
    tau = biorbd.Model.muscularJointTorque(biorbd_model, states_dynamics, states[:nb_q], states[nb_q:])

    tau_final = tau.get_array() + u[nb_muscle:nb_muscle + nb_tau]

    cs = biorbd_model.getConstraints()
    qddot = biorbd.Model.ForwardDynamicsConstraintsDirect(biorbd_model, states[:nb_q], states[nb_q:],
                                                          tau_final, cs).get_array()
    rsh = np.ndarray(nb_q + nb_qdot)
    for i in range(nb_q):
        rsh[i] = states[nb_q + i]
        rsh[i + nb_q] = qddot[i]
    return rsh


def dynamics_from_joint_torque(t_int, states, biorbd_model, u):
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    qddot = biorbd.Model.ForwardDynamics(biorbd_model, states[:nb_q], states[nb_q:], u).get_array()
    rsh = np.ndarray(nb_q + nb_qdot)
    for i in range(nb_q):
        rsh[i] = states[nb_q+i]
        rsh[i + nb_q] = qddot[i]

    return rsh


def dynamics_from_joint_torque_and_contact(t_int, states, biorbd_model, u):
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    cs = biorbd_model.getConstraints()
    qddot = biorbd.Model.ForwardDynamicsConstraintsDirect(biorbd_model, states[:nb_q], states[nb_q:], u, cs).get_array()
    rsh = np.ndarray(nb_q + nb_qdot)
    for i in range(nb_q):
        rsh[i] = states[nb_q+i]
        rsh[i + nb_q] = qddot[i]

    return rsh


def dynamics_from_accelerations(t_int, states, biorbd_model, u):
    nb_q = biorbd_model.nbQ()
    nb_qdot = biorbd_model.nbQdot()
    rsh = np.ndarray(nb_q + nb_qdot)
    for i in range(nb_q):
        rsh[i] = states[nb_q+i]
        rsh[i + nb_q] = u[i]

    return rsh


def runge_kutta_4(fun, t_span, y0, n_step):
    h = (t_span[1] - t_span[0]) / n_step  # Length of steps
    y = np.ndarray((y0.shape[0], n_step + 1))
    y[:, 0] = y0
    t = np.linspace(t_span[0], t_span[1], n_step + 1)

    for i in range(1, n_step + 1):
        k1 = fun(i*h, y[:, i-1])
        k2 = fun(i*h, y[:, i-1] + h/2 * k1)
        k3 = fun(i*h, y[:, i-1] + h/2 * k2)
        k4 = fun(i*h, y[:, i-1] + h * k3)
        y[:, i] = y[:, i-1] + h/6 * (k1 + 2*k2 + 2*k3 + k4)

    # Produce similar output as scipy integrator
    out_keys = {'success': True, 't': t, 'y': y}
    return SimpleNamespace(**out_keys)


def integrate_states_from_controls(biorbd_model, t, all_q, all_qdot, all_u, dyn_fun, verbose=False,
                                   use_previous_as_init=False, algo="rk45"):
    all_t = np.ndarray(0)
    integrated_state = np.ndarray((biorbd_model.nbQ() + biorbd_model.nbQdot(), 0))

    q_init = np.concatenate((all_q[:, 0], all_qdot[:, 0]))
    for interval in range(t.shape[0] - 1):  # integration between each point (but the last point)
        u = all_u[:, interval]

        if algo == "rk45":
            integrated_tp = integrate.solve_ivp(
                fun=lambda t, y: dyn_fun(t, y, biorbd_model, u),
                t_span=(t[interval], t[interval + 1]), y0=q_init, method='RK45', atol=1e-8, rtol=1e-6)
        elif algo == "rk4":
            integrated_tp = runge_kutta_4(fun=lambda t, y: dyn_fun(t, y, biorbd_model, u),
                                          t_span=(t[interval], t[interval + 1]), y0=q_init, n_step=10)
        else:
            raise IndentationError(f"{algo} is not implemented")

        q_init_previous = q_init
        if use_previous_as_init:
            q_init = integrated_tp.y[:, -1]
        else:
            q_init = np.concatenate((all_q[:, interval+1], all_qdot[:, interval+1]))

        if interval < t.shape[0] - 2:
            all_t = np.concatenate((all_t, integrated_tp.t[:-1]))
            integrated_state = np.concatenate((integrated_state, integrated_tp.y[:, :-1]), axis=1)
        else:
            all_t = np.concatenate((all_t, integrated_tp.t[:]))
            integrated_state = np.concatenate((integrated_state, integrated_tp.y[:, :]), axis=1)

        if verbose:
            print(f"Time: {t[interval]}")
            print(f"Initial states: {q_init_previous}")
            print(f"Control: {u}")
            print(f"Final states: {integrated_tp.y[:, -1]}")
            print(f"Expected final states: {np.concatenate((all_q[:, interval + 1], all_qdot[:, interval + 1]))}")
            print(f"Difference: {(integrated_tp.y[:, -1]-np.concatenate((all_q[:, interval + 1], all_qdot[:, interval + 1])))}")
            print("")

    return all_t, integrated_state


def interpolate_integration(nb_frames, t_int, y_int):
    nb_dof = y_int.shape[0]
    q_interp = np.ndarray((nb_frames, nb_dof))
    time_interp = np.linspace(0, t_int[-1], nb_frames)
    for q in range(nb_dof):
        tck = interpolate.splrep(t_int, y_int[q, :], s=0)
        q_interp[:, q] = interpolate.splev(time_interp, tck, der=0)
    return time_interp, q_interp


def plot_piecewise_constant(t, data):
    # Double the data
    new_t = np.repeat(t, 2, axis=0)
    if len(data.shape) == 1:
        new_data = np.repeat(data, 2, axis=0)
    else:
        new_data = np.repeat(data, 2, axis=1)

    # Realign
    new_t = new_t[1:]
    new_data = new_data[:-1]

    plt.plot(new_t, new_data)


def plot_piecewise_linear(t, data):
    plt.plot(t, data)


def derive(q, t):
    der = np.ndarray(q.shape)
    for i in range(q.shape[1]):
        for j in range(q.shape[0]-1):
            der[j][i] = (q[j+1][i]-q[j][i])/(t[j+1]-t[j])

    return der

# Get values from biorbd
def fatigue_dyn_biorbd(_model, _muscle, _q, _q_dot, fun_load, is_state, is_muscle_updated=True):
    _fatigue_model = biorbd.HillThelenTypeFatigable(_muscle)
    _fatigue_state = biorbd.FatigueDynamicStateXia(_fatigue_model.fatigueState())
    if is_state and type(fun_load) != biorbd.StateDynamics:
        print("Warning: command function is not of type StateDynamics")
        return 1
    if type(fun_load) == biorbd.StateDynamics:
        is_state = True

    def dyn(t, x):
        if not is_state:
            _load = fun_load(t)
            _emg = biorbd.StateDynamics(0, _load)
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