import biorbd
import numpy as np
from matplotlib import pyplot as plt
from optimal_control_python.utils import Bow, Violin

from bioptim import (
    OptimalControlProgram,
    Objective,
    ObjectiveList,
    DynamicsType,
    DynamicsTypeOption,
    Constraint,
    ConstraintList,
    BoundsOption,
    QAndQDotBounds,
    InitialGuessOption,
    Node,
    InterpolationType,
    Data,
)


def prepare_generic_ocp(biorbd_model_path, number_shooting_points, final_time, x_init, u_init, x0, acados, use_sx):
    biorbd_model = biorbd.Model(biorbd_model_path)
    n_tau = biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max, tau_init = -100, 100, 0
    violin = Violin("E")
    bow = Bow("frog")
    if acados:
        weight1 = 1000
        weight2 = 1

    else:
        weight1 = 100
        weight2 = 1
    constraints = ConstraintList()
    for j in range(1, number_shooting_points + 1):
        constraints.add(
            Constraint.ALIGN_MARKERS,
            node=j,
            min_bound=0,
            max_bound=0,
            first_marker_idx=Bow.contact_marker,
            second_marker_idx=violin.bridge_marker,
            list_index=j
        )
    # for j in range(1, 5):
    #     constraints.add(Constraint.ALIGN_MARKERS,
    #                     node=j,
    #                     min_bound=0,
    #                     max_bound=0,
    #                     first_marker_idx=Bow.contact_marker,
    #                     second_marker_idx=violin.bridge_marker, list_index=j)
    # for j in range(5, number_shooting_points + 1):
    #     constraints.add(Constraint.ALIGN_MARKERS,
    #                     node=j,
    #                     min_bound=-10**(j-14), #-10**(j-14) donne 25 itérations
    #                     max_bound=10**(j-14), # (j-4)/10 donne 21 itérations
    #                     first_marker_idx=Bow.contact_marker,
    #                     second_marker_idx=violin.bridge_marker, list_index=j)

    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, list_index=0)
    objective_functions.add(
        Objective.Lagrange.ALIGN_SEGMENT_WITH_CUSTOM_RT,
        weight=weight1,
        segment_idx=Bow.segment_idx,
        rt_idx=violin.rt_on_string,
        list_index=1
    )
    objective_functions.add(
        Objective.Lagrange.MINIMIZE_TORQUE, node=Node.ALL, index=bow.hair_idx,
        weight=weight2, list_index=2)  # permet de réduire le nombre d'itérations avant la convergence

    dynamics = DynamicsTypeOption(DynamicsType.TORQUE_DRIVEN)

    x_bounds = BoundsOption(QAndQDotBounds(biorbd_model))
    x_bounds[:, 0] = x0
    x_init = InitialGuessOption(x_init, interpolation=InterpolationType.EACH_FRAME)

    u_bounds = BoundsOption([[tau_min] * n_tau, [tau_max] * n_tau])
    u_init = InitialGuessOption(u_init, interpolation=InterpolationType.EACH_FRAME)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        use_SX=use_sx
    ), x_bounds


def warm_start_nmpc(sol, ocp, window_len, n_q, n_qdot, n_tau, biorbd_model, acados, shift=1):
    data_sol_prev = Data.get_data(ocp, sol, concatenate=False)
    q = data_sol_prev[0]["q"]
    dq = data_sol_prev[0]["q_dot"]
    u = data_sol_prev[1]["tau"]
    x = np.vstack([q, dq])
    x_out = x[:, 0]
    u_out = u[:, 0]
    x_init = np.vstack([q, dq])
    x_init[:, :-shift] = x[:, shift:]
    x_init[:, -shift:] = np.tile(np.array(x[:, -1])[:, np.newaxis], shift)  # constant
    x_bounds = BoundsOption(QAndQDotBounds(biorbd_model))
    x_bounds[:, 0] = x_init[:, 0]
    x_init = InitialGuessOption(x_init, interpolation=InterpolationType.EACH_FRAME)
    u_init = u[:, :-1]
    u_init[:, :-shift] = u[:, shift+1:]
    u_init[:, -shift:] = np.tile(np.array(u[:, -2])[:, np.newaxis], shift)
    u_init = InitialGuessOption(u_init, interpolation=InterpolationType.EACH_FRAME)

    ocp.update_initial_guess(x_init, u_init)
    ocp.update_bounds(x_bounds=x_bounds)
    if not acados:
        lam_g = np.ndarray(((((n_qdot + n_q) + 3) * window_len), 1))
        lam_g[:((n_q + n_qdot) * shift) * (window_len - 1)] = \
            sol['lam_g'][(shift*(n_q+n_qdot)):(n_qdot + n_q) * shift * window_len]
        # shift n_q + n_qdot * shift var
        lam_g[(n_q + n_qdot) * shift * (window_len - 1):(n_q + n_qdot) * shift * window_len] = \
            sol['lam_g'][(n_q + n_qdot) * shift * (window_len - 1):(n_q + n_qdot) * shift * window_len]
        # last 20 var are copied
        lam_g[(n_q + n_qdot) * shift * window_len:-3 * shift] = sol['lam_g'][((n_q + n_qdot) * window_len + 3) * shift:]
        # shift 3 etats (1 constraint ALIGN MARKERS)
        lam_g[-3*shift:] = sol['lam_g'][-3*shift:]
        # copied 3 last
        lam_x = np.ndarray(((n_qdot+n_q) * (window_len + 1) + (n_tau * window_len), 1))

        # shift 30 var, n_tau + n_q + n_dot
        lam_x[:-((n_tau + n_q + n_qdot) * shift)] = sol['lam_x'][((n_tau + n_q + n_qdot) * shift):]
        lam_x[-((n_tau + n_q + n_qdot)*shift):] = sol['lam_x'][-((n_tau + n_q + n_qdot) * shift):]

        return x_init, u_init, x_out, u_out, x_bounds, u, lam_g, lam_x

    else:
        return x_init, u_init, x_out, u_out, x_bounds, u


def warm_start_nmpc_same_iter(sol, ocp, biorbd_model):
    data_sol_prev = Data.get_data(ocp, sol, concatenate=False)
    q = data_sol_prev[0]["q"]
    dq = data_sol_prev[0]["q_dot"]
    u = data_sol_prev[1]["tau"]
    x = np.vstack([q, dq])
    x_out = x[:, 0]
    u_out = u[:, 0]
    lam_g = sol['lam_g']
    lam_x = sol['lam_x']
    x_init = np.vstack([q, dq])
    x_init[:, :] = x[:, :]
    # x_init[:, -shift:] = np.tile(np.array(x[:, -1])[:, np.newaxis], shift) # constant
    x_bounds = BoundsOption(QAndQDotBounds(biorbd_model))
    x_bounds[:, 0] = x_init[:, 0]
    x_init = InitialGuessOption(x_init, interpolation=InterpolationType.EACH_FRAME)
    u_init = u[:, :-1]
    u_init = InitialGuessOption(u_init, interpolation=InterpolationType.EACH_FRAME)

    ocp.update_initial_guess(x_init, u_init)
    ocp.update_bounds(x_bounds=x_bounds)

    return x_init, u_init, x_out, u_out, x_bounds, u, lam_g, lam_x


def define_new_objectives(weight, ocp, q_target, bow):
    new_objectives = ObjectiveList()
    new_objectives.add(
        Objective.Lagrange.TRACK_STATE, node=Node.ALL, weight=weight, target=q_target[bow.hair_idx:bow.hair_idx+1, :],
        index=bow.hair_idx,
        list_index=3
    )
    ocp.update_objectives(new_objectives)


def display_graphics_x_est(target, x_est):
    plt.suptitle('X_est')
    for dof in range(10):
        plt.subplot(2, 5, int(dof + 1))
        if dof == 9:
            plt.plot(target[:x_est.shape[1]], color="red")
        plt.plot(x_est[dof, :], color="blue")
        plt.title(f"dof {dof}")
        plt.show()


def display_x_est(target, x_est, bow):
    plt.suptitle('X_est and target')
    plt.plot(target[:x_est.shape[1]], color="red")
    plt.title(f"target")
    plt.plot(x_est[bow.hair_idx, :], color="blue")
    plt.title(f"dof {bow.hair_idx}")
    plt.show()


def compare_target(target, target_curve):
    plt.suptitle('target_curve et target modulo')
    plt.subplot(2, 1, 1)
    plt.plot(target, color="blue")
    plt.title(f"target")
    plt.subplot(2, 1, 2)
    plt.plot(target_curve, color="red")
    plt.title(f"target_curve")
    plt.show()
