import biorbd
import numpy as np
import matplotlib
from optimal_control_python.generate_bow_trajectory import generate_bow_trajectory, curve_integral
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
    ShowResult,
    Simulate,
)

def prepare_generic_ocp(biorbd_model_path, number_shooting_points, final_time, x_init, u_init, x0, acados, useSX):
    biorbd_model = biorbd.Model(biorbd_model_path)
    n_tau = biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max, tau_init = -100, 100, 0
    violin = Violin("E")
    bow = Bow("frog")
    if acados == True:
        weight1=1000
        weight2=1

    else:
        weight1 = 100
        weight2 = 1
    constraints = ConstraintList()
    for j in range(1, number_shooting_points + 1):
        constraints.add(Constraint.ALIGN_MARKERS,
                            node=j,
                            min_bound=0,
                            max_bound=0,
                            first_marker_idx=Bow.contact_marker,
                            second_marker_idx=violin.bridge_marker, list_index=j)
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
        use_SX=useSX
    ), x_bounds


def warm_start_nmpc(sol, ocp, nb_shooting_pts_window, n_q, n_qdot, n_tau, biorbd_model, acados, shift=1):
    data_sol_prev = Data.get_data(ocp, sol, concatenate=False)
    q = data_sol_prev[0]["q"]
    dq = data_sol_prev[0]["q_dot"]
    u = data_sol_prev[1]["tau"]
    x = np.vstack([q, dq])
    X_out = x[:, 0]
    U_out = u[:, 0]
    x_init = np.vstack([q, dq])
    x_init[:, :-shift] = x[:, shift:]
    x_init[:, -shift:] = np.tile(np.array(x[:, -1])[:, np.newaxis], shift) # constant
    x_bounds = BoundsOption(QAndQDotBounds(biorbd_model))
    x_bounds[:, 0] = x_init[:, 0]
    x_init=InitialGuessOption(x_init, interpolation=InterpolationType.EACH_FRAME)
    u_init = u[:, :-1]
    u_init[:, :-shift] = u[:, shift+1:]
    u_init[:, -shift:]= np.tile(np.array(u[:, -2])[:, np.newaxis], shift)
    u_init=InitialGuessOption(u_init, interpolation=InterpolationType.EACH_FRAME)

    ocp.update_initial_guess(x_init, u_init)
    ocp.update_bounds(x_bounds=x_bounds)
    if acados==False:


        lam_g = np.ndarray(((((n_qdot + n_q) + 3) * nb_shooting_pts_window), 1))
        lam_g[:((n_q + n_qdot) * shift) * (nb_shooting_pts_window - 1)] = sol['lam_g'][(shift*(n_q+n_qdot)):(n_qdot + n_q) * shift *nb_shooting_pts_window]
        # shift n_q + n_qdot * shift var
        lam_g[(n_q + n_qdot) * shift * (nb_shooting_pts_window - 1):(n_q + n_qdot) * shift * nb_shooting_pts_window] = sol['lam_g'][(n_q + n_qdot) * shift * (nb_shooting_pts_window - 1):(n_q + n_qdot) * shift * nb_shooting_pts_window]
        # last 20 var are copied
        lam_g[(n_q + n_qdot) * shift* nb_shooting_pts_window:-3*shift] = sol['lam_g'][((n_q + n_qdot) * nb_shooting_pts_window+ 3)*shift:]
        # shift 3 etats (1 constraint ALIGN MARKERS)
        lam_g[-3*shift:] = sol['lam_g'][-3*shift:]
        # copied 3 last
        lam_x = np.ndarray(((n_qdot+n_q)*(nb_shooting_pts_window+1)+(n_tau*nb_shooting_pts_window), 1))

        lam_x[:-((n_tau + n_q + n_qdot) * shift)] = sol['lam_x'][((n_tau + n_q + n_qdot) * shift):] # shift 30 var, n_tau + n_q + n_dot
        lam_x[-((n_tau + n_q + n_qdot)*shift):] = sol['lam_x'][-((n_tau + n_q + n_qdot) * shift):]


        return x_init, u_init, X_out, U_out, x_bounds, u, lam_g, lam_x
    else:
        return x_init, u_init, X_out, U_out, x_bounds, u



def warm_start_nmpc_same_iter(sol, ocp, biorbd_model):
    data_sol_prev = Data.get_data(ocp, sol, concatenate=False)
    q = data_sol_prev[0]["q"]
    dq = data_sol_prev[0]["q_dot"]
    u = data_sol_prev[1]["tau"]
    x = np.vstack([q, dq])
    X_out = x[:, 0]
    U_out = u[:, 0]
    lam_g = sol['lam_g']
    lam_x = sol['lam_x']
    x_init = np.vstack([q, dq])
    x_init[:, :] = x[:, :]
    # x_init[:, -shift:] = np.tile(np.array(x[:, -1])[:, np.newaxis], shift) # constant
    x_bounds = BoundsOption(QAndQDotBounds(biorbd_model))
    x_bounds[:, 0] = x_init[:, 0]
    x_init=InitialGuessOption(x_init, interpolation=InterpolationType.EACH_FRAME)
    u_init = u[:, :-1]
    u_init=InitialGuessOption(u_init, interpolation=InterpolationType.EACH_FRAME)

    ocp.update_initial_guess(x_init, u_init)
    ocp.update_bounds(x_bounds=x_bounds)

    return x_init, u_init, X_out, U_out, x_bounds, u, lam_g, lam_x

def define_new_objectives(weight, ocp, q_target, bow):
    new_objectives = ObjectiveList()
    #for i in range(nb_shooting_pts_window):
    new_objectives.add(
        Objective.Lagrange.TRACK_STATE, node=Node.ALL, weight=weight, target=q_target[bow.hair_idx:bow.hair_idx+1, :],
        index=bow.hair_idx,
        list_index=3
    )
    ocp.update_objectives(new_objectives)

def display_graphics_X_est(target, X_est):
    matplotlib.pyplot.suptitle('X_est')
    for dof in range(10):
        matplotlib.pyplot.subplot(2, 5, int(dof + 1))
        if dof == 9:
            matplotlib.pyplot.plot(target[:X_est.shape[1]], color="red")
        matplotlib.pyplot.plot(X_est[dof, :], color="blue")
        matplotlib.pyplot.title(f"dof {dof}")
        matplotlib.pyplot.show()

def display_X_est(target, X_est, bow):
    matplotlib.pyplot.suptitle('X_est and target')
    matplotlib.pyplot.plot(target[:X_est.shape[1]], color="red")
    matplotlib.pyplot.title(f"target")
    matplotlib.pyplot.plot(X_est[bow.hair_idx, :], color="blue")
    matplotlib.pyplot.title(f"dof {bow.hair_idx}")
    matplotlib.pyplot.show()

def compare_target(target, target_curve):
    matplotlib.pyplot.suptitle('target_curve et target modulo')
    matplotlib.pyplot.subplot(2, 1, 1)
    matplotlib.pyplot.plot(target, color="blue")
    matplotlib.pyplot.title(f"target")
    matplotlib.pyplot.subplot(2, 1, 2)
    matplotlib.pyplot.plot(target_curve, color="red")
    matplotlib.pyplot.title(f"target_curve")
    matplotlib.pyplot.show()