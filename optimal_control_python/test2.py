import biorbd
import numpy as np
import matplotlib
from optimal_control_python.generate_up_and_down_bow_target import generate_up_and_down_bow_target
from optimal_control_python.generate_up_and_down_bow_target import curve_integral
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

def prepare_generic_ocp(biorbd_model_path, number_shooting_points, final_time, x_init, u_init, x0, ):
    biorbd_model = biorbd.Model(biorbd_model_path)
    n_tau = biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max, tau_init = -100, 100, 0

    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, list_index=0)
    objective_functions.add(
        Objective.Lagrange.ALIGN_SEGMENT_WITH_CUSTOM_RT,
        weight=100,
        segment_idx=Bow.segment_idx,
        rt_idx=violin.rt_on_string,
        list_index=1
    )
    objective_functions.add(
        Objective.Lagrange.MINIMIZE_TORQUE, node=Node.ALL, index=bow.hair_idx,
        weight=1, list_index=2)  # permet de réduire le nombre d'itérations avant la convergence


    dynamics = DynamicsTypeOption(DynamicsType.TORQUE_DRIVEN)

    x_bounds = BoundsOption(QAndQDotBounds(biorbd_model))
    x_bounds[:, 0] = x0
    x_init = InitialGuessOption(x_init, interpolation=InterpolationType.EACH_FRAME)

    u_bounds = BoundsOption([[tau_min] * n_tau, [tau_max] * n_tau])
    u_init = InitialGuessOption(u_init, interpolation=InterpolationType.EACH_FRAME)

    constraints = ConstraintList()
    for j in range(1, 5):
        constraints.add(Constraint.ALIGN_MARKERS,
                            node=j,
                            min_bound=0,
                            max_bound=0,
                            first_marker_idx=Bow.contact_marker,
                            second_marker_idx=violin.bridge_marker, list_index=j)
    for j in range(5, nb_shooting_pts_window + 1):
        constraints.add(Constraint.ALIGN_MARKERS,
                            node=j,
                            # min_bound=-1, #-10**(j-14) donne 25 itérations
                            # max_bound=1, # (j-4)/10 donne 21 itérations
                            first_marker_idx=Bow.contact_marker,
                            second_marker_idx=violin.bridge_marker, list_index=j)

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
    ), x_bounds



def warm_start_nmpc(sol, shift=1):
    data_sol_prev = Data.get_data(ocp, sol, concatenate=False)
    q = data_sol_prev[0]["q"]
    dq = data_sol_prev[0]["q_dot"]
    u = data_sol_prev[1]["tau"]
    x = np.vstack([q, dq])
    X_out = x[:, 0]
    U_out = u[:, 0]
    lam_g = np.ndarray(((((n_qdot+n_q)+3)*nb_shooting_pts_window), 1)) # 20 états
    lam_g[:-(3*nb_shooting_pts_window)-((n_q+n_qdot)*shift)] = sol['lam_g'][(n_q+n_qdot)*shift+(3*nb_shooting_pts_window):] # shift 20 var, n_q + n_qdot
    lam_g[-(n_q+n_qdot)*shift:] = sol['lam_g'][-(n_q+n_qdot)*shift:] # last 20 var are copied
    lam_g[:-3*shift] = sol['lam_g'][3*shift:] # shift 3 etats
    lam_g[-3*shift:] = sol['lam_g'][-3*shift:] #copied 3 last
    lam_x = np.ndarray(((n_qdot+n_q)*(nb_shooting_pts_window+1)+(n_tau*nb_shooting_pts_window), 1))
    lam_x[:-(3*n_tau*shift)] = sol['lam_x'][(3*n_tau*shift):] # shift 30 var, n_q + n_qdot
    lam_x[-(3*n_tau*shift):] = sol['lam_x'][-(3*n_tau*shift):] # last 30 var are copied
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

    return x_init, u_init, X_out, U_out, x_bounds, u, lam_g, lam_x


def warm_start_nmpc_same_iter(sol):
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
    # u_init[:, :] = u[:, :]
    # u_init[:, -shift:]= np.tile(np.array(u[:, -2])[:, np.newaxis], shift)
    u_init=InitialGuessOption(u_init, interpolation=InterpolationType.EACH_FRAME)

    ocp.update_initial_guess(x_init, u_init)
    ocp.update_bounds(x_bounds=x_bounds)

    return x_init, u_init, X_out, U_out, x_bounds, u, lam_g, lam_x


def define_new_objectives():
    new_objectives = ObjectiveList()

    new_objectives.add(
        Objective.Lagrange.TRACK_STATE, node=Node.ALL, weight=10000, target=q_target[bow.hair_idx:bow.hair_idx+1 , :],
        index=bow.hair_idx,
        list_index=3
    )
    # new_objectives.add(Objective.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, node=Node.ALL, index=bow.hair_idx,
    #                    # weight=1,
    #                    list_index=4)  # rajoute des itérations et ne semble riuen changer au mouvement...
    ocp.update_objectives(new_objectives)

def display_graphics_X_est():
    matplotlib.pyplot.suptitle('X_est')
    for dof in range(10):
        matplotlib.pyplot.subplot(2, 5, int(dof + 1))
        if dof == 9:
            matplotlib.pyplot.plot(target[:X_est.shape[1]], color="red")
        matplotlib.pyplot.plot(X_est[dof, :], color="blue")
        matplotlib.pyplot.title(f"dof {dof}")
        matplotlib.pyplot.show()

def display_X_est():
    matplotlib.pyplot.suptitle('X_est and target')
    matplotlib.pyplot.plot(target[:X_est.shape[1]], color="red")
    matplotlib.pyplot.title(f"target")
    matplotlib.pyplot.plot(X_est[bow.hair_idx, :], color="blue")
    matplotlib.pyplot.title(f"dof {bow.hair_idx}")
    matplotlib.pyplot.show()

def compare_target():
    matplotlib.pyplot.suptitle('target_curve et target modulo')
    matplotlib.pyplot.subplot(2, 1, 1)
    matplotlib.pyplot.plot(target, color="blue")
    matplotlib.pyplot.title(f"target")
    matplotlib.pyplot.subplot(2, 1, 2)
    matplotlib.pyplot.plot(target_curve, color="red")
    matplotlib.pyplot.title(f"target_curve")
    matplotlib.pyplot.show()


if __name__ == "__main__":
    # Parameters
    biorbd_model_path = "/home/carla/Documents/Programmation/ViolinOptimalControl/models/BrasViolon.bioMod"
    biorbd_model = biorbd.Model(biorbd_model_path)
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()
    n_muscles = biorbd_model.nbMuscles()
    final_time = 1/3  # duration of the
    nb_shooting_pts_window = 15  # size of NMPC window
    ns_tot_up_and_down = nb_shooting_pts_window* 10 # size of the up_and_down gesture

    violin = Violin("E")
    bow = Bow("frog")

    # np.save("bow_target_param", generate_up_and_down_bow_target(200))
    bow_target_param = np.load("bow_target_param.npy")
    frame_to_init_from = 290
    nb_shooting_pts_all_optim = 300

    X_est = np.zeros((n_qdot + n_q , nb_shooting_pts_all_optim))
    U_est = np.zeros((n_tau, nb_shooting_pts_all_optim))
    begin_at_first_iter = True
    if begin_at_first_iter == True :
        # Initial guess and bounds
        x0 = np.array(violin.initial_position()[bow.side] + [0] * n_qdot)

        x_init = np.tile(np.array(violin.initial_position()[bow.side] + [0] * n_qdot)[:, np.newaxis],
                         nb_shooting_pts_window+1)
        u_init = np.tile(np.array([0.5] * n_tau)[:, np.newaxis],
                         nb_shooting_pts_window)
    else:
        X_est_init = np.load('X_est.npy')[:, :frame_to_init_from+1]
        # X_est_init = np.delete(X_est_init, np.s_[75:], axis=1)
        U_est_init = np.load('U_est.npy')[:, :frame_to_init_from+1]
        # U_est_init = np.delete(U_est_init, np.s_[75:], axis=1)
        # x0 = X_est_init[:, -1]
        x_init = X_est_init[:, -(nb_shooting_pts_window+1):]
        x0 = x_init[:, 0]
        u_init = U_est_init[:, -nb_shooting_pts_window:]




    # position initiale de l'ocp
    ocp, x_bounds = prepare_generic_ocp(
        biorbd_model_path=biorbd_model_path,
        number_shooting_points=nb_shooting_pts_window,
        final_time=final_time,
        x_init=x_init,
        u_init=u_init,
        x0=x0,
    )



    t = np.linspace(0, 2, ns_tot_up_and_down)
    target_curve = curve_integral(bow_target_param, t)
    q_target = np.ndarray((n_q, nb_shooting_pts_window + 1))
    # q_target=np.ndarray(nb_shooting_pts_all_optim+1)
    # q_target=np.zeros(nb_shooting_pts_all_optim+1)
    Nmax = nb_shooting_pts_all_optim+50
    target = np.ndarray(Nmax)
    T = np.ndarray((Nmax))
    for i in range(Nmax):
        a=i % ns_tot_up_and_down
        T[i]=t[a]
    target = curve_integral(bow_target_param, T)


    shift = 1


    # Init from known position
    # ocp_load, sol_load = OptimalControlProgram.load(f"saved_iterations/{frame_to_init_from-1}_iter.bo")
    # data_sol_prev = Data.get_data(ocp_load, sol_load, concatenate=False)
    # x_init, u_init, X_out, U_out, x_bounds, u, lam_g, lam_x = warm_start_nmpc(sol=sol_load) #
    # U_est[:, :U_est_init.shape[1]] = U_est_init
    # X_est[:, :X_est_init.shape[1]] = X_est_init


    LAM_G_evolution = np.ndarray((345, frame_to_init_from))
    LAM_X_evolution = np.ndarray((470, frame_to_init_from))

    for i in range(0, frame_to_init_from):
        q_target[bow.hair_idx, :] = target[0 * shift: nb_shooting_pts_window + (0 * shift) + 1]
        define_new_objectives()
        sol = ocp.solve(
            show_online_optim=False,
            solver_options={"max_iter": 1000, "hessian_approximation": "exact", "bound_push": 10 ** (-10),
                            "bound_frac": 10 ** (-10), "warm_start_init_point":"yes",
                            "warm_start_bound_push" : 10 ** (-16), "warm_start_bound_frac" : 10 ** (-16),
                            "nlp_scaling_method": "none", "warm_start_mult_bound_push": 10 ** (-16),
                            # "warm_start_slack_bound_push": 10 ** (-16)," warm_start_slack_bound_frac":10 ** (-16),
                            }
        )
        # sol = Simulate.from_controls_and_initial_states(ocp, x_init.initial_guess, u_init.initial_guess)
        x_init, u_init, X_out, U_out, x_bounds, u, lam_g, lam_x = warm_start_nmpc_same_iter(sol=sol)
        sol['lam_g'] = lam_g
        sol['lam_x'] = lam_x
        X_est[:, i] = X_out
        U_est[:, i] = U_out
        # np.load(U_est)

    np.save("X_est", X_est)
    np.save("U_est", U_est)
    np.load(U_est)


    matplotlib.pyplot.suptitle('Lam_G')
    for dof in range(30):
        matplotlib.pyplot.subplot(3, 10, int(dof + 1))
        matplotlib.pyplot.plot(LAM_G_evolution[dof, :], color="blue")
        matplotlib.pyplot.title(f"ligne n° {dof} in 345")
        matplotlib.pyplot.show()

## Vérifier que Lam X est pareil ou non en chaque noeud de shooting sur un graphe superposé

    for dof in range(470):
        matplotlib.pyplot.suptitle('Lam X')
        matplotlib.pyplot.plot(LAM_X_evolution[dof, :], color="red")
        # matplotlib.pyplot.plot(LAM_G_evolution[dof*2, :], color="blue")
        matplotlib.pyplot.show()

    for dof in range(345):
        matplotlib.pyplot.suptitle('Lam G')
        matplotlib.pyplot.plot(LAM_G_evolution[dof, :], color="red")
        # matplotlib.pyplot.plot(LAM_G_evolution[dof*2, :], color="blue")
        matplotlib.pyplot.show()
