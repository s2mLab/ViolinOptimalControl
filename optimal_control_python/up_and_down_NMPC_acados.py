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
    Solver,
    Simulate,
)

def prepare_generic_ocp(biorbd_model_path, number_shooting_points, final_time, x_init, u_init, x0, useSX=True):
    biorbd_model = biorbd.Model(biorbd_model_path)
    n_tau = biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max, tau_init = -100, 100, 0

    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, list_index=0)
    objective_functions.add(
        Objective.Lagrange.ALIGN_SEGMENT_WITH_CUSTOM_RT,
        weight=10,
        segment_idx=Bow.segment_idx,
        rt_idx=violin.rt_on_string,
        list_index=1
    )
    objective_functions.add(
        Objective.Lagrange.MINIMIZE_TORQUE, node=Node.ALL, index=bow.hair_idx,
        weight=0.1, list_index=2)  # permet de réduire le nombre d'itérations avant la convergence


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
                            min_bound=-1, #-10**(j-14) donne 25 itérations
                            max_bound=1, # (j-4)/10 donne 21 itérations
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
        use_SX=useSX
    ), x_bounds



def warm_start_nmpc(sol, shift=1):
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

    return x_init, u_init, X_out, U_out, x_bounds, u,



def define_new_objectives():
    new_objectives = ObjectiveList()

    new_objectives.add(
        Objective.Lagrange.TRACK_STATE, node=Node.ALL, weight=10, target=q_target[bow.hair_idx:bow.hair_idx+1, :],
        index=bow.hair_idx,
        list_index=3
    )
    # new_objectives.add(Objective.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, node=Node.ALL, states_idx=bow.hair_idx,
    #                    # weight=1,
    #                    idx=4)  # rajoute des itérations et ne semble riuen changer au mouvement...
    ocp.update_objectives(new_objectives)

def display_graphics_X_est():
    matplotlib.pyplot.suptitle('X_est')
    for dof in range(10):
        matplotlib.pyplot.subplot(2, 5, int(dof + 1))
        if dof == 9:
            matplotlib.pyplot.plot(target[:X_est_acados.shape[1]], color="red")
        matplotlib.pyplot.plot(X_est_acados[dof, :], color="blue")
        matplotlib.pyplot.title(f"dof {dof}")
        matplotlib.pyplot.show()

def display_X_est():
    matplotlib.pyplot.suptitle('X_est and target')
    matplotlib.pyplot.plot(target[:X_est_acados.shape[1]], color="red")
    matplotlib.pyplot.title(f"target")
    matplotlib.pyplot.plot(X_est_acados[9, :], color="blue")
    matplotlib.pyplot.title(f"dof {9}")
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
    biorbd_model_path = "../models/BrasViolon.bioMod"
    biorbd_model = biorbd.Model(biorbd_model_path)
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()
    n_muscles = biorbd_model.nbMuscles()
    window_time = 1/8  # duration of the window
    nb_shooting_pts_window = 100  # size of NMPC window
    ns_tot_up_and_down = 150 # size of the up_and_down gesture

    violin = Violin("E")
    bow = Bow("frog")

    # np.save("bow_target_param", generate_up_and_down_bow_target(200))
    bow_target_param = np.load("bow_target_param.npy")
    frame_to_init_from = 35
    nb_shooting_pts_all_optim = 300

    X_est_acados = np.zeros((n_qdot + n_q , nb_shooting_pts_all_optim))
    U_est_acados = np.zeros((n_tau, nb_shooting_pts_all_optim))
    begin_at_first_iter = True
    if begin_at_first_iter == True :
        # Initial guess and bounds
        x0 = np.array(violin.initial_position()[bow.side] + [0] * n_qdot)

        x_init = np.tile(np.array(violin.initial_position()[bow.side] + [0] * n_qdot)[:, np.newaxis],
                         nb_shooting_pts_window+1)
        u_init = np.tile(np.array([0.5] * n_tau)[:, np.newaxis],
                         nb_shooting_pts_window)
    else:
        X_est_init = np.load('X_est_acados.npy')[:, :frame_to_init_from+1]
        U_est_init = np.load('U_est_acados.npy')[:, :frame_to_init_from+1]
        x_init = X_est_init[:, -(nb_shooting_pts_window+1):]
        x0 = x_init[:, 0]
        u_init = U_est_init[:, -nb_shooting_pts_window:]




    # position initiale de l'ocp
    ocp, x_bounds = prepare_generic_ocp(
        biorbd_model_path=biorbd_model_path,
        number_shooting_points=nb_shooting_pts_window,
        final_time=window_time,
        x_init=x_init,
        u_init=u_init,
        x0=x0,
        useSX=True,
    )



    t = np.linspace(0, 2, ns_tot_up_and_down)
    target_curve = curve_integral(bow_target_param, t)
    q_target = np.ndarray((n_q, nb_shooting_pts_window + 1))
    Nmax = nb_shooting_pts_all_optim + 50
    target = np.ndarray(Nmax)
    T = np.ndarray((Nmax))
    for i in range(Nmax):
        a=i % ns_tot_up_and_down
        T[i]=t[a]
    target = curve_integral(bow_target_param, T)

    shift = 1


    # Init from known position
    # ocp_load, sol_load = OptimalControlProgram.load(f"saved_iterations/{frame_to_init_from-1}_iter_acados.bo")
    # data_sol_prev = Data.get_data(ocp_load, sol_load, concatenate=False)
    # x_init, u_init, X_out, U_out, x_bounds, u,  = warm_start_nmpc(sol=sol_load, shift=shift)
    # U_est_acados[:, :U_est_init.shape[1]] = U_est_init
    # X_est_acados[:, :X_est_init.shape[1]] = X_est_init



    # for i in range(frame_to_init_from, 200):
    for i in range(0, 200):
        q_target[bow.hair_idx, :] = target[i * shift: nb_shooting_pts_window + (i * shift) + 1]
        define_new_objectives()
        if i==0:
            sol = ocp.solve(
                show_online_optim=False,
                solver=Solver.ACADOS,
                solver_options={"nlp_solver_max_iter": 10},
            )
        else:
            sol = ocp.solve(
                show_online_optim=False,
                solver=Solver.ACADOS,
            )
        x_init, u_init, X_out, U_out, x_bounds, u = warm_start_nmpc(sol=sol, shift=shift)
        X_est_acados[:, i] = X_out
        U_est_acados[:, i] = U_out

        ocp.save(sol, f"saved_iterations/{i}_iter_acados")  # you don't have to specify the extension ".bo"

    np.save("X_est_acados", X_est_acados)
    np.save("U_est_acados", U_est_acados)
    np.load(U_est_acados)



## Pour afficher mouvement global

ocp, x_bounds = prepare_generic_ocp(
    biorbd_model_path=biorbd_model_path,
    number_shooting_points=frame_to_init_from,
    final_time=2,
    x_init=X_est_acados,
    u_init=U_est_acados,
    x0=x0,
    )
sol = Simulate.from_controls_and_initial_states(ocp, x_init.initial_guess, u_init.initial_guess)
ShowResult(ocp, sol).graphs()