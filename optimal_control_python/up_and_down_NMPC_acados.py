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
    Instant,
    InterpolationType,
    Data,
    ShowResult,
    Simulate,
    Solver
)

def prepare_generic_ocp(biorbd_model_path, number_shooting_points, final_time, x_init, u_init, x0, useSX=True):
    biorbd_model = biorbd.Model(biorbd_model_path)
    n_tau = biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max, tau_init = -100, 100, 0

    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, idx=0)
    objective_functions.add(
        Objective.Lagrange.ALIGN_SEGMENT_WITH_CUSTOM_RT,
        weight=100,
        segment_idx=Bow.segment_idx,
        rt_idx=violin.rt_on_string,
        idx=1
    )
    objective_functions.add(
        Objective.Lagrange.MINIMIZE_TORQUE, instant=Instant.ALL, state_idx=bow.hair_idx,
        weight=1, idx=2)  # permet de réduire le nombre d'itérations avant la convergence


    dynamics = DynamicsTypeOption(DynamicsType.TORQUE_DRIVEN)

    x_bounds = BoundsOption(QAndQDotBounds(biorbd_model))
    x_bounds[:, 0] = x0
    x_init = InitialGuessOption(x_init, interpolation=InterpolationType.EACH_FRAME)

    u_bounds = BoundsOption([[tau_min] * n_tau, [tau_max] * n_tau])
    u_init = InitialGuessOption(u_init, interpolation=InterpolationType.EACH_FRAME)

    constraints = ConstraintList()
    for j in range(1, 5):
        constraints.add(Constraint.ALIGN_MARKERS,
                            instant=j,
                            min_bound=0,
                            max_bound=0,
                            first_marker_idx=Bow.contact_marker,
                            second_marker_idx=violin.bridge_marker, idx=j)
    for j in range(5, nb_shooting_pts_window + 1):
        constraints.add(Constraint.ALIGN_MARKERS,
                            instant=j,
                            # min_bound=-1, #-10**(j-14) donne 25 itérations
                            # max_bound=1, # (j-4)/10 donne 21 itérations
                            first_marker_idx=Bow.contact_marker,
                            second_marker_idx=violin.bridge_marker, idx=j)

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
        Objective.Lagrange.TRACK_STATE, instant=Instant.ALL, weight=10000, target=q_target[bow.hair_idx:bow.hair_idx+1 , :],
        states_idx=bow.hair_idx,
        idx=3
    )
    # new_objectives.add(Objective.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, instant=Instant.ALL, state_idx=bow.hair_idx,
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
    frame_to_init_from = 280
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
        # X_est_init = np.delete(X_est_init, np.s_[75:], axis=1)
        U_est_init = np.load('U_est_acados.npy')[:, :frame_to_init_from+1]
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
        useSX=True,
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
    # ocp_load, sol_load = OptimalControlProgram.load(f"saved_iterations/{frame_to_init_from-1}_iter_acados.bo")
    # data_sol_prev = Data.get_data(ocp_load, sol_load, concatenate=False)
    # x_init, u_init, X_out, U_out, x_bounds, u,  = warm_start_nmpc(sol=sol_load, shift=shift)
    # U_est_acados[:, :U_est_init.shape[1]] = U_est_init
    # X_est_acados[:, :X_est_init.shape[1]] = X_est_init


    # for i in range(frame_to_init_from, nb_shooting_pts_all_optim):
    for i in range(1, frame_to_init_from):
        q_target[bow.hair_idx, :] = target[i * shift: nb_shooting_pts_window + (i * shift) + 1]
        # q_target[bow.hair_idx, i] = target[i * shift]
        # q_target[i] = target[i * shift]
        define_new_objectives()

        sol = ocp.solve(
            show_online_optim=False,
            solver=Solver.ACADOS
        )
        # sol = Simulate.from_controls_and_initial_states(ocp, x_init.initial_guess, u_init.initial_guess)
        x_init, u_init, X_out, U_out, x_bounds, u = warm_start_nmpc(sol=sol, shift=shift)
        # sol['lam_g']
        # sol['lam_x']
        X_est_acados[:, i] = X_out
        U_est_acados[:, i] = U_out

        ocp.save(sol, f"saved_iterations/{i}_iter_acados")  # you don't have to specify the extension ".bo"
        # np.save("X_est", X_est)

    np.save("X_est_acados", X_est_acados)
    np.save("U_est_acados", U_est_acados)
    np.load(U_est_acados)



## Pour afficher mouvement global

ocp, x_bounds = prepare_generic_ocp(
    biorbd_model_path=biorbd_model_path,
    number_shooting_points=nb_shooting_pts_all_optim,
    final_time=2,
    x_init=X_est,
    u_init=U_est,
    x0=x0,
    )
sol = ocp.solve(
    show_online_optim=False,
    solver_options={"max_iter": 0, "hessian_approximation": "exact", "bound_push": 10 ** (-10),
                    "bound_frac": 10 ** (-10), "print_options_documentation": "yes"}  # , "bound_push": 10**(-10), "bound_frac": 10**(-10)}
    )
ShowResult(ocp, sol).graphs()