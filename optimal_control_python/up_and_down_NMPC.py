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
)

def prepare_generic_ocp(biorbd_model_path, number_shooting_points, final_time, x_init, u_init, x0, ):
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
    # ocp.update_constraints(new_constraints)

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
    x_init = np.vstack([q, dq])

    # x_init = np.zeros(x.shape)
    x_init[:, :-shift] = x[:, shift:]
    x_init[:, -shift:] = np.tile(np.array(x[:, -1])[:, np.newaxis], shift) # constant
    x_bounds = BoundsOption(QAndQDotBounds(biorbd_model))
    x_bounds[:, 0] = x_init[:, 0]
    # x_bounds_prev = BoundsOption(QAndQDotBounds(biorbd_model))
    # x_bounds_prev[:, 0] = x0
    x_init=InitialGuessOption(x_init, interpolation=InterpolationType.EACH_FRAME)
    u_init = u[:, :-1]
    u_init[:, :-shift] = u[:, shift+1:]  # [:, -1:]  # discard oldest estimate of the window
    u_init[:, -shift:]= np.tile(np.array(u[:, -2])[:, np.newaxis], shift)
    u_init=InitialGuessOption(u_init, interpolation=InterpolationType.EACH_FRAME)

    ocp.update_initial_guess(x_init, u_init)

    # x_bounds = BoundsOption(QAndQDotBounds(biorbd_model))
    # x_bounds[:, 0] = x0 # _init[:, 0]

    ocp.update_bounds(x_bounds=x_bounds)

    return x_init, u_init, X_out, U_out, x_bounds, u # , x_bounds_prev



def define_new_objectives():
    new_objectives = ObjectiveList()

    new_objectives.add(
        Objective.Lagrange.TRACK_STATE, instant=Instant.ALL, weight=1000, target=q_target, states_idx=bow.hair_idx,
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
        matplotlib.pyplot.plot(X_est[dof, :], color="blue")
        matplotlib.pyplot.title(f"dof {dof}")
        matplotlib.pyplot.show()

def display_X_est_():
    X_est_ = np.load('X_est_.npy')
    matplotlib.pyplot.suptitle('X_est')
    matplotlib.pyplot.plot(X_est_[9, :], color="blue")
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

def display_graphics():
    matplotlib.pyplot.figure(1)

    data_sol = Data.get_data(ocp, sol, concatenate=False)

    matplotlib.pyplot.suptitle('Q et x_init')
    for dof in range(10):  # idx = degré de liberté
        matplotlib.pyplot.subplot(2, 5, int(dof + 1))

        matplotlib.pyplot.plot(data_sol[0]["q"][dof], color="blue")  # degré de liberté idx à tous les noeuds
        matplotlib.pyplot.plot(data_sol_prev[0]["q"][dof], color="yellow")
        matplotlib.pyplot.title(f"dof {dof}")
        # matplotlib.pyplot.plot(x_init[dof, :], color="red")  # degré de liberté idx à tous les noeuds
        # matplotlib.pyplot.plot(x_bounds_prev.min[dof, :], color="red")
        # matplotlib.pyplot.plot(x_bounds_prev.max[dof, :], color="red")
        matplotlib.pyplot.plot(x_bounds.min[dof, :], color="green")
        matplotlib.pyplot.plot(x_bounds.max[dof, :], color="green")


    matplotlib.pyplot.figure(2)
    matplotlib.pyplot.suptitle('Qdot et x_init')

    for dof in range(10):  # dof = degré de liberté
        matplotlib.pyplot.subplot(2, 5, int(dof + 1))
        matplotlib.pyplot.title(f"dof {dof}")
        matplotlib.pyplot.plot(data_sol[0]["q_dot"][dof], color="blue")
        matplotlib.pyplot.plot(data_sol_prev[0]["q_dot"][dof], color="yellow")
        # matplotlib.pyplot.plot(x_init[dof + n_q, :], color="red")
        matplotlib.pyplot.plot(x_bounds.min[dof + n_q, :], color="green")
        matplotlib.pyplot.plot(x_bounds.max[dof + n_q, :], color="green")
    matplotlib.pyplot.show()

    matplotlib.pyplot.figure(3)
    matplotlib.pyplot.suptitle('tau et u_init')
    for dof in range(10):
        matplotlib.pyplot.subplot(2, 5, int(dof + 1))
        matplotlib.pyplot.title(f"dof {dof}")
        matplotlib.pyplot.plot(data_sol[1]["tau"][dof], color="blue")
        matplotlib.pyplot.plot(data_sol_prev[1]["tau"][dof], color="yellow")
        # matplotlib.pyplot.plot(u_init[dof, :], color="red")
    matplotlib.pyplot.show()


if __name__ == "__main__":
    # Parameters
    biorbd_model_path = "/home/carla/Documents/Programmation/ViolinOptimalControl/models/BrasViolon.bioMod"
    biorbd_model = biorbd.Model(biorbd_model_path)
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()
    n_muscles = biorbd_model.nbMuscles()
    final_time = 1/3  # duration of the simulation
    nb_shooting_pts_window = 15  # size of NMPC window
    ns_tot = nb_shooting_pts_window* 10 # size of the entire optimization

    violin = Violin("E")
    bow = Bow("frog")

    bow_target_param = generate_up_and_down_bow_target(200) #np.load("bow_target_param.npy")  # generate_up_and_down_bow_target(200)


    X_est = np.zeros((n_qdot + n_q , ns_tot+1))
    U_est = np.zeros((n_tau, ns_tot))

    begin_at_first_iter = True
    if begin_at_first_iter == True :
        # Initial guess and bounds
        x0 = np.array(violin.initial_position()[bow.side] + [0] * n_qdot)

        x_init = np.tile(np.array(violin.initial_position()[bow.side] + [0] * n_qdot)[:, np.newaxis],
                         nb_shooting_pts_window+1)
        u_init = np.tile(np.array([0.5] * n_tau)[:, np.newaxis],
                         nb_shooting_pts_window)
    else:
        X_est_init = np.load('X_est.npy')
        # X_est_init = np.delete(X_est_init, np.s_[75:], axis=1)
        U_est_init = np.load('U_est.npy')
        # U_est_init = np.delete(U_est_init, np.s_[75:], axis=1)
        x0 = X_est_init[:, -1]
        x_init = X_est_init[:, -(nb_shooting_pts_window+1):]
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



    t = np.linspace(0, 2, ns_tot)
    target_curve = curve_integral(bow_target_param, t)
    q_target = np.ndarray((n_q, nb_shooting_pts_window + 1))
    target = np.ndarray((ns_tot * 2))
    Nmax = 250
    T = np.ndarray((Nmax))
    for i in range(Nmax):
        a=i%150
        T[i]=t[a]
    target = curve_integral(bow_target_param, T)


    shift = 1
    frame_to_init_from = 75

    # Init from known position
    ocp_load, sol_load = OptimalControlProgram.load(f"saved_iterations/{frame_to_init_from}_iter.bo")
    data_sol_prev = Data.get_data(ocp_load, sol_load, concatenate=False)
    x_init, u_init, X_out, U_out, x_bounds, u = warm_start_nmpc(sol=sol_load, shift=shift)
    U_est[:, :U_est_init.shape[1]] = U_est_init
    X_est[:, :X_est_init.shape[1]] = X_est_init


    # for i in range(frame_to_init_from, 150):
    for i in range(frame_to_init_from, ns_tot):
        q_target[bow.hair_idx, :] = target[i * shift: nb_shooting_pts_window + (i * shift) + 1]
        define_new_objectives()

        sol = ocp.solve(
            show_online_optim=False,
            solver_options={"max_iter": 1000, "hessian_approximation": "exact", "bound_push": 10 ** (-10),
                            "bound_frac": 10 ** (-10)}  # , "bound_push": 10**(-10), "bound_frac": 10**(-10)}
        )
        # sol = Simulate.from_controls_and_initial_states(ocp, x_init.initial_guess, u_init.initial_guess)
        x_init, u_init, X_out, U_out, x_bounds, u = warm_start_nmpc(sol=sol, shift=shift)

        X_est[:, i] = X_out
        U_est[:, i] = U_out

        ocp.save(sol, f"/saved_iterations/{i}_iter")  # you don't have to specify the extension ".bo"

    np.save("X_est", X_est)
    np.save("U_est", U_est)
    np.load(U_est)


        # sol = ocp.solve(
        #     show_online_optim=False,
        #     solver_options={"max_iter": 1000, "hessian_approximation": "exact", "bound_push": 10**(-10), "bound_frac": 10**(-10)}  #, "bound_push": 10**(-10), "bound_frac": 10**(-10)}
        # )

        # display_graphics()

        # ShowResult(ocp, sol).graphs()
        # print(f"NUMERO DE LA FENETRE : {i}")
        # data_sol = Data.get_data(ocp, sol, concatenate=False)

    #     x_init, u_init, x0, u0 = warm_start_nmpc(data_sol)
    #
    #     X_est = x0
    #     U_est = u0
    #
    # # --- Show results --- #
    # # ocp.save_get_data(sol, "up_and_down_NMPC")
    #
    # np.save("results", X_est)
    # bow_target_param = np.load("results.npy")

## Pour afficher mouvement global

# X_est_75iter = np.load('X_est.npy')
# U_est_75iter = np.load('U_est.npy')
# x0 = X_est_75iter[:, -1]
# x_init = X_est_75iter[:, -16:]
# u_init = U_est_75iter[:, -16:]

# x_init.shape
# (20, 16)
# X_est.shape
# (20, 91)
#
