import biorbd
import numpy as np
import matplotlib
from optimal_control_python.generate_bow_trajectory import generate_bow_trajectory, curve_integral
from optimal_control_python.utils import Bow, Violin
from optimal_control_python.utils_functions import prepare_generic_ocp, warm_start_nmpc


from bioptim import (
    Objective,
    ObjectiveList,
    Node,
    Solver,
    Simulate,
    OptimalControlProgram,
    Data,
)


def define_new_objectives(weight):
    new_objectives = ObjectiveList()
    new_objectives.add(
        Objective.Lagrange.TRACK_STATE, node=Node.ALL, weight=weight, target=q_target[bow.hair_idx:bow.hair_idx+1, :],
        index=bow.hair_idx,
        list_index=3
    )
    new_objectives.add(
        Objective.Lagrange.ALIGN_MARKERS, node=Node.ALL, weight=1000000, first_marker_idx=Bow.contact_marker,
        second_marker_idx=violin.bridge_marker, list_index=4)
    new_objectives.add(
        Objective.Lagrange.MINIMIZE_ALL_CONTROLS, node=Node.ALL, weight=10, list_index=5)
    new_objectives.add(
        Objective.Lagrange.MINIMIZE_STATE, node=Node.ALL, weight=10, list_index=6)
    # new_objectives.add(Objective.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, node=Node.ALL, states_idx=bow.hair_idx,
    #                    # weight=1,
    #                    idx=4)  # rajoute des itérations et ne semble riuen changer au mouvement...
    ocp.update_objectives(new_objectives)

def display_graphics_X_est():
    matplotlib.pyplot.suptitle('X_est')
    for dof in range(10):
        matplotlib.pyplot.subplot(2, 5, int(dof + 1))
        if dof == 9:
            matplotlib.pyplot.plot(target[:Q_est_acados.shape[1]], color="red")
        matplotlib.pyplot.plot(Q_est_acados[dof, :], color="blue")
        matplotlib.pyplot.title(f"dof {dof}")
        matplotlib.pyplot.show()

def display_X_est():
    matplotlib.pyplot.suptitle('X_est and target')
    matplotlib.pyplot.plot(target[:Q_est_acados.shape[1]], color="red")
    matplotlib.pyplot.title(f"target")
    matplotlib.pyplot.plot(Q_est_acados[9, :], color="blue")
    matplotlib.pyplot.title(f"dof {9}")
    matplotlib.pyplot.show()


# Parameters
biorbd_model_path = "../models/BrasViolon.bioMod"
biorbd_model = biorbd.Model(biorbd_model_path)
n_q = biorbd_model.nbQ()
n_qdot = biorbd_model.nbQdot()
n_tau = biorbd_model.nbGeneralizedTorque()
n_muscles = biorbd_model.nbMuscles()
window_time = 1/8  # duration of the window
nb_shooting_pts_window = 80  # size of NMPC window
ns_tot_up_and_down = 150 # size of the up_and_down gesture

violin = Violin("E")
bow = Bow("frog")

# np.save("bow_target_param", generate_bow_trajectory(200))
bow_target_param = np.load("bow_target_param.npy")
frame_to_init_from = nb_shooting_pts_window
nb_shooting_pts_all_optim = 600

Q_est_acados = np.zeros((n_q , nb_shooting_pts_all_optim))
X_est_acados= np.zeros((n_q+n_qdot , nb_shooting_pts_all_optim))
Qdot_est_acados = np.zeros((n_qdot , nb_shooting_pts_all_optim))
U_est_acados = np.zeros((n_tau, nb_shooting_pts_all_optim))
begin_at_first_iter = False
if begin_at_first_iter == True :
    # Initial guess and bounds
    x0 = np.array(violin.initial_position()[bow.side] + [0] * n_qdot)

    x_init = np.tile(np.array(violin.initial_position()[bow.side] + [0] * n_qdot)[:, np.newaxis],
                     nb_shooting_pts_window+1)
    u_init = np.tile(np.array([0.5] * n_tau)[:, np.newaxis],
                     nb_shooting_pts_window)
else:
    X_est_init = np.load('X_est.npy')[:, :nb_shooting_pts_window+1]
    U_est_init = np.load('U_est.npy')[:, :nb_shooting_pts_window+1]
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
    acados=True,
    useSX=True,
)



t = np.linspace(0, 2, ns_tot_up_and_down)
target_curve = curve_integral(bow_target_param, t)
q_target = np.ndarray((n_q, nb_shooting_pts_window + 1))
Nmax = nb_shooting_pts_all_optim + nb_shooting_pts_window
target = np.ndarray(Nmax)
T = np.ndarray((Nmax))
for i in range(Nmax):
    a=i % ns_tot_up_and_down
    T[i] = t[a]
target = curve_integral(bow_target_param, T)
shift = 1


# Init from known position
# ocp_load, sol_load = OptimalControlProgram.load(f"saved_iterations/{frame_to_init_from-1}_iter_acados.bo")
# data_sol_prev = Data.get_data(ocp_load, sol_load, concatenate=False)
# x_init, u_init, X_out, U_out, x_bounds, u = warm_start_nmpc(sol=sol_load, ocp=ocp,
# nb_shooting_pts_window=nb_shooting_pts_window, n_q=n_q, n_qdot=n_qdot, n_tau=n_tau, biorbd_model=biorbd_model,
#                                                             acados=True, shift=1)
# U_est_acados[:, :U_est_init.shape[1]] = U_est_init
# X_est_acados[:, :X_est_init.shape[1]] = X_est_init
# Q_est_acados[:, :X_est_init.shape[1]]=X_est_init[:10]
# Qdot_est_acados[:, :X_est_init.shape[1]] = X_est_init[10:]



# for i in range(frame_to_init_from, 200):
for i in range(0, 150):
    print(f"iteration:{i}")
    if i < 300:
        q_target[bow.hair_idx, :] = target[i * shift: nb_shooting_pts_window + (i * shift) + 1]
        new_objectives = ObjectiveList()
        new_objectives.add(
            Objective.Lagrange.ALIGN_MARKERS, node=Node.ALL, weight=100000, first_marker_idx=Bow.contact_marker,
            second_marker_idx=violin.bridge_marker, list_index=1)
        new_objectives.add(
            Objective.Lagrange.MINIMIZE_ALL_CONTROLS, node=Node.ALL, weight=10, list_index=2)
        new_objectives.add(
            Objective.Lagrange.MINIMIZE_STATE, node=Node.ALL, weight=10, list_index=3)
        new_objectives.add(
            Objective.Lagrange.TRACK_STATE, node=Node.ALL, weight=100000,
            target=q_target[bow.hair_idx:bow.hair_idx + 1, :],
            index=bow.hair_idx,
            list_index=4
        )
        ocp.update_objectives(new_objectives)
    else:
        q_target[bow.hair_idx, :] = target[i-40 * shift: nb_shooting_pts_window + (i-40 * shift) + 1]
        if target[i] < -0.45: # but : mettre des poids plus lourds aux extremums de la target pour que les extremums
            weight = 1500 # ne soient pas dépassés par le poids des des autres valeurs "itermédiaires" de la target
        if target[i] > -0.17: # qui sont majoritaire dans la fenêtre
            weight = 1500
        else:
            weight = 1000
        define_new_objectives(weight=weight)
    if i == 0:
        sol = ocp.solve(
            show_online_optim=False,
            solver=Solver.ACADOS,
            solver_options={"nlp_solver_max_iter": 1000},
        )
    else:
        sol = ocp.solve(
            show_online_optim=False,
            solver=Solver.ACADOS,
        )
    x_init, u_init, X_out, U_out, x_bounds, u = warm_start_nmpc(sol=sol, ocp=ocp,
                                                                nb_shooting_pts_window=nb_shooting_pts_window,
                                                                n_q=n_q, n_qdot=n_qdot, n_tau=n_tau,
                                                                biorbd_model=biorbd_model,
                                                                acados=True, shift=shift)
    Q_est_acados[:, i] = X_out[:10]
    X_est_acados[:, i] = X_out
    Qdot_est_acados[:, i] = X_out[10:]
    U_est_acados[:, i] = U_out

    # ocp.save(sol, f"saved_iterations/{i}_iter_acados")  # you don't have to specify the extension ".bo"

np.save("Q_est_acados", Q_est_acados)
np.save("X_est_acados", X_est_acados)
np.save("Qdot_est_acados", Qdot_est_acados)
np.save("U_est_acados", U_est_acados)
out = np.load("U_est_acados.npy")

