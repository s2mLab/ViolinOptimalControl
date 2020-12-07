import biorbd
import numpy as np
import matplotlib
from optimal_control_python.generate_up_and_down_bow_target import generate_up_and_down_bow_target
from optimal_control_python.generate_up_and_down_bow_target import curve_integral
from optimal_control_python.utils import Bow, Violin
from optimal_control_python.utils_functions import prepare_generic_ocp, warm_start_nmpc, define_new_objectives, \
    display_graphics_X_est, display_X_est, compare_target



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
    ns_tot_up_and_down = 150 # size of the up_and_down gesture

    violin = Violin("E")
    bow = Bow("frog")

    # np.save("bow_target_param", generate_up_and_down_bow_target(200))
    bow_target_param = np.load("bow_target_param.npy")
    frame_to_init_from = 290
    nb_shooting_pts_all_optim = 300

    X_est = np.zeros((n_qdot + n_q , nb_shooting_pts_all_optim))
    U_est = np.zeros((n_tau, nb_shooting_pts_all_optim))
    Q_est = np.zeros((n_q , nb_shooting_pts_all_optim))
    Qdot_est = np.zeros((n_qdot , nb_shooting_pts_all_optim))

    begin_at_first_iter = True
    if begin_at_first_iter == True :
        x0 = np.array(violin.initial_position()[bow.side] + [0] * n_qdot)

        x_init = np.tile(np.array(violin.initial_position()[bow.side] + [0] * n_qdot)[:, np.newaxis],
                         nb_shooting_pts_window+1)
        u_init = np.tile(np.array([0.5] * n_tau)[:, np.newaxis],
                         nb_shooting_pts_window)
    else:
        X_est_init = np.load('X_est.npy')[:, :frame_to_init_from+1]
        U_est_init = np.load('U_est.npy')[:, :frame_to_init_from+1]
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
        acados=False,
        useSX=False,
    )


    t = np.linspace(0, 2, ns_tot_up_and_down)
    target_curve = curve_integral(bow_target_param, t)
    q_target = np.ndarray((n_q, nb_shooting_pts_window + 1))
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
    # x_init, u_init, X_out, U_out, x_bounds, u, lam_g, lam_x = warm_start_nmpc(sol=sol_load, ocp=ocp, nb_shooting_pts_window=nb_shooting_pts_window, n_q, n_qdot, n_tau, biorbd_model, acados, shift=1)
    # U_est[:, :U_est_init.shape[1]] = U_est_init
    # X_est[:, :X_est_init.shape[1]] = X_est_init



    for i in range(0, 20):
        q_target[bow.hair_idx, :] = target[i * shift: nb_shooting_pts_window + (i * shift) + 1]
        define_new_objectives(weight=10000, ocp=ocp, q_target=q_target, bow=bow)
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
        x_init, u_init, X_out, U_out, x_bounds, u= warm_start_nmpc(sol=sol, ocp=ocp,
                                                                    nb_shooting_pts_window=nb_shooting_pts_window,
                                                                    n_q = n_q, n_qdot=n_qdot, n_tau = n_tau,
                                                                    biorbd_model=biorbd_model,
                                                                    acados=True, shift=shift) #, lam_g, lam_x
        # warm_start_nmpc(sol, ocp, nb_shooting_pts_window, n_q, n_qdot, n_tau, biorbd_model, acados, shift=1)
        # sol['lam_g'] = lam_g
        # sol['lam_x'] = lam_x
        Q_est[:, i] = X_out[:10]
        X_est[:, i] = X_out
        Qdot_est[:, i] = X_out[10:]
        U_est[:, i] = U_out

        ocp.save(sol, f"saved_iterations/{i}_iter_acados")  # you don't have to specify the extension ".bo"

    np.save("Q_est_acados", Q_est)
    np.save("X_est_acados", X_est)
    np.save("Qdot_est_acados", Qdot_est)
    np.save("U_est_acados", U_est)




# Besides setting an option, you need to pass the 'x0', 'lam_g0', 'lam_x0' inputs to have an effect.
# Other relevant options to do warm-starting with ipopt: mu_init, warm_start_mult_bound_push, warm_start_slack_bound_push, warm_start_bound_push. Usually, you want to set these low.

## Pour afficher mouvement global

# ocp, x_bounds = prepare_generic_ocp(
#     biorbd_model_path=biorbd_model_path,
#     number_shooting_points=nb_shooting_pts_all_optim,
#     final_time=2,
#     x_init=X_est,
#     u_init=U_est,
#     x0=x0,
#     )
# sol = Simulate.from_controls_and_initial_states(ocp, x_init.initial_guess, u_init.initial_guess)
# ShowResult(ocp, sol).graphs()