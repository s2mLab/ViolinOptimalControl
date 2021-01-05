import biorbd
import numpy as np
from optimal_control_python.generate_bow_trajectory import generate_bow_trajectory, curve_integral
from optimal_control_python.utils import Bow, Violin
from optimal_control_python.utils_functions import prepare_generic_ocp, warm_start_nmpc, define_new_objectives


# Parameters
biorbd_model_path = "../models/BrasViolon.bioMod"
regenerate_bow_trajectory = False
biorbd_model = biorbd.Model(biorbd_model_path)
n_q = biorbd_model.nbQ()
n_qdot = biorbd_model.nbQdot()
n_tau = biorbd_model.nbGeneralizedTorque()
n_muscles = biorbd_model.nbMuscles()
final_time = 1/8  # duration of the
window_len = 15  # size of NMPC window
ns_tot_up_and_down = 150  # size of the up_and_down gesture

violin = Violin("E")
bow = Bow("frog")

if regenerate_bow_trajectory:
    np.save("bow_target_param", generate_bow_trajectory(200))
bow_target_param = np.load("bow_target_param.npy")
frame_to_init_from = window_len + 1
nb_shooting_pts_all_optim = 300

X_est = np.zeros((n_qdot + n_q, nb_shooting_pts_all_optim))
U_est = np.zeros((n_tau, nb_shooting_pts_all_optim))
Q_est = np.zeros((n_q, nb_shooting_pts_all_optim))
Qdot_est = np.zeros((n_qdot, nb_shooting_pts_all_optim))

begin_at_first_iter = True
if begin_at_first_iter:
    x0 = np.array(violin.initial_position()[bow.side] + [0] * n_qdot)

    x_init = np.tile(np.array(violin.initial_position()[bow.side] + [0] * n_qdot)[:, np.newaxis],
                     window_len + 1)
    u_init = np.tile(np.array([0.5] * n_tau)[:, np.newaxis],
                     window_len)
else:
    X_est_init = np.load('X_est.npy')[:, :frame_to_init_from+1]
    U_est_init = np.load('U_est.npy')[:, :frame_to_init_from+1]
    x_init = X_est_init[:, -(window_len + 1):]
    x0 = x_init[:, 0]
    u_init = U_est_init[:, -window_len:]

# position initiale de l'ocp
ocp, x_bounds = prepare_generic_ocp(
    biorbd_model_path=biorbd_model_path,
    number_shooting_points=window_len,
    final_time=final_time,
    x_init=x_init,
    u_init=u_init,
    x0=x0,
    acados=False,
    use_sx=False,
)


t = np.linspace(0, 2, ns_tot_up_and_down)
target_curve = curve_integral(bow_target_param, t)
q_target = np.ndarray((n_q, window_len + 1))
Nmax = nb_shooting_pts_all_optim+50
target = np.ndarray((Nmax, ))
T = np.ndarray((Nmax, ))
for i in range(Nmax):
    a = i % ns_tot_up_and_down
    T[i] = t[a]
target = curve_integral(bow_target_param, T)

shift = 1


# Init from known position
# ocp_load, sol_load = OptimalControlProgram.load(f"saved_iterations/{frame_to_init_from-1}_iter.bo")
# data_sol_prev = Data.get_data(ocp_load, sol_load, concatenate=False)
# x_init, u_init, X_out, U_out, x_bounds, u, lam_g, lam_x = warm_start_nmpc(
#     sol=sol_load,
#     ocp=ocp,
#     window_len=window_len,
#     n_q=n_q,
#     n_qdot=n_qdot,
#     n_tau=n_tau,
#     biorbd_model=biorbd_model,
#     acados=False,
#     shift=1
# )
# U_est[:, :U_est_init.shape[1]] = U_est_init
# X_est[:, :X_est_init.shape[1]] = X_est_init

for i in range(0, 30):
    print(f"iteration:{i}")
    q_target[bow.hair_idx, :] = target[i * shift: window_len + (i * shift) + 1]
    define_new_objectives(weight=1000, ocp=ocp, q_target=q_target, bow=bow)
    sol = ocp.solve(
        show_online_optim=False,
        solver_options={"max_iter": 1000, "hessian_approximation": "exact", "bound_push": 10 ** (-10),
                        "bound_frac": 10 ** (-10), "warm_start_init_point": "yes",
                        "warm_start_bound_push": 10 ** (-16), "warm_start_bound_frac": 10 ** (-16),
                        "nlp_scaling_method": "none", "warm_start_mult_bound_push": 10 ** (-16),
                        # "warm_start_slack_bound_push": 10 ** (-16)," warm_start_slack_bound_frac":10 ** (-16),
                        }
    )
    # sol = Simulate.from_controls_and_initial_states(ocp, x_init.initial_guess, u_init.initial_guess)
    x_init, u_init, X_out, U_out, x_bounds, u, lam_g, lam_x = warm_start_nmpc(
        sol=sol, ocp=ocp,
        window_len=window_len,
        n_q=n_q,
        n_qdot=n_qdot,
        n_tau=n_tau,
        biorbd_model=biorbd_model,
        acados=False,
        shift=shift
    )  # , lam_g, lam_x
    # x_init, u_init, X_out, U_out, x_bounds, u= warm_start_nmpc_same_iter(sol=sol, ocp=ocp, biorbd_model=biorbd_model)
    # warm_start_nmpc(sol, ocp, window_len, n_q, n_qdot, n_tau, biorbd_model, acados, shift=1)
    # A = lam_g
    # sol['lam_g'] = lam_g
    # # B = lam_x
    # sol['lam_x'] = lam_x
    for a in range(sol['lam_g'].shape[0]):
        sol['lam_g'][a] = lam_g[a]
    for a in range(sol['lam_x'].shape[0]):
        sol['lam_x'][a] = lam_x[a]

    # lam_g est un array et sol['lam_g'] est un DM, envoyer l'un dans l'autre est-t-il compatible ?
    # Ne semble pas être la bonne solution initiale... Le shift est-il exact? Pour passer d'une solution à la suivante?
    Q_est[:, i] = X_out[:10]
    X_est[:, i] = X_out
    Qdot_est[:, i] = X_out[10:]
    U_est[:, i] = U_out

    ocp.save(sol, f"saved_iterations/{i}_iter_acados")  # you don't have to specify the extension ".bo"

np.save("Q_est_", Q_est)
np.save("X_est_", X_est)
np.save("Qdot_est_", Qdot_est)
np.save("U_est_", U_est)
np.save("U_est_", U_est)

# Besides setting an option, you need to pass the 'x0', 'lam_g0', 'lam_x0' inputs to have an effect.
# Other relevant options to do warm-starting with ipopt: mu_init, warm_start_mult_bound_push,
# warm_start_slack_bound_push, warm_start_bound_push. Usually, you want to set these low.

# Pour afficher mouvement global

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
