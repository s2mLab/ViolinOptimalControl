import biorbd
import numpy as np

from biorbd_optim import (
    OptimalControlProgram,
    Objective,
    ObjectiveList,
    DynamicsType,
    DynamicsTypeOption,
    Constraint,
    ConstraintList,
    BoundsOption,
    QAndQDotBounds,
    InitialConditionsOption,
    Instant,
    InterpolationType,
    Data,
)
from optimal_control_python.utils import Bow, Violin

def prepare_ocp(biorbd_model_path, number_shooting_points, final_time, x_init, u_init, x0):
    biorbd_model = biorbd.Model(biorbd_model_path)
    n_tau = biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max, tau_init = -100, 100, 0

    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=1)

    dynamics = DynamicsTypeOption(DynamicsType.TORQUE_DRIVEN)

    x_bounds = BoundsOption(QAndQDotBounds(biorbd_model))
    x_bounds[:, 0] = x0
    x_init = InitialConditionsOption(x_init, interpolation=InterpolationType.EACH_FRAME)

    # Define control bounds
    u_bounds = BoundsOption([[tau_min] * n_tau, [tau_max] * n_tau])
    u_init = InitialConditionsOption(u_init, interpolation=InterpolationType.EACH_FRAME)

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
    )



# "/home/carla/Documents/Programmation/ViolinOptimalControl/models/BrasViolon.bioMod", ode_solver=OdeSolver.RK):

def warm_start_mhe(data_sol_prev):
    q = data_sol_prev[0]["q"]
    dq = data_sol_prev[0]["q_dot"]
    u = data_sol_prev[1]["tau"]
    x = np.vstack([q, dq])
    x_init = np.hstack((x[:, 1:], np.tile(x[:, [-1]], 1)))  # discard oldest estimate of the window, duplicates youngest
    u_init = u[:, 1:]  # discard oldest estimate of the window
    X_out = x[:, 0]
    return x_init, u_init, X_out


if __name__ == "__main__":
    biorbd_model_path = "/home/carla/Documents/Programmation/ViolinOptimalControl/models/BrasViolon.bioMod"
    biorbd_model = biorbd.Model(biorbd_model_path)
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()
    n_muscles = biorbd_model.nbMuscles()
    final_time = 2  # duration of the simulation
    nb_shooting_pts_window = 15  # size of MHE window
    # window_time = 0.5  # duration of a window simulation

    # Choose the string of the violin
    violon_string = Violin("D")
    inital_bow_side = Bow("frog")
    x0 = np.array(violon_string.initial_position()[inital_bow_side.side] + [0] * n_qdot)
    x_init = np.tile(np.array(violon_string.initial_position()[inital_bow_side.side] + [0] * n_qdot)[:, np.newaxis],
                     nb_shooting_pts_window+1)
    u_init = np.tile(np.array([0.5] * n_tau)[:, np.newaxis],
                     nb_shooting_pts_window)


    # X_est = np.zeros((biorbd_model.nbQ() * 2, int(number_shooting_points - window)))
    # U_est = np.zeros((biorbd_model.nbQ()*2, ))
    # for i in range(number_shooting_points - window):
    ocp = prepare_ocp(
        biorbd_model_path=biorbd_model_path,
        number_shooting_points=nb_shooting_pts_window,
        final_time=final_time,
        x_init=x_init,
        u_init=u_init,
        x0=x0,
    )

    new_objectives = ObjectiveList()
    new_objectives.add(Objective.Lagrange.ALIGN_MARKERS, first_marker_idx=Bow.contact_marker, second_marker_idx=violon_string.bridge_marker, idx=1)
    new_objectives.add(Objective.Mayer.TRACK_STATE, instant=Instant.END, states_idx=10, idx=2)
    ocp.update_objectives(new_objectives)

    new_constraints = ConstraintList()
    for j in range(1, 5):
        new_constraints.add(Constraint.ALIGN_MARKERS,
                            instant=j,
                            min_bound=0,
                            max_bound=0,
                            first_marker_idx=Bow.contact_marker,
                            second_marker_idx=violon_string.bridge_marker)
    for j in range(5, nb_shooting_pts_window):
        new_constraints.add(Constraint.ALIGN_MARKERS,
                            instant=j,
                            min_bound=-0.0000001*(10 ^ j),
                            max_bound=0.0000001*(10 ^ j),
                            first_marker_idx=Bow.contact_marker,
                            second_marker_idx=violon_string.bridge_marker)
    ocp.update_constraints(new_constraints)

    sol = ocp.solve(
        show_online_optim=False,
        solver_options={"max_iter": 1000, "hessian_approximation": "exact"}
    )
    data_sol = Data.get_data(ocp, sol, concatenate=False)
    x_init, u_init, x0 = warm_start_mhe(data_sol)
    X_est = x_init

    # --- Show results --- #
    ocp.save_get_data(sol, "up_and_down_NMPC")
    # np.save("coucou", X_est)




