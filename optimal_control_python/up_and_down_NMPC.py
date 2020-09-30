import biorbd

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
    OdeSolver,
)
from optimal_control_python.utils import Bow, Violin


def prepare_ocp(biorbd_model_path="/home/carla/Documents/Programmation/ViolinOptimalControl/models/BrasViolon.bioMod", ode_solver=OdeSolver.RK):
    biorbd_model = biorbd.Model(biorbd_model_path)
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()


    # Problem parameters
    number_shooting_points = 15
    final_time = 0.5

    tau_min, tau_max, tau_init = -100, 100, 0

    # Choose the string of the violin
    violon_string = Violin("E")
    inital_bow_side = Bow("frog")

    # Objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE)
    objective_functions.add(Objective.Mayer.ALIGN_MARKERS, first_marker_idx=Bow.segment_idx,
                            second_marker_idx=violon_string.bridge_marker)

    # Dynamics
    dynamics = DynamicsTypeOption(DynamicsType.TORQUE_DRIVEN)

    # Constraints
    constraints = ConstraintList()
    constraints.add(Constraint.ALIGN_MARKERS, instant=Instant.START, first_marker_idx=Bow.frog_marker,
                    second_marker_idx=violon_string.bridge_marker)
    constraints.add(Constraint.ALIGN_MARKERS, instant=Instant.END, first_marker_idx=Bow.tip_marker,
                    second_marker_idx=violon_string.bridge_marker)


    # Path constraint
    x_bounds = BoundsOption(QAndQDotBounds(biorbd_model))

    x_bounds.min[n_q:n_q+n_qdot, [0, -1]] = 0
    x_bounds.max[n_q:n_q+n_qdot, [0, -1]] = 0

    # Initial guess
    x_init = InitialConditionsOption(
        violon_string.initial_position()[inital_bow_side.side] + [0] * n_qdot
    )

    # Define control bounds
    u_bounds = BoundsOption([[tau_min] * n_tau, [tau_max] * n_tau])

    u_init = InitialConditionsOption([tau_init] * n_tau)

    # _________________#

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
        # show_online_optim = show_online_optim,
        use_SX=False
    )


if __name__ == "__main__":
    ocp = prepare_ocp()

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=False,
                    solver_options={"max_iter": 1000, "hessian_approximation": "exact"})
    #result = ShowResult(ocp, sol)
    # result.graphs()
    # result.animate()

    # # --- Show results --- #
    # result = ShowResult(ocp, sol)
    ocp.save_get_data(sol, "up_and_down_NMPC")
    # result.keep_matplotlib()
    # result.show_biorbd_viz(show_meshes=False)
