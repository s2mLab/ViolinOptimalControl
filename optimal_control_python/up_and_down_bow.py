import time
import pickle

import biorbd
import numpy as np
from biorbd_optim import (
    Instant,
    InterpolationType,
    InitialConditionsList,
    BoundsList,
    OdeSolver,
    OptimalControlProgram,
    DynamicsTypeList,
    PathCondition,
    ConstraintList,
    Objective,
    ObjectiveList,
    Constraint,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
)

from utils import Bow, Violin, xia


def prepare_ocp(biorbd_model_path="../models/BrasViolon.bioMod"):
    """
    Mix .bioMod and users data to call OptimalControlProgram constructor.
    :param biorbd_model_path: path to the .bioMod file.
    :param show_online_optim: bool which active live plot function.
    :return: OptimalControlProgram object.
    """
    optimal_initial_values = False
    nb_phases = 2

    biorbd_model = []
    objective_functions = ObjectiveList()
    dynamics = DynamicsTypeList()
    constraints = ConstraintList()
    x_bounds = BoundsList()
    u_bounds = BoundsList()
    x_init = InitialConditionsList()
    u_init = InitialConditionsList()

    # --- Options --- #
    number_shooting_points = [20] * nb_phases
    final_time = [1] * nb_phases

    muscle_activated_init, muscle_fatigued_init, muscle_resting_init = 0, 0, 1
    torque_min, torque_max, torque_init = -10, 10, 0
    muscle_states_min, muscle_states_max = 0, 1

    # --- Aliasing --- #
    model_tp = biorbd.Model(biorbd_model_path)
    n_q = model_tp.nbQ()
    n_qdot = model_tp.nbQdot()
    n_tau = model_tp.nbGeneralizedTorque()
    n_mus = model_tp.nbMuscles()
    violon_string = Violin("G")
    inital_bow_side = Bow("frog")

    # --- External forces --- #
    external_forces = [
        np.repeat(violon_string.external_force[:, np.newaxis], number_shooting_points[0], axis=1)
    ] * nb_phases

    for idx_phase in range(nb_phases):

        biorbd_model.append(biorbd.Model(biorbd_model_path))

        # --- Objective --- #
        objective_functions.add(Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=1, phase=idx_phase)
        objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, weight=1, phase=idx_phase)
        objective_functions.add(
            Objective.Lagrange.ALIGN_SEGMENT_WITH_CUSTOM_RT,
            weight=1,
            segment_idx=Bow.segment_idx,
            rt_idx=violon_string.rt_on_string,
            phase=idx_phase,
        )
        objective_functions.add(
            Objective.Lagrange.MINIMIZE_TORQUE, controls_idx=[0, 1, 2, 3], weight=10, phase=idx_phase
        )

        # --- Dynamics --- #
        dynamics.add(xia.xia_model_configuration, dynamic_function=xia.xia_model_dynamic)

        # --- Constraints --- #
        if idx_phase == 0:
            constraints.add(
                Constraint.ALIGN_MARKERS,
                instant=Instant.START,
                min_bound=-0.00001,
                max_bound=0.00001,
                first_marker_idx=Bow.frog_marker,
                second_marker_idx=violon_string.bridge_marker,
                phase=idx_phase,
            )
        constraints.add(
            Constraint.ALIGN_MARKERS,
            instant=Instant.MID,
            min_bound=-0.00001,
            max_bound=0.00001,
            first_marker_idx=Bow.tip_marker,
            second_marker_idx=violon_string.bridge_marker,
            phase=idx_phase,
        )
        constraints.add(
            Constraint.ALIGN_MARKERS,
            instant=Instant.END,
            min_bound=-0.00001,
            max_bound=0.00001,
            first_marker_idx=Bow.frog_marker,
            second_marker_idx=violon_string.bridge_marker,
            phase=idx_phase,
        )
        # constraints.add(
        #     Constraint.ALIGN_SEGMENT_WITH_CUSTOM_RT,
        #     instant=Instant.ALL,
        #     min_bound=-0.00001,
        #     max_bound=0.00001,
        #     segment_idx=Bow.segment_idx,
        #     rt_idx=violon_string.rt_on_string,
        #     phase=idx_phase,
        # )
        # constraints.add(
        #     Constraint.ALIGN_MARKER_WITH_SEGMENT_AXIS,
        #     instant=Instant.ALL,
        #     min_bound=-0.00001,
        #     max_bound=0.00001,
        #     marker_idx=violon_string.bridge_marker,
        #     segment_idx=Bow.segment_idx,
        #     axis=(Axe.Y),
        #     phase=idx_phase,
        # )
        constraints.add(
            Constraint.ALIGN_MARKERS,
            instant=Instant.ALL,
            first_marker_idx=Bow.contact_marker,
            second_marker_idx=violon_string.bridge_marker,
            phase=idx_phase,
            # TODO: add constraint about velocity in a marker of bow (start and end instant)
        )

        # --- Path constraints --- #
        x_bounds.add(QAndQDotBounds(biorbd_model[0]), phase=idx_phase)

        # Start and finish with zero velocity
        if idx_phase == 0:
            x_bounds[idx_phase][n_q :, 0] = 0
        if idx_phase == nb_phases - 1:
            x_bounds[idx_phase][n_q :, -1] = 0

        muscle_states_bounds = Bounds(
            [muscle_states_min] * n_mus * 3,
            [muscle_states_max] * n_mus * 3,
        )
        if idx_phase == 0:
            # fatigued_fibers = activated_fibers = 0 and resting_fibers = 1 at start
            muscle_states_bounds[:2 * n_mus, 0] = 0
            muscle_states_bounds[2 * n_mus:, 0] = 1
        x_bounds[idx_phase].concatenate(muscle_states_bounds)

        u_bounds.add(
            [[torque_min] * n_tau + [muscle_states_min] * n_mus, [torque_max] * n_tau + [muscle_states_max] * n_mus],
            phase=idx_phase,
        )

        # --- Initial guess --- #
        if optimal_initial_values:
            # TODO: Fix this part (avoid using .bio)
            raise NotImplementedError("optimal_initial_values = True should be reviewed")
            if idx_phase == 0:
                with open(f"utils/optimal_init_{number_shooting_points[0]}_nodes_constr_first.bio", "rb") as file:
                    dict = pickle.load(file)
            else:
                with open(f"utils/optimal_init_{number_shooting_points[0]}_nodes_constr_first.bio", "rb") as file:
                    dict = pickle.load(file)

            x_init.add(dict["states"], interpolation=InterpolationType.EACH_FRAME, phase=idx_phase)
            u_init.add(dict["controls"], interpolation=InterpolationType.EACH_FRAME, phase=idx_phase)

        else:
            # TODO: x_init could be a LINEAR from frog to tip
            x_init.add(
                violon_string.initial_position()[inital_bow_side.side] + [0] * n_qdot,
                interpolation=InterpolationType.CONSTANT,
                phase=idx_phase,
            )
            muscle_states_init = InitialConditions(
                [muscle_activated_init] * n_mus + [muscle_fatigued_init] * n_mus + [muscle_resting_init] * n_mus,
                interpolation=InterpolationType.CONSTANT,
            )
            x_init[idx_phase].concatenate(muscle_states_init)

            u_init.add(
                [torque_init] * n_tau + [muscle_activated_init] * n_mus,
                interpolation=InterpolationType.CONSTANT,
                phase=idx_phase,
            )
    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        external_forces=external_forces,
        ode_solver=OdeSolver.RK,
        nb_threads=4,
    )


if __name__ == "__main__":
    ocp = prepare_ocp()

    # --- Solve the program --- #
    tic = time.time()
    sol, sol_obj = ocp.solve(
        show_online_optim=True,
        return_iterations=False,
        return_objectives=True,
        solver_options={
            "tol": 1e-3,
            "max_iter": 500000,
            "ipopt.bound_push": 1e-10,
            "ipopt.bound_frac": 1e-10,
            "ipopt.hessian_approximation": "limited-memory",
            "output_file": "output.bot",
            "ipopt.linear_solver": "mumps",  # "mumps",  # "ma57", "ma86"
            # "file_print_level": 5,
        },
    )
    toc = time.time() - tic
    print(f"Time to solve : {toc}sec")

    analyse = Objective.Printer(ocp, sol_obj)

    t = time.localtime(time.time())
    date = f"{t.tm_year}_{t.tm_mon}_{t.tm_mday}"
    OptimalControlProgram.save(ocp, sol, f"results/{date}_upDown.bo")
    OptimalControlProgram.save_get_data(ocp, sol, f"results/{date}_upDown.bob")
    OptimalControlProgram.save_get_data(ocp, sol, f"results/{date}_upDown_interpolate.bob", interpolate_nb_frames=100)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.graphs()
