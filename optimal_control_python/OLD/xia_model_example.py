import time

import biorbd_casadi as biorbd
from bioptim import (
    InterpolationType,
    OptimalControlProgram,
    DynamicsTypeOption,
    ObjectiveList,
    Objective,
    Bounds,
    QAndQDotBounds,
    InitialConditionsOption,
    ShowResult,
)

from violin_ocp import xia as xia


def prepare_nlp(biorbd_model_path="../models/Bras.bioMod"):
    """
    Mix .bioMod and users data to call OptimalControlProgram constructor.
    :param biorbd_model_path: path to the .bioMod file.
    :param show_online_optim: bool which active live plot function.
    :return: OptimalControlProgram object.
    """

    # --- Options --- #
    biorbd_model = biorbd.Model(biorbd_model_path)
    muscle_activated_init, muscle_fatigued_init, muscle_resting_init = 0, 0, 1
    torque_min, torque_max, torque_init = -10, 10, 0
    muscle_states_ratio_min, muscle_states_ratio_max = 0, 1
    number_shooting_points = 30
    final_time = 0.5

    # --- ObjectiveFcn --- #
    objective_functions = ObjectiveList()
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=10)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, name="tau", weight=1)
    objective_functions.add(Objective.Lagrange.MINIMIZE_CONTROL, name="tau", derivative=True, weight=100)
    # objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, name="tau", controls_idx=[0, 1, 2, 3], weight=2000)

    # --- Dynamics --- #
    dynamics = DynamicsTypeOption(xia.xia_model_configuration, dynamic_function=xia.xia_model_dynamic)

    # --- Path constraints --- #
    X_bounds = QAndQDotBounds(biorbd_model)

    X_bounds[biorbd_model.nbQ() :, 0] = 0
    X_bounds[biorbd_model.nbQ() :, 2] = -1.5

    muscle_states_bounds = Bounds(
        [muscle_states_ratio_min] * biorbd_model.nbMuscleTotal() * 3,
        [muscle_states_ratio_max] * biorbd_model.nbMuscleTotal() * 3,
    )
    muscle_states_bounds.min[:, 0] = (
        [muscle_activated_init] * biorbd_model.nbMuscleTotal()
        + [muscle_fatigued_init] * biorbd_model.nbMuscleTotal()
        + [muscle_resting_init] * biorbd_model.nbMuscleTotal()
    )
    muscle_states_bounds.max[:, 0] = (
        [muscle_activated_init] * biorbd_model.nbMuscleTotal()
        + [muscle_fatigued_init] * biorbd_model.nbMuscleTotal()
        + [muscle_resting_init] * biorbd_model.nbMuscleTotal()
    )

    X_bounds.bounds.concatenate(muscle_states_bounds.bounds)

    U_bounds = Bounds(
        [torque_min] * biorbd_model.nbGeneralizedTorque() + [muscle_states_ratio_min] * biorbd_model.nbMuscleTotal(),
        [torque_max] * biorbd_model.nbGeneralizedTorque() + [muscle_states_ratio_max] * biorbd_model.nbMuscleTotal(),
    )

    # --- Initial guess --- #
    X_init = InitialConditionsOption(
        [0] * biorbd_model.nbQ() + [0] * biorbd_model.nbQdot(),
        InterpolationType.CONSTANT,
    )
    U_init = InitialConditionsOption(
        [torque_init] * biorbd_model.nbGeneralizedTorque() + [muscle_activated_init] * biorbd_model.nbMuscleTotal(),
        InterpolationType.CONSTANT,
    )

    muscle_states_init = InitialConditionsOption(
        [muscle_activated_init] * biorbd_model.nbMuscleTotal()
        + [muscle_fatigued_init] * biorbd_model.nbMuscleTotal()
        + [muscle_resting_init] * biorbd_model.nbMuscleTotal(),
        InterpolationType.CONSTANT,
    )
    X_init.initial_condition.concatenate(muscle_states_init.initial_condition)

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions=objective_functions,
        nb_threads=4,
    )


if __name__ == "__main__":
    ocp = prepare_nlp()

    # --- Solve the program --- #
    tic = time.time()
    sol, sol_iterations = ocp.solve(
        show_online_optim=True,
        return_iterations=True,
        solver_options={"tol": 1e-4, "max_iter": 3000, "ipopt.hessian_approximation": "limited-memory"},
    )
    toc = time.time() - tic
    print(f"Time to solve : {toc}sec")

    t = time.localtime(time.time())
    date = f"{t.tm_year}_{t.tm_mon}_{t.tm_mday}"
    OptimalControlProgram.save(ocp, sol, f"results/{date}_xiaModel.bo")
    OptimalControlProgram.save_get_data(ocp, sol, f"results/{date}_xiaModel.bob", sol_iterations=sol_iterations)
    OptimalControlProgram.save_get_data(ocp, sol, f"results/{date}_xiaModel_interpolate.bob", interpolate_nb_frames=100)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.graphs()
