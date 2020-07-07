import time

import biorbd
from casadi import MX, vertcat, if_else, lt, gt

from biorbd_optim import (
    Instant,
    InterpolationType,
    Axe,
    OptimalControlProgram,
    Dynamics,
    Problem,
    CustomPlot,
    PlotType,
    ProblemType,
    Objective,
    Constraint,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
)

from utils import Muscles


def xia_model_dynamic(states, controls, parameters, nlp):
    nbq = nlp["model"].nbQ()
    nbqdot = nlp["model"].nbQdot()
    nb_q_qdot = nbq + nbqdot

    q = states[:nbq]
    qdot = states[nbq:nb_q_qdot]
    active_fibers = states[nb_q_qdot : nb_q_qdot + nlp["nbMuscle"]]
    fatigued_fibers = states[nb_q_qdot + nlp["nbMuscle"] : nb_q_qdot + 2 * nlp["nbMuscle"]]
    resting_fibers = states[nb_q_qdot + 2 * nlp["nbMuscle"] :]

    residual_tau = controls[: nlp["nbTau"]]
    activation = controls[nlp["nbTau"] :]
    command = MX()

    comp = 0
    for i in range(nlp["model"].nbMuscleGroups()):
        for k in range(nlp["model"].muscleGroup(i).nbMuscles()):
            develop_factor = (
                nlp["model"].muscleGroup(i).muscle(k).characteristics().fatigueParameters().developFactor().to_mx()
            )
            recovery_factor = (
                nlp["model"].muscleGroup(i).muscle(k).characteristics().fatigueParameters().recoveryFactor().to_mx()
            )

            command = vertcat(
                command,
                if_else(
                    lt(active_fibers[comp], activation[comp]),
                    (
                        if_else(
                            gt(resting_fibers[comp], activation[comp] - active_fibers[comp]),
                            develop_factor * (activation[comp] - active_fibers[comp]),
                            develop_factor * resting_fibers[comp],
                        )
                    ),
                    recovery_factor * (active_fibers[comp] - activation[comp]),
                ),
            )
            comp += 1

    restingdot = -command + Muscles.R * fatigued_fibers
    activatedot = command - Muscles.F * active_fibers
    fatiguedot = Muscles.F * active_fibers - Muscles.R * fatigued_fibers

    muscles_states = biorbd.VecBiorbdMuscleState(nlp["nbMuscle"])
    for k in range(nlp["nbMuscles"]):
        muscles_states[k].setActivation(active_fibers[k])
    # todo fix force max

    muscles_tau = nlp["model"].muscularJointTorque(muscles_states, q, qdot).to_mx()
    # todo get muscle forces and multiplicate them by activate [k] and same as muscularJointTorque
    tau = muscles_tau + residual_tau

    qddot = biorbd.Model.ForwardDynamics(nlp["model"], q, qdot, tau).to_mx()
    return vertcat(qdot, qddot, activatedot, fatiguedot, restingdot)


def xia_model_configuration(ocp, nlp):
    Problem.configure_q_qdot(nlp, True, False)
    Problem.configure_tau(nlp, False, True)
    Problem.configure_muscles(nlp, False, True)

    x = MX()
    for i in range(nlp["nbMuscle"]):
        x = vertcat(x, MX.sym(f"Muscle_{nlp['muscleNames']}_active", 1, 1))
    for i in range(nlp["nbMuscle"]):
        x = vertcat(x, MX.sym(f"Muscle_{nlp['muscleNames']}_fatigue", 1, 1))
    for i in range(nlp["nbMuscle"]):
        x = vertcat(x, MX.sym(f"Muscle_{nlp['muscleNames']}_resting", 1, 1))

    nlp["x"] = vertcat(nlp["x"], x)
    nlp["var_states"]["muscles_active"] = nlp["nbMuscle"]
    nlp["var_states"]["muscles_fatigue"] = nlp["nbMuscle"]
    nlp["var_states"]["muscles_resting"] = nlp["nbMuscle"]
    nlp["nx"] = nlp["x"].rows()

    nb_q_qdot = nlp["nbQ"] + nlp["nbQdot"]
    nlp["plot"]["muscles_active"] = CustomPlot(
        lambda x, u, p: x[nb_q_qdot : nb_q_qdot + nlp["nbMuscle"]],
        plot_type=PlotType.INTEGRATED,
        legend=nlp["muscleNames"],
        color="r",
        ylim=[0, 1],
    )

    combine = "muscles_active"
    nlp["plot"]["muscles_fatigue"] = CustomPlot(
        lambda x, u, p: x[nb_q_qdot + nlp["nbMuscle"] : nb_q_qdot + 2 * nlp["nbMuscle"]],
        plot_type=PlotType.INTEGRATED,
        legend=nlp["muscleNames"],
        combine_to=combine,
        color="g",
        ylim=[0, 1],
    )
    nlp["plot"]["muscles_resting"] = CustomPlot(
        lambda x, u, p: x[nb_q_qdot + 2 * nlp["nbMuscle"] : nb_q_qdot + 3 * nlp["nbMuscle"]],
        plot_type=PlotType.INTEGRATED,
        legend=nlp["muscleNames"],
        combine_to=combine,
        color="b",
        ylim=[0, 1],
    )

    Problem.configure_forward_dyn_func(ocp, nlp, nlp["problem_type"]["dynamic"])


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

    # --- Objective --- #
    objective_functions = (
        # {"type": Objective.Lagrange.MINIMIZE_MUSCLES_CONTROL, "weight": 10},
        {"type": Objective.Lagrange.MINIMIZE_TORQUE, "weight": 1},
        # {"type": Objective.Lagrange.MINIMIZE_TORQUE, "controls_idx": [0, 1, 2, 3], "weight": 2000},
    )

    # --- Dynamics --- #
    problem_type = {"type": ProblemType.CUSTOM, "configure": xia_model_configuration, "dynamic": xia_model_dynamic}

    # --- Constraints --- #
    constraints = ()

    # --- Path constraints --- #
    X_bounds = QAndQDotBounds(biorbd_model)

    X_bounds.min[biorbd_model.nbQ() :, 0] = 0
    X_bounds.max[biorbd_model.nbQ() :, 0] = 0
    X_bounds.min[biorbd_model.nbQ() :, 2] = -1.5
    X_bounds.max[biorbd_model.nbQ() :, 2] = -1.5

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

    X_bounds.concatenate(muscle_states_bounds)

    U_bounds = Bounds(
        [torque_min] * biorbd_model.nbGeneralizedTorque() + [muscle_states_ratio_min] * biorbd_model.nbMuscleTotal(),
        [torque_max] * biorbd_model.nbGeneralizedTorque() + [muscle_states_ratio_max] * biorbd_model.nbMuscleTotal(),
    )

    # --- Initial guess --- #
    X_init = InitialConditions([0] * biorbd_model.nbQ() + [0] * biorbd_model.nbQdot(), InterpolationType.CONSTANT,)
    U_init = InitialConditions(
        [torque_init] * biorbd_model.nbGeneralizedTorque() + [muscle_activated_init] * biorbd_model.nbMuscleTotal(),
        InterpolationType.CONSTANT,
    )

    muscle_states_init = InitialConditions(
        [muscle_activated_init] * biorbd_model.nbMuscleTotal()
        + [muscle_fatigued_init] * biorbd_model.nbMuscleTotal()
        + [muscle_resting_init] * biorbd_model.nbMuscleTotal(),
        InterpolationType.CONSTANT,
    )
    X_init.concatenate(muscle_states_init)

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        problem_type,
        number_shooting_points,
        final_time,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        objective_functions,
        constraints,
        nb_threads=4,
    )


if __name__ == "__main__":
    ocp = prepare_nlp()

    # --- Solve the program --- #
    tic = time.time()
    sol, sol_iterations = ocp.solve(
        show_online_optim=True,
        return_iterations=True,
        options_ipopt={"tol": 1e-4, "max_iter": 3000, "ipopt.hessian_approximation": "limited-memory"},
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
