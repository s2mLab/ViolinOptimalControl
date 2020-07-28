import time
import pickle
import biorbd
import numpy as np
from casadi import MX, vertcat, if_else, lt, gt

from biorbd_optim import (
    Instant,
    InterpolationType,
    InitialConditionsList,
    Axe,
    BoundsList,
    OptimalControlProgram,
    DynamicsTypeList,
    PathCondition,
    Problem,
    ConstraintList,
    CustomPlot,
    PlotType,
    Objective,
    ObjectiveList,
    Constraint,
    Bounds,
    QAndQDotBounds,
    InitialConditions,
    ShowResult,
)

from utils import Bow, Violin, Muscles


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
                    recovery_factor * (activation[comp] - active_fibers[comp]),
                ),
            )
            comp += 1
    restingdot = -command + Muscles.r * Muscles.R * fatigued_fibers  # todo r=r when activation=0
    activatedot = command - Muscles.F * active_fibers
    fatiguedot = Muscles.F * active_fibers - Muscles.R * fatigued_fibers

    muscles_states = biorbd.VecBiorbdMuscleState(nlp["nbMuscle"])
    for k in range(nlp["nbMuscle"]):
        muscles_states[k].setActivation(active_fibers[k])
    # todo fix force max

    muscles_tau = nlp["model"].muscularJointTorque(muscles_states, q, qdot).to_mx()
    # todo get muscle forces and multiply them by activate [k] and same as muscularJointTorque
    tau = muscles_tau + residual_tau

    dxdt = MX(nlp["nx"], nlp["ns"])
    for i, f_ext in enumerate(nlp["external_forces"]):
        qddot = biorbd.Model.ForwardDynamics(nlp["model"], q, qdot, tau, f_ext).to_mx()
        dxdt[:, i] = vertcat(qdot, qddot, activatedot, fatiguedot, restingdot)
    return dxdt


def xia_model_configuration(ocp, nlp):
    Problem.configure_q_qdot(nlp, True, False)
    Problem.configure_tau(nlp, False, True)
    Problem.configure_muscles(nlp, False, True)

    x = MX()
    for i in range(nlp["nbMuscle"]):
        x = vertcat(x, MX.sym(f"Muscle_{nlp['muscleNames']}_active_{nlp['phase_idx']}", 1, 1))
    for i in range(nlp["nbMuscle"]):
        x = vertcat(x, MX.sym(f"Muscle_{nlp['muscleNames']}_fatigue_{nlp['phase_idx']}", 1, 1))
    for i in range(nlp["nbMuscle"]):
        x = vertcat(x, MX.sym(f"Muscle_{nlp['muscleNames']}_resting_{nlp['phase_idx']}", 1, 1))

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

    Problem.configure_forward_dyn_func(ocp, nlp, xia_model_dynamic)


def xia_initial_fatigue_at_zero(ocp, nlp, t, x, u, p):
    offset = nlp["nbQ"] + nlp["nbQdot"] + nlp["nbMuscle"]
    val = []
    for k in range(nlp["nbMuscle"]):
        val = vertcat(val, x[0][offset + k])
    return val


def xia_model_fibers(ocp, nlp, t, x, u, p):
    offset = nlp["model"].nbQ() + nlp["model"].nbQdot()
    val = []
    for k in range(nlp["nbMuscle"]):
        val = vertcat(
            val, 1 - x[0][offset + k] - x[0][offset + k + nlp["nbMuscle"]] - x[0][offset + k + 2 * nlp["nbMuscle"]]
        )
    return val


def prepare_ocp(biorbd_model_path="../models/BrasViolon.bioMod"):
    """
    Mix .bioMod and users data to call OptimalControlProgram constructor.
    :param biorbd_model_path: path to the .bioMod file.
    :param show_online_optim: bool which active live plot function.
    :return: OptimalControlProgram object.
    """
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
    number_shooting_points = [10] * nb_phases
    final_time = [1] * nb_phases

    muscle_activated_init, muscle_fatigued_init, muscle_resting_init = 0, 0, 1
    torque_min, torque_max, torque_init = -10, 10, 0
    muscle_states_ratio_min, muscle_states_ratio_max = 0, 1

    # --- String of the violin --- #
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
            Objective.Lagrange.MINIMIZE_TORQUE, controls_idx=[0, 1, 2, 3], weight=10, phase=idx_phase
        )

        # --- Dynamics --- #
        dynamics.add(xia_model_configuration, dynamic_function=xia_model_dynamic)

        # --- Constraints --- #
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
        constraints.add(
            Constraint.ALIGN_SEGMENT_WITH_CUSTOM_RT,
            instant=Instant.ALL,
            min_bound=-0.00001,
            max_bound=0.00001,
            segment_idx=Bow.segment_idx,
            rt_idx=violon_string.rt_on_string,
            phase=idx_phase,
        )
        constraints.add(
            Constraint.ALIGN_MARKER_WITH_SEGMENT_AXIS,
            instant=Instant.ALL,
            min_bound=-0.00001,
            max_bound=0.00001,
            marker_idx=violon_string.bridge_marker,
            segment_idx=Bow.segment_idx,
            axis=(Axe.Y),
            phase=idx_phase,
        )
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

        x_bounds[idx_phase].min[biorbd_model[0].nbQ() :, [0, 2]] = 0
        x_bounds[idx_phase].max[biorbd_model[0].nbQ() :, [0, 2]] = 0
        # todo enlever vitesse nulle

        muscle_states_bounds = Bounds(
            [muscle_states_ratio_min] * biorbd_model[0].nbMuscleTotal() * 3,
            [muscle_states_ratio_max] * biorbd_model[0].nbMuscleTotal() * 3,
        )

        if idx_phase == 0:
            # fatigued_fibers = 0 = actived_fibers and resting_fibers = 1
            muscle_states_bounds.min[36:] = PathCondition(
                np.repeat(np.array((1, 0, 0))[:, np.newaxis].T, biorbd_model[0].nbMuscleTotal(), axis=0),
                interpolation=InterpolationType.EACH_FRAME,
            )

            muscle_states_bounds.max[:36] = PathCondition(
                np.repeat(np.array((0, 1, 1))[:, np.newaxis].T, 2 * biorbd_model[0].nbMuscleTotal(), axis=0),
                interpolation=InterpolationType.EACH_FRAME,
            )
        x_bounds[idx_phase].concatenate(muscle_states_bounds)

        u_bounds.add(
            [
                [torque_min] * biorbd_model[0].nbGeneralizedTorque()
                + [muscle_states_ratio_min] * biorbd_model[0].nbMuscleTotal(),
                [torque_max] * biorbd_model[0].nbGeneralizedTorque()
                + [muscle_states_ratio_max] * biorbd_model[0].nbMuscleTotal(),
            ],
            phase=idx_phase,
        )

        # --- Initial guess --- #
        optimal_initial_values = True
        if optimal_initial_values:
            if idx_phase == 1:
                with open(f"utils/optimal_init_{number_shooting_points[0]}_nodes_first.bio", "rb") as file:
                    dict = pickle.load(file)
            else:
                with open(f"utils/optimal_init_{number_shooting_points[0]}_nodes_others.bio", "rb") as file:
                    dict = pickle.load(file)

            x_init.add(dict["states"], interpolation=InterpolationType.EACH_FRAME, phase=idx_phase)
            u_init.add(dict["controls"], interpolation=InterpolationType.EACH_FRAME, phase=idx_phase)

        else:
            x_init.add(
                violon_string.initial_position()[inital_bow_side.side] + [0] * biorbd_model[0].nbQdot(),
                interpolation=InterpolationType.CONSTANT,
                phase=idx_phase,
            )
            u_init.add(
                [torque_init] * biorbd_model[0].nbGeneralizedTorque()
                + [muscle_activated_init] * biorbd_model[0].nbMuscleTotal(),
                interpolation=InterpolationType.CONSTANT,
                phase=idx_phase,
            )

            muscle_states_init = InitialConditions(
                [muscle_activated_init] * biorbd_model[0].nbMuscleTotal()
                + [muscle_fatigued_init] * biorbd_model[0].nbMuscleTotal()
                + [muscle_resting_init] * biorbd_model[0].nbMuscleTotal(),
                interpolation=InterpolationType.CONSTANT,
            )
            x_init[idx_phase].concatenate(muscle_states_init)

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
            "tol": 1e-4,
            "max_iter": 500000,
            "ipopt.bound_push": 1e-10,
            "ipopt.bound_frac": 1e-10,
            "ipopt.hessian_approximation": "limited-memory",
            "output_file": "output.bot",
            "ipopt.linear_solver": "ma57",  # "mumps",  # "ma57", "ma86"
            # "file_print_level": 5,
        },
    )
    toc = time.time() - tic
    print(f"Time to solve : {toc}sec")

    analyse = Objective.Analyse(ocp, sol_obj)

    t = time.localtime(time.time())
    date = f"{t.tm_year}_{t.tm_mon}_{t.tm_mday}"
    OptimalControlProgram.save(ocp, sol, f"results/{date}_upDown.bo")
    OptimalControlProgram.save_get_data(ocp, sol, f"results/{date}_upDown.bob")
    OptimalControlProgram.save_get_data(ocp, sol, f"results/{date}_upDown_interpolate.bob", interpolate_nb_frames=100)

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.graphs()
