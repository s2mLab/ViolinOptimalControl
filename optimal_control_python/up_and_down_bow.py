import biorbd
import numpy as np

from biorbd_optim import OptimalControlProgram
from biorbd_optim.objective_functions import ObjectiveFunction
from biorbd_optim.constraints import Constraint
from biorbd_optim.problem_type import ProblemType
from biorbd_optim.path_conditions import Bounds, QAndQDotBounds, InitialConditions
from utils import Bow, Violin

def prepare_nlp(biorbd_model_path="../models/BrasViolon.bioMod", show_online_optim=False):
    """
    Mix .bioMod and users data to call OptimalControlProgram constructor.
    :param biorbd_model_path: path to the .bioMod file.
    :param show_online_optim: bool which active live plot function.
    :return: OptimalControlProgram object.
    """
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    muscle_min, muscle_max, muscle_init = 0, 1, 0.5
    torque_min, torque_max, torque_init = -100, 100, 0

    # Problem parameters
    number_shooting_points = 31
    final_time = 0.5

    # Choose the string of the violin
    violon_string = Violin("E")
    inital_bow_side = Bow("frog")

    # Add objective functions
    objective_functions = (
        (ObjectiveFunction.minimize_torque, {"weight": 100}),
        (ObjectiveFunction.minimize_muscle, {"weight": 1}),
    )

    # Dynamics
    problem_type = ProblemType.torque_driven

    # Constraints
    constraints = (
        # (Constraint.Type.MARKERS_TO_PAIR, Constraint.Instant.START, (Bow.frog_marker, violon_string.bridge_marker),),
        # (Constraint.Type.MARKERS_TO_PAIR, Constraint.Instant.MID, (Bow.tip_marker, violon_string.bridge_marker),),
        # (Constraint.Type.MARKERS_TO_PAIR, Constraint.Instant.END, (Bow.frog_marker, violon_string.bridge_marker),),
        (Constraint.Type.ALIGN_WITH_CUSTOM_RT, Constraint.Instant.ALL, (Bow.segment_idx, violon_string.rt_on_string),),
    )

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)

    # Initial guess
    X_init = InitialConditions(violon_string.initial_position()[inital_bow_side.side] + [0] * biorbd_model.nbQdot())

    # Define control path constraint
    U_bounds = Bounds(
        [torque_min] * biorbd_model.nbGeneralizedTorque() + [muscle_min] * biorbd_model.nbMuscleTotal(),
        [torque_max] * biorbd_model.nbGeneralizedTorque() + [muscle_max] * biorbd_model.nbMuscleTotal(),
    )

    U_init = InitialConditions(
        [torque_init] * biorbd_model.nbGeneralizedTorque() + [muscle_init] * biorbd_model.nbMuscleTotal()
    )
    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        problem_type,
        number_shooting_points,
        final_time,
        objective_functions,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        constraints,
        show_online_optim=show_online_optim,
    )


if __name__ == "__main__":
    ocp = prepare_nlp(show_online_optim=True)

    # --- Solve the program --- #
    sol = ocp.solve()

    x, _, _ = ProblemType.get_data_from_V(ocp, sol["x"])
    x = ocp.nlp[0]["dof_mapping"].expand(x)

    np.save("up_and_down", x.T)

    try:
        from BiorbdViz import BiorbdViz

        b = BiorbdViz(loaded_model=ocp.nlp[0]["model"], show_meshes=False)
        b.load_movement(x.T)
        b.exec()
    except ModuleNotFoundError:
        print("Install BiorbdViz if you want to have a live view of the optimization")
        from matplotlib import pyplot as plt

        plt.show()
