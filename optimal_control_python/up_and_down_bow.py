import biorbd
import numpy as np

from biorbd_optim import OptimalControlProgram
from biorbd_optim.objective_functions import ObjectiveFunction
from biorbd_optim.constraints import Constraint
from biorbd_optim.problem_type import ProblemType
from biorbd_optim.path_conditions import Bounds, QAndQDotBounds, InitialConditions


class Bow:
    """
    Contains references from useful markers.
    """

    segment_idx = 8
    frog_marker = 16
    tip_marker = 18

    def __init__(self, bow_side):
        """
        Contains the side of the bow.
        """
        if bow_side not in ["frog", "tip"]:
            raise RuntimeError(bow_side + " is not a valid side of bow, it must be frog or tip.")
        self.bow_side = bow_side

    @property
    def side(self):
        return self.bow_side


class Violin:
    """
    Contains initial values and references from useful markers and segments.
    """

    segment_idx = 16

    def __init__(self, string):
        """
        Contains some references and values specific to the string.
        :param string: violin string letter
        :param bow_side: side of the bow, "frog" or "tip".
        """
        if string not in ["E", "A", "D", "G"]:
            raise RuntimeError(string + " is not a valid string, it must be E, A, D or G. Do you know violin ?")
        self.string = string

    def initial_position(self):
        """
        :return: List of initial positions according to the string and the side of the bow.
        """
        return {
            "E": {
                "frog": [-0.2908, -0.4622, 0.6952, 1.1347, 1.4096, -0.1030, 0.1516, -0.2379, -0.2633],
                "tip": [0.0876, -0.5649, 0.6498, 1.0598, -0.1866, 0.2434, 0.1582, 0.2087, 0.7162],
            },
            "A": {
                "frog": [-0.1569, -0.5216, 0.5900, 1.1063, 1.4728, 0.0393, 0.3143, -0.3959, -0.4446],
                "tip": [0.0305, -0.6904, 0.3695, 0.8809, 0.1557, 0.2997, 0.2071, 0.1471, 0.5546],
            },
            "D": {
                "frog": [-0.1259, -0.4520, 0.5822, 1.1106, 1.4595, 0.1194, 0.5033, -0.4040, -0.4567],
                "tip": [0.0378, -0.7034, 0.2345, 0.947, 0.1111, 0.4134, 0.2470, 0.2606, 0.4842],
            },
            "G": {
                "frog": [-0.2697, -0.3733, 0.5529, 1.1676, 1.5453, 0.0877, 0.6603, -0.5842, -0.6424],
                "tip": [-0.0182, -1.3112, 0.1928, 0.6092, 0.7065, -0.0755, 0.1720, 0.1136, 0.2626],
            },
        }[self.string]

    @property
    def bridge_marker(self):
        """
        :return: Marker number on the bridge, associate to the string.
        """
        return {"E": 34, "A": 36, "D": 38, "G": 40,}[self.string]

    @property
    def neck_marker(self):
        """
        :return: Marker number on the neck, associate to the string.
        """
        return {"E": 35, "A": 37, "D": 39, "G": 41,}[self.string]

    @property
    def rt_on_string(self):
        """
        :return: RT number according to the string.
        """
        return {"E": 3, "A": 2, "D": 1, "G": 0,}[self.string]


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
