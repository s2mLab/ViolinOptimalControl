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
            "E_frog": [
                -0.32244523,
                -0.45567388,
                0.69477217,
                1.14551489,
                1.40942749,
                -0.10300415,
                0.14266607,
                -0.23330034,
                -0.25421303,
            ],
            "E_tip": [
                0.08773515,
                -0.56553214,
                0.64993785,
                1.0591878,
                -0.18567152,
                0.24296588,
                0.15829188,
                0.21021353,
                0.71442364,
            ],
            "A_frog": [
                -0.15691089,
                -0.52162508,
                0.59001626,
                1.10637291,
                1.47285539,
                0.03932967,
                0.31431404,
                -0.39598565,
                -0.44465406,
            ],
            "A_tip": [
                0.03051712,
                -0.69048243,
                0.36951694,
                0.88094724,
                0.15574657,
                0.29978535,
                0.20718762,
                0.14710871,
                0.55469901,
            ],
            "D_frog": [
                -0.12599098,
                -0.45205593,
                0.5822579,
                1.11068584,
                1.45957662,
                0.11948427,
                0.50336002,
                -0.40407875,
                -0.456703117,
            ],
            "D_tip": [
                0.03788864,
                -0.70345511,
                0.23451146,
                0.9479002,
                0.11111476,
                0.41349365,
                0.24701369,
                0.2606112,
                0.48426223,
            ],
            "G_frog": [
                -0.26963739,
                -0.37332812,
                0.55297438,
                1.16757958,
                1.5453081,
                0.08781926,
                0.66038247,
                -0.58420915,
                -0.6424003,
            ],
            "G_tip": [
                -0.01828739,
                -1.31128207,
                0.19282409,
                0.60925735,
                0.70654631,
                -0.07557834,
                0.17204947,
                0.11369929,
                0.26267182,
            ],
        }[self.string + "_" + self.bow_side]

    @property
    def bridge_marker(self):
        """
        :return: Marker number on the bridge, associate to the string.
        """
        return {"E": 34, "A": 36, "D": 38, "G": 40,}[self.string]

    def neck_marker(self):
        """
        :return: Marker number on the neck, associate to the string.
        """
        return {"E": 35, "A": 37, "D": 39, "G": 41,}[self.string]

    def rt(self):
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
    string_name = "E"
    inital_bow_side = "frog"
    violon_string = ViolinString(string_name, inital_bow_side)

    # Add objective functions
    objective_functions = (
        (ObjectiveFunction.minimize_torque, {"weight": 100}),
        (ObjectiveFunction.minimize_muscle, {"weight": 1}),
    )

    # Dynamics
    problem_type = ProblemType.torque_driven

    # Constraints
    constraints = (
        (
            Constraint.Type.MARKERS_TO_PAIR,
            Constraint.Instant.START,
            (ViolinString.marker_frog, violon_string.bridge_marker),
        ),
        (
            Constraint.Type.MARKERS_TO_PAIR,
            Constraint.Instant.MID,
            (ViolinString.marker_tip, violon_string.bridge_marker),
        ),
        (
            Constraint.Type.MARKERS_TO_PAIR,
            Constraint.Instant.END,
            (ViolinString.marker_frog, violon_string.bridge_marker),
        ),
    )

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)

    # Initial guess
    X_init = InitialConditions(violon_string.initial_position() + [0] * biorbd_model.nbQdot())

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
