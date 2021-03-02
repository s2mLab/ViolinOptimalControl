from up_and_down_bow import prepare_ocp
from utils import Bow, Violin, Muscles

from biorbd_optim import InitialConditions, Simulate, ShowResult, InterpolationType, OptimalControlProgram

ocp = prepare_ocp()

muscle_activated_init, muscle_fatigued_init, muscle_resting_init = 0, 0, 1
torque_init = 0
command = 0.3
violon_string = Violin("G")
inital_bow_side = Bow("frog")

X = InitialConditions(
    violon_string.q()[inital_bow_side.side] + [0] * ocp.nlp[0]["nbQdot"], InterpolationType.CONSTANT,
)
U = InitialConditions(
    [torque_init] * ocp.nlp[0]["nbTau"] + [command] * ocp.nlp[0]["model"].nbMuscles(), InterpolationType.CONSTANT,
)

muscle_states_init = InitialConditions(
    [muscle_activated_init] * ocp.nlp[0]["model"].nbMuscles()
    + [muscle_fatigued_init] * ocp.nlp[0]["model"].nbMuscles()
    + [muscle_resting_init] * ocp.nlp[0]["model"].nbMuscles(),
    InterpolationType.CONSTANT,
)
X.concatenate(muscle_states_init)

# --- Simulate --- #
sol_simulate = Simulate.from_controls_and_initial_states(ocp, X, U, single_shoot=True)

# --- Save for biorbdviz --- #
OptimalControlProgram.save_get_data(ocp, sol_simulate, f"results/simulate.bob", interpolate_nb_frames=100)

# --- Graph --- #
result_single = ShowResult(ocp, sol_simulate)
result_single.graphs()
# todo multiphase
print("ok")

# result_single.animate()
