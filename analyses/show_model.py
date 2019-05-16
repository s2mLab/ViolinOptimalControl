import matplotlib.pyplot as plt
import biorbd
from pyoviz.BiorbdViz import BiorbdViz

import utils

# Options
model_name = "SansMuscle"
fun_dyn = utils.dynamics_from_joint_torque
nb_nodes = 30
nb_phases = 1
nb_frame_inter = 500

# Load the biorbd model
m = biorbd.s2mMusculoSkeletalModel(f"../optimal_control/Modeles/Modele{model_name}.bioMod")
if fun_dyn == utils.dynamics_from_muscles:
    nb_controls = m.nbMuscleTotal()
elif fun_dyn == utils.dynamics_from_joint_torque:
    nb_controls = m.nbTau()
else:
    raise NotImplementedError("Dynamic not implemented yet")

# Read values
t, all_q, all_qdot = utils.read_acado_output_states(f"../optimal_control/Results/States{model_name}.txt", m, nb_nodes, nb_phases)
all_u = utils.read_acado_output_controls(f"../optimal_control/Results/Controls{model_name}.txt", nb_nodes, nb_phases, nb_controls)

# Integrate
t_integrate, q_integrate = utils.integrate_states_from_controls(m, t, all_q, all_qdot, all_u, fun_dyn,
                                                                verbose=False, use_previous_as_init=True)

# Interpolate
t_interp, q_interp = utils.interpolate_integration(nb_frames=nb_frame_inter, t_int=t_integrate, y_int=q_integrate)
qdot_interp = q_interp[:, m.nbQ():]
q_interp = q_interp[:, :m.nbQ()]

# Show data
plt.figure("States")
for i in range(m.nbQ()):
    plt.subplot(m.nbQ(), 3, 1+(3*i))
    plt.plot(t_interp, q_interp[:, i])
    plt.title("Q %i" %i)

    plt.subplot(m.nbQ(), 3, 2+(3*i))
    plt.plot(t_interp, qdot_interp[:, i])
    plt.title("Qdot %i" %i)

    plt.subplot(m.nbQ(), 3, 3+(3*i))
    utils.plot_piecewise_constant(t, all_u[i, :])
    plt.title("Control %i" %i)
plt.ion()  # Non blocking plt.show
plt.show()

# Animate the model
b = BiorbdViz(loaded_model=m, show_muscles=False)
frame = 0
while b.vtk_window.is_active:
    b.set_q(q_interp[frame, :])
    frame = (frame+1) % nb_frame_inter

