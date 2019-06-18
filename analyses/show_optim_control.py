import matplotlib.pyplot as plt
import biorbd
from pyoviz.BiorbdViz import BiorbdViz

import utils

# Options
model_name = "Bras"
output_files = "Av2Muscles"
fun_dyn = utils.dynamics_from_muscles_and_torques
nb_nodes = 30
nb_phases = 1
nb_frame_inter = 500

# Load the model
m = biorbd.s2mMusculoSkeletalModel(f"../models/{model_name}.bioMod")
if fun_dyn == utils.dynamics_from_muscles:
    nb_controls = m.nbMuscleTotal()
elif fun_dyn == utils.dynamics_from_joint_torque:
    nb_controls = m.nbTau()
elif fun_dyn == utils.dynamics_from_muscles_and_torques:
    nb_controls = m.nbMuscleTotal()+m.nbTau()
else:
    raise NotImplementedError("Dynamic not implemented yet")

# Read values
t, all_q, all_qdot = utils.read_acado_output_states(f"../optimal_control/Results/States{output_files}.txt", m, nb_nodes,
                                                    nb_phases)
all_u = utils.read_acado_output_controls(f"../optimal_control/Results/Controls{output_files}.txt", nb_nodes, nb_phases,
                                         nb_controls)
t_final = utils.organize_time(f"../optimal_control/Results/Parameters{output_files}.txt", t, nb_phases, nb_nodes)


# Integrate
t_integrate, q_integrate = utils.integrate_states_from_controls(m, t_final, all_q, all_qdot, all_u, fun_dyn,
                                                                verbose=False, use_previous_as_init=False)

# Interpolate
t_interp, q_interp = utils.interpolate_integration(nb_frames=nb_frame_inter, t_int=t_integrate, y_int=q_integrate)
qdot_interp = q_interp[:, m.nbQ():]
q_interp = q_interp[:, :m.nbQ()]

# Show data
plt.figure("States and torques res")
for i in range(m.nbQ()):
    plt.subplot(m.nbQ(), 3, 1+(3*i))
    plt.plot(t_interp, q_interp[:, i])
    plt.plot(t_integrate, q_integrate[i, :])
    plt.title("Q %i" % i)

    plt.subplot(m.nbQ(), 3, 2+(3*i))
    plt.plot(t_interp, qdot_interp[:, i])
    plt.plot(t_integrate, q_integrate[m.nbQ() + i, :])
    # plt.plot(t_interp, utils.derive(q_interp, t_interp), '--')
    plt.title("Qdot %i" % i)

for i in range(m.nbTau()):
    plt.subplot(m.nbTau(), 3, 3 + (3 * i))
    utils.plot_piecewise_constant(t_final, all_u[m.nbMuscleTotal()+i, :])
    plt.title("Torques %i" % i)

L = []
for i in range(m.nbMuscleGroups()):
    L.append(m.muscleGroup(i).nbMuscles())
nb_muscles_max = max(L)
plt.figure("Activations")
cmp = 0
for i in range(m.nbMuscleGroups()):
    for j in range(m.muscleGroup(i).nbMuscles()):
        plt.subplot(nb_muscles_max, m.nbMuscleGroups(), i+1+(m.nbMuscleGroups()*j))
        utils.plot_piecewise_constant(t_final, all_u[cmp, :])
        plt.title(biorbd.s2mMuscleHillType.getRef(m.muscleGroup(i).muscle(j)).name())
        plt.ylim((0, 1))
        cmp += 1

# plt.ion()  # Non blocking plt.show
plt.show()

# Animate the model
b = BiorbdViz(loaded_model=m)
frame = 0
while b.vtk_window.is_active:
    b.set_q(q_interp[frame, :])
    frame = (frame+1) % nb_frame_inter
