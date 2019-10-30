import numpy as np
import matplotlib.pyplot as plt
import biorbd
from BiorbdViz import BiorbdViz

import analyses.utils as utils


# Options
model_name = "BrasViolon"
output_files = "AvNPhases"
fun_dyn = utils.dynamics_from_muscles_and_torques
runge_kutta_algo = 'rk45'
nb_intervals = 30
nb_phases = 2
nb_frame_inter = 500
muscle_plot_mapping = \
    [[14, 7, 0, 0, 0, 0],  # Trapeze1
     [15, 7, 1, 0, 0, 0],  # Trapeze2
     [16, 8, 0, 0, 0, 0],  # Trapeze3
     [17, 8, 1, 0, 0, 0],  # Trapeze4
     [10, 5, 2, 1, 0, 1],  # SupraSpin
     [8,  5, 0, 1, 0, 1],  # InfraSpin
     [11, 5, 3, 1, 0, 1],  # SubScap
     [6,  4, 0, 0, 1, 2],  # Pectoral1
     [0,  0, 0, 0, 1, 2],  # Pectoral2
     [1,  0, 1, 0, 1, 2],  # Pectoral3
     [7,  4, 1, 1, 1, 3],  # Deltoid1
     [9,  5, 1, 1, 1, 3],  # Deltoid2
     [2,  1, 0, 1, 1, 3],  # Deltoid3
     [12, 6, 0, 2, 0, 4],  # BicepsLong
     [13, 6, 1, 2, 0, 4],  # BicepsShort
     [3,  2, 0, 2, 1, 5],  # TricepsLong
     [5,  3, 1, 2, 1, 5],  # TricepsMed
     [4,  3, 0, 2, 1, 5],  # TricepsLat
     ]
muscle_plot_names = ["Trapèzes", "Coiffe des rotateurs", "Pectoraux", "Deltoïdes", "Biceps", "Triceps"]

# Load the model
m = biorbd.Model(f"../models/{model_name}.bioMod")
if fun_dyn == utils.dynamics_from_muscles:
    nb_controls = m.nbMuscleTotal()
elif fun_dyn == utils.dynamics_from_joint_torque:
    nb_controls = m.nbGeneralizedTorque()
elif fun_dyn == utils.dynamics_from_muscles_and_torques \
        or fun_dyn == utils.dynamics_from_muscles_and_torques_and_contact:
    nb_controls = m.nbMuscleTotal()+m.nbGeneralizedTorque()
elif fun_dyn == utils.dynamics_from_accelerations:
    nb_controls = m.nbQ()
else:
    raise NotImplementedError("Dynamic not implemented yet")

# Read values
t, all_q, all_qdot = utils.read_acado_output_states(f"../optimal_control/Results/States{output_files}.txt", m, nb_intervals,
                                                    nb_phases)
all_u = utils.read_acado_output_controls(f"../optimal_control/Results/Controls{output_files}.txt", nb_intervals, nb_phases,
                                         nb_controls)
t_final = utils.organize_time(f"../optimal_control/Results/Parameters{output_files}.txt", t, nb_phases, nb_intervals, parameter=False)


# Integrate
t_integrate, q_integrate = utils.integrate_states_from_controls(
    m, t_final, all_q, all_qdot, all_u, fun_dyn, verbose=False, use_previous_as_init=False, algo=runge_kutta_algo
)

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
    plt.plot(t_final, all_q[i, :])
    plt.title("Q %i" % i)

    plt.subplot(m.nbQ(), 3, 2+(3*i))
    plt.plot(t_interp, qdot_interp[:, i])
    plt.plot(t_integrate, q_integrate[m.nbQ() + i, :])
    plt.plot(t_final, all_qdot[i, :])
    # plt.plot(t_interp, utils.derive(q_interp, t_interp), '--')
    plt.title("Qdot %i" % i)

for i in range(m.nbGeneralizedTorque()):
    plt.subplot(m.nbGeneralizedTorque(), 3, 3 + (3 * i))
    utils.plot_piecewise_constant(t_final, all_u[m.nbMuscleTotal()+i, :])
    plt.title("Torques %i" % i)
plt.tight_layout(w_pad=-1.0, h_pad=-1.0)

plt.figure("Activations")
cmp = 0
if muscle_plot_mapping is None:
    for i in range(m.nbMuscleGroups()):
        for j in range(m.muscleGroup(i).nbMuscles()):
            plt.subplot(3, 6, cmp + 1)
            utils.plot_piecewise_constant(t_final, all_u[cmp, :])
            plt.title(m.muscleGroup(i).muscle(j).name().getString())
            plt.ylim((0, 1))
            cmp += 1
else:
    nb_row = np.max(muscle_plot_mapping, axis=0)[3] + 1
    nb_col = np.max(muscle_plot_mapping, axis=0)[4] + 1
    for muscle_map in muscle_plot_mapping:
        plt.subplot(nb_row, nb_col, muscle_map[3] * nb_col + muscle_map[4] + 1)
        utils.plot_piecewise_constant(t_final, all_u[muscle_map[0], :])
        # plt.title(m.muscleGroup(map[1]).muscle(map[2]).name().getString())
        plt.title(muscle_plot_names[muscle_map[5]])
        plt.ylim((0, 1))
    plt.tight_layout(w_pad=-1.0, h_pad=-1.0)

# plt.ion()  # Non blocking plt.show
plt.show()

# Animate the model
b = BiorbdViz(loaded_model=m)
b.load_movement(q_interp)
b.exec()
