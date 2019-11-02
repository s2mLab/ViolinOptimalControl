import matplotlib.pyplot as plt
import numpy as np
import biorbd
from BiorbdViz import BiorbdViz

import analyses.utils as utils


# Options
model_name = "BrasViolon"
output_files = "AvNPhases_noActivRule"
fun_dyn = utils.dynamics_from_muscles_and_torques
runge_kutta_algo = 'rk45'
nb_intervals = 30
nb_phases = 2
nb_frame_inter = 500
# Mapping is np.array([[muscle_group, muscle_in_group, muscle_in_control, row_to_plot, col_to_plot], [...]]) or None
muscle_plot_mapping = np.array(
    [[4, 0, 6, 0, 0],  # Pectoral1
     [0, 0, 0, 1, 0],  # Pectoral2
     [0, 1, 1, 2, 0],  # Pectoral3
     [7, 0, 14, 0, 1],  # Trapeze1
     [7, 1, 15, 1, 1],  # Trapeze2
     [8, 0, 16, 2, 1],  # Trapeze3
     [8, 1, 17, 3, 1],  # Trapeze4
     [4, 1, 7, 0, 2],  # Deltoid1
     [5, 1, 9, 1, 2],  # Deltoid2
     [1, 0, 2, 2, 2],  # Deltoid3
     [5, 0, 8, 0, 3],  # InfraSpin
     [5, 2, 10, 1, 3],  # SupraSpin
     [5, 3, 11, 2, 3],  # SubScap
     [6, 0, 12, 0, 4],  # BicepsLong
     [6, 1, 13, 1, 4],  # BicepsShort
     [2, 0, 3, 0, 5],  # TricepsLong
     [3, 1, 5, 1, 5],  # TricepsMed
     [3, 0, 4, 2, 5],  # TricepsLat
    ])

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
t, all_q, all_qdot = utils.read_acado_output_states(f"../optimal_control/Results/States{output_files}.txt", m,
                                                    nb_intervals, nb_phases)
all_u = utils.read_acado_output_controls(f"../optimal_control/Results/Controls{output_files}.txt", nb_intervals,
                                         nb_phases, nb_controls)
all_u = np.append(all_u, all_u[:, -1:], axis=1)  # For facilitate the visualization, add back the last value
t_final = utils.organize_time(f"../optimal_control/Results/Parameters{output_files}.txt", t, nb_phases,
                              nb_intervals, parameter=False)


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
    if fun_dyn == utils.dynamics_from_muscles_and_torques or \
            fun_dyn == utils.dynamics_from_muscles_and_torques_and_contact:
        utils.plot_piecewise_constant(t_final, all_u[m.nbMuscleTotal()+i, :])
    else:
        utils.plot_piecewise_constant(t_final, all_u[i, :])
    plt.title("Torques %i" % i)
plt.tight_layout(w_pad=-1.5, h_pad=-0.5)

if fun_dyn == utils.dynamics_from_muscles_and_torques or \
            fun_dyn == utils.dynamics_from_muscles_and_torques_and_contact:
    plt.figure("Activations")
    if muscle_plot_mapping is None:
        L = []
        for i in range(m.nbMuscleGroups()):
            L.append(m.muscleGroup(i).nbMuscles())
        nb_muscles_max = max(L)
        cmp = 0
        for i in range(m.nbMuscleGroups()):
            for j in range(m.muscleGroup(i).nbMuscles()):
                plt.subplot(nb_muscles_max, m.nbMuscleGroups(), i+1+(m.nbMuscleGroups()*j))
                utils.plot_piecewise_constant(t_final, all_u[cmp, :])
                plt.title(m.muscleGroup(i).muscle(j).name().getString())
                plt.ylim((0, 1))
                cmp += 1
    else:
        nb_rows = max(muscle_plot_mapping[:, 3]) + 1
        nb_cols = max(muscle_plot_mapping[:, 4]) + 1
        for i in range(muscle_plot_mapping.shape[0]):
            plt.subplot(nb_rows, nb_cols, int(nb_cols*muscle_plot_mapping[i, 3] + muscle_plot_mapping[i, 4]) + 1)
            utils.plot_piecewise_constant(t_final, all_u[muscle_plot_mapping[i, 2], :], 'r')
            plt.title(m.muscleGroup(int(muscle_plot_mapping[i, 0])).muscle(int(muscle_plot_mapping[i, 1])).name().getString())
            plt.ylim((0, 1))
    plt.tight_layout(w_pad=-1.5, h_pad=-0.5)

plt.show()

# Animate the model
b = BiorbdViz(loaded_model=m)
b.load_movement(q_interp)
b.exec()

