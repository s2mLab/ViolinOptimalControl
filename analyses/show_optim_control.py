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

# for i in range(nb_controls):
#     plt.subplot(nb_controls, 3, 3 + (3 * i))
#     utils.plot_piecewise_constant(t_final, all_u[i, :])
#     plt.title("Acceleration %i" % i)

for i in range(m.nbGeneralizedTorque()):
    plt.subplot(m.nbGeneralizedTorque(), 3, 3 + (3 * i))
    if fun_dyn == utils.dynamics_from_muscles_and_torques or \
            fun_dyn == utils.dynamics_from_muscles_and_torques_and_contact:
        utils.plot_piecewise_constant(t_final, all_u[m.nbMuscleTotal()+i, :])
    else:
        utils.plot_piecewise_constant(t_final, all_u[i, :])
    utils.plot_piecewise_constant(t_final, all_u[m.nbMuscleTotal()+i, :])
    plt.title("Torques %i" % i)
plt.tight_layout(w_pad=-1.5, h_pad=-0.5)

# L = []
# for i in range(m.nbMuscleGroups()):
#     L.append(m.muscleGroup(i).nbMuscles())
# nb_muscles_max = max(L)
plt.figure("Activations")
cmp = 0
for i in range(m.nbMuscleGroups()):
    for j in range(m.muscleGroup(i).nbMuscles()):
        #plt.subplot(nb_muscles_max, m.nbMuscleGroups(), i+1+(m.nbMuscleGroups()*j))
        plt.subplot(3, 6, cmp+1)
        utils.plot_piecewise_constant(t_final, all_u[cmp, :])
        plt.title(biorbd.HillType.getRef(m.muscleGroup(i).muscle(j)).name().getString())
        plt.ylim((0, 1))
        cmp += 1

# plt.ion()  # Non blocking plt.show
plt.show()

# Animate the model
b = BiorbdViz(loaded_model=m)
b.load_movement(q_interp)
b.exec()
