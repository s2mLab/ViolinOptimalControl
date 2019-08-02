import matplotlib.pyplot as plt
import biorbd
import time

import analyses.utils as utils


# Options
model_name = "BrasViolon"
output_files = "Av2Phases"
fun_dyn = utils.dynamics_from_muscles_and_torques  # _and_contact
nb_nodes = 30
nb_phases = 2
nb_frame_inter = 500

# Load the model
m = biorbd.s2mMusculoSkeletalModel(f"../models/{model_name}.bioMod")
if fun_dyn == utils.dynamics_from_muscles:
    nb_controls = m.nbMuscleTotal()
elif fun_dyn == utils.dynamics_from_joint_torque:
    nb_controls = m.nbTau()
elif fun_dyn == utils.dynamics_from_muscles_and_torques \
        or fun_dyn == utils.dynamics_from_muscles_and_torques_and_contact:
    nb_controls = m.nbMuscleTotal()+m.nbTau()
elif fun_dyn == utils.dynamics_from_accelerations:
    nb_controls = m.nbQ()
else:
    raise NotImplementedError("Dynamic not implemented yet")

# Read values
t, all_q, all_qdot = utils.read_acado_output_states(f"../optimal_control/Results/States{output_files}.txt", m, nb_nodes,
                                                    nb_phases)
all_u = utils.read_acado_output_controls(f"../optimal_control/Results/Controls{output_files}.txt", nb_nodes, nb_phases,
                                         nb_controls)
t_final = utils.organize_time(f"../optimal_control/Results/Parameters{output_files}.txt", t, nb_phases, nb_nodes, parameter=False)

n_run = 20  # Compare time
# Integrate rk45
t_rk45 = time.perf_counter()
for i in range(n_run):
    t_integrate_rk45, q_integrate_rk45 = utils.integrate_states_from_controls(m, t_final, all_q, all_qdot, all_u, fun_dyn,
                                                                          verbose=False, use_previous_as_init=False)
t_rk45 = time.perf_counter() - t_rk45

# Interpolate
t_interp_rk45, q_interp_rk45 = utils.interpolate_integration(nb_frames=nb_frame_inter, t_int=t_integrate_rk45, y_int=q_integrate_rk45)

# Integrate rk4
t_rk4 = time.perf_counter()
t_integrate_rk4, q_integrate_rk4 = utils.integrate_states_from_controls(m, t_final, all_q, all_qdot, all_u, fun_dyn,
                                                                    verbose=False, use_previous_as_init=False, algo='rk4')
t_rk4 = time.perf_counter() - t_rk4
# Interpolate
t_interp_rk4, q_interp_rk4 = utils.interpolate_integration(nb_frames=nb_frame_inter, t_int=t_integrate_rk4, y_int=q_integrate_rk4)

print(t_rk45)
print(t_rk4)
# tata = q_interp_rk4 - q_interp_rk45
# tata[abs(tata) < 1e-10] = 0
# for i in range(len(q_integrate_rk45)):
#     plt.plot(t_integrate_rk45, q_integrate_rk45[i])
#     plt.plot(t_integrate_rk4, q_integrate_rk4[i], '-.')
#
#     # plt.plot(t_interp_rk4, tata[:, i])
#     # plt.plot(t_interp_rk45, q_interp_rk45[:, i])
#
#     # plt.plot(t_interp_rk4, q_interp_rk4[:, i], '-.')
#
# plt.show()
