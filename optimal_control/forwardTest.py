import biorbd
from BiorbdViz import BiorbdViz
from scipy import integrate
import numpy as np

import analyses.utils as utils

fun_dyn = utils.dynamics_from_muscles_and_torques_and_contact

biorbd_model = biorbd.s2mMusculoSkeletalModel(f"../models/testloop.bioMod")

# q_init = np.array([1.000e-01, 1.000e-01, 1.095, 1.571, 1.056, 1.061, -1.726, -3.746e-01, -3.882e-01, -3.993e-01,
#                    -9.2626e-02, -1.327e-01, -1.322e-01, 5.593e-02])
# u = np.array([9.239e-01, 9.969e-01, 1.968e-01, 9.193e-01, 4.829e-02, 3.218e-02, 1.233e-02, 9.999e-03, 9.907e-01,
#               2.808e-02, 3.112e-02, 3.256e-01, 7.611e-01, 1.899e-02, 1.481e-01, 1.147e-01, 3.727e-01, 4.772e-02,
#               5.7938e-03, -3.122e-02, -6.179e-02, -6.635e-02, 2.789e-02, -3.107e-01, 3.081e-01])

q_init = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
u=np.array([0, 0, 0, 0, 0, 0])

integrated_tp = integrate.solve_ivp(fun=lambda t, y: fun_dyn(t, y, biorbd_model, u),
                                    t_span=(0, 5), y0=q_init, method='RK45', atol=1e-8, rtol=1e-6)

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
