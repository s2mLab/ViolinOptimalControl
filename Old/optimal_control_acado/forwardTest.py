import biorbd_casadi as biorbd
from bioviz import Viz
from scipy import integrate
import numpy as np

import analyses.utils as utils

fun_dyn = utils.dynamics_from_muscles_and_torques_and_contact

biorbd_model = biorbd.Model(f"../models/testloop.bioMod")

# q_init = np.array([1.000e-01, 1.000e-01, 1.095, 1.571, 1.056, 1.061, -1.726, -3.746e-01, -3.882e-01, -3.993e-01,
#                    -9.2626e-02, -1.327e-01, -1.322e-01, 5.593e-02])
# u = np.array([9.239e-01, 9.969e-01, 1.968e-01, 9.193e-01, 4.829e-02, 3.218e-02, 1.233e-02, 9.999e-03, 9.907e-01,
#               2.808e-02, 3.112e-02, 3.256e-01, 7.611e-01, 1.899e-02, 1.481e-01, 1.147e-01, 3.727e-01, 4.772e-02,
#               5.7938e-03, -3.122e-02, -6.179e-02, -6.635e-02, 2.789e-02, -3.107e-01, 3.081e-01])

q_init = np.concatenate((np.zeros(biorbd_model.nbQ()), np.zeros(biorbd_model.nbQ())))
u = np.zeros(biorbd_model.nbGeneralizedTorque())

integrated_tp = integrate.solve_ivp(fun=lambda t, y: fun_dyn(t, y, biorbd_model, u),
                                    t_span=(0, 1), y0=q_init, method='RK45', atol=1e-8, rtol=1e-6)

t_interp, q_interp = utils.interpolate_integration(nb_frames=1000, t_int=integrated_tp.t, y_int=integrated_tp.y)



integrated_tp2 = integrate.solve_ivp(fun=lambda t, y: utils.dynamics_from_muscles_and_torques(t, y, biorbd_model, u),
                                    t_span=(0, 1), y0=q_init, method='RK45', atol=1e-8, rtol=1e-6)

t_interp2, q_interp2 = utils.interpolate_integration(nb_frames=1000, t_int=integrated_tp2.t, y_int=integrated_tp2.y)

from matplotlib import pyplot as plt
plt.plot(t_interp, q_interp)
plt.plot(t_interp2, q_interp2, '-.')
plt.show()

bioviz = Viz(loaded_model=biorbd_model)
bioviz2 = Viz(loaded_model=biorbd_model)
bioviz.load_movement(q_interp)
bioviz2.load_movement(q_interp2)

bioviz.is_executing = True
bioviz2.is_executing = True
while bioviz.vtk_window.is_active and bioviz2.vtk_window.is_active:
    if bioviz.show_options and bioviz.is_animating:
        bioviz.movement_slider[0].setValue(
            (bioviz.movement_slider[0].value() + 1) % bioviz.movement_slider[0].maximum()
        )
    if bioviz2.show_options and bioviz2.is_animating:
        bioviz2.movement_slider[0].setValue(
            (bioviz2.movement_slider[0].value() + 1) % bioviz2.movement_slider[0].maximum()
        )
    bioviz.refresh_window()
    bioviz2.refresh_window()
