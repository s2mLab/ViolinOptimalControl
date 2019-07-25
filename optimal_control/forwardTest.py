import matplotlib.pyplot as plt
import biorbd
from BiorbdViz import BiorbdViz
from scipy import integrate, interpolate
import numpy as np

import analyses.utils as utils


def dyn_fun(t, x, model, u):
    q = x[:model.nbQ()]
    qdot = x[model.nbQ():]
    tau = u
    cs = model.getConstraints(model)
    qddot = model.ForwardDynamicsConstraintsDirect(q, qdot, tau, cs).get_array()
    return np.concatenate((qdot, qddot))


biorbd_model = biorbd.s2mMusculoSkeletalModel(f"../models/BrasViolon.bioMod")
q_init = np.array([0.1000001, 0.1000001, 1.0946872, 1.5707965, 1.0564277, 1.0607269, -1.725867,
                   0, 0, 0, 0, 0, 0, 0])
u = np.array([0, 0, 0, 0, -1, 0, 0])

integrated_tp = integrate.solve_ivp(fun = lambda t, y: dyn_fun(t, y, biorbd_model, u),
                                    t_span = (0, 5), y0 = q_init,
                                    method = 'RK45', atol = 1e-8, rtol = 1e-6)

t_interp, q_interp = utils.interpolate_integration(nb_frames=5000, t_int=integrated_tp.t, y_int=integrated_tp.y)

bioviz = BiorbdViz(loaded_model=biorbd_model)
bioviz.load_movement(q_interp)
bioviz.exec()
