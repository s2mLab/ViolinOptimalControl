import biorbd
import numpy as np
import utils

m = biorbd.s2mMusculoSkeletalModel(f"../models/Bras.bioMod")

t_int = 1
u = np.array([-0.869068446341981, 0.950642904867541, 0.55855735824, 2.9335678965637, 3.59209049476152])


states = np.array([0.303003976195457, -0.392644704393514, 0.0653044939522075, 0.230210688801268, 2.02388706173711,
                   2.69264643228778, -4.58811801120136, 10.8369256610561, 9.55908506355071, 27.1153950595677])




np.set_printoptions(precision=15)
print(utils.dynamics_from_joint_torque(t_int, states, m, u))

