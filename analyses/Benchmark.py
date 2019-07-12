import matplotlib.pyplot as plt
import biorbd
import numpy as np
import utils

mSimple = biorbd.s2mMusculoSkeletalModel(f"../models/BrasSimple.bioMod")
mThelen = biorbd.s2mMusculoSkeletalModel(f"../models/BrasThelen.bioMod")

t_int = 1
u = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
              0.001, 0.001, 0.001, 0.001, 0.001])


states = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

rshSimple = utils.dynamics_from_muscles_and_torques(t_int, states, mSimple, u)
rshThelen = utils.dynamics_from_muscles_and_torques(t_int, states, mThelen, u)

print(rshSimple)
print(rshThelen)

plt.figure("Dynamics Simple/Thelen")
plt.plot(rshSimple[mSimple.nbQ():])
plt.plot(rshThelen[mThelen.nbQ():])
plt.show()
