import numpy as np
import biorbd
from optimal_control_python import up_and_down_bow

from utils import Muscles

biorbd_model_path = "../models/BrasViolon.bioMod"
biorbd_model = biorbd.Model(biorbd_model_path)
muscle_activated_init, muscle_fatigued_init, muscle_resting_init = 0, 0, 1
torque_min, torque_max, torque_init = -10, 10, 0
muscle_states_ratio_min, muscle_states_ratio_max = 0, 1
number_shooting_points = 30
final_time = 0.5

nbq = biorbd_model.nbQ()
nbqdot = biorbd_model.nbQdot()
nx = nbq + nbqdot

nbtau = biorbd_model.nbGeneralizedTorque()
nbm = biorbd_model.nbMuscles()
nu = nbtau + nbm

q = np.array([0] * nbq)
qdot = np.array([0] * nbqdot)

active_fibers = np.array([0] * nbm)
fatigued_fibers = np.array([0] * nbm)
resting_fibers = np.array([1] * nbm)

residual_tau = np.array([0] * nbtau)
activation = np.array([0.01] * nbm)
command = np.array([0] * nbm)

idx = 0
for i in range(biorbd_model.nbMuscleGroups()):
    for k in range(biorbd_model.muscleGroup(i).nbMuscles()):

        develop_factor = 10
        # (biorbd_model.muscleGroup(i).muscle(k).characteristics().fatigueParameters().developFactor().to_mx())
        recovery_factor = 10
        # (biorbd_model.muscleGroup(i).muscle(k).characteristics().fatigueParameters().recoveryFactor().to_mx())

        if active_fibers[idx] < activation[idx]:
            if resting_fibers[idx] > activation[idx] - active_fibers[idx]:
                command[idx] = develop_factor * (activation[idx] - active_fibers[idx])
            else:
                command[idx] = develop_factor * resting_fibers[idx]
        else:
            command[idx] = recovery_factor * (active_fibers[idx] - activation[idx])

        idx += 1

restingdot = -command + Muscles.R * fatigued_fibers
activatedot = command - Muscles.F * active_fibers
fatiguedot = Muscles.F * active_fibers - Muscles.R * fatigued_fibers

muscles_states = biorbd.VecBiorbdMuscleState(nbm)
for k in range(nbm):
    muscles_states[k].setActivation(active_fibers[k])

muscles_tau = biorbd_model.muscularJointTorque(muscles_states, q, qdot).to_mx()
tau = muscles_tau + residual_tau

qddot = biorbd.Model.ForwardDynamics(biorbd_model, q, qdot, tau).to_mx()

# qdot, qddot, activatedot, fatiguedot, restingdot
