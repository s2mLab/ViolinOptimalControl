"""
Example script for animating a model
"""
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate

from pyoviz.BiorbdViz import BiorbdViz
import biorbd

import matplotlib.pyplot as plt

# Load the biorbd model

m = biorbd.s2mMusculoSkeletalModel("Modeles/ModeleAv2Muscles.s2mMod")

nFrame = 500            # number of frames
nNoeuds = 30                # number of points
nMuscle =m.nbMuscleTotal()
nPhase = 1
nPoints = (nPhase*nNoeuds)+1
nQtotal = m.nbQ()+m.nbQdot()

##### State from the file

fichierS = open("Resultats/StatesAv2Muscles.txt", "r")

i = 0
t = []                                      # initialization of the time

all_Q = np.ones((m.nbQ(), nPoints))             # initialization of the states nbQ lines and nbP columns
all_Qdot = np.ones((m.nbQdot(), nPoints))

# initialization of the derived states

# Nnoeuds first lines
for l in range(nNoeuds):
    ligne = fichierS.readline()
    lin = ligne.split('\t')                 # separation of the line in element
    lin[:1] = []                            # remove the first element ( [ )
    lin[(nPhase*m.nbQ()) + (nPhase*m.nbQdot()) + 1:] = []     # remove the last ( ] )

    t.append(float(lin[0]))                                                         # complete the time with the first column
    for p in range(nPhase):
        all_Q[:, i+p*nNoeuds] = [float(j) for j in lin[1+p*nQtotal:m.nbQ()+p*nQtotal+1]]                              # complete the states with the nQ next columns
        all_Qdot[:, i+p*nNoeuds] = [float(k) for k in lin[m.nbQ()+1+p*nQtotal:nQtotal*(p+1)+1]]        # complete the states with the nQdot next columns

    i += 1
# Last line
ligne = fichierS.readline()
lin = ligne.split('\t')                 # separation of the line in element
lin[:1] = []                            # remove the first element ( [ )
lin[(nPhase*m.nbQ()) + (nPhase*m.nbQdot()) + 1:] = []     # remove the last ( ] )
t.append(float(lin[0]))
all_Q[:, -1] = [float(j) for j in lin[1+(nPhase-1)*nQtotal:m.nbQ()+(nPhase-1)*nQtotal+1]]                              # complete the states with the nQ next columns
all_Qdot[:, -1] = [float(k) for k in lin[m.nbQ()+1+(nPhase-1)*nQtotal:nQtotal*nPhase+1]]


fichierS.close()

# Complete the time
for p in range(1, nPhase):
    for i in range(nNoeuds):
        t.append(p+t[i+1])

##### Controls from the file

fichierU = open("Resultats/ControlsAv2Muscles.txt", "r")

i = 0
all_U = np.ones((nMuscle, nPoints))          # initialization of the controls

for l in range(nNoeuds):
    ligne = fichierU.readline()
    lin = ligne.split('\t')
    lin[:1] = []
    lin[(nPhase*nMuscle) + 1:] = []

    for p in range(nPhase):
        all_U[:, i+p*nNoeuds] = [float(i) for i in lin[1+p*nMuscle:nMuscle*(p+1)+1]]

    i += 1

ligne = fichierU.readline()
lin = ligne.split('\t')
lin[:1] = []
lin[(2*nMuscle) + 1:] = []

all_U[:, -1] = [float(i) for i in lin[1+(nPhase-1)*nMuscle:nMuscle*nPhase+1]]

fichierU.close()

#### integration

#dynamique avec activation des muscles
def dyn(t_int, X):
    states_actual = biorbd.VecS2mMuscleStateActual(nMuscle)
    for i in range(len(states_actual)):
        states_actual[i] = biorbd.s2mMuscleStateActual(0, u[i])

    m.updateMuscles(m, X[:m.nbQ()], X[m.nbQ():], True)

    Tau = biorbd.s2mMusculoSkeletalModel.muscularJointTorque(m, states_actual, X[:m.nbQ()], X[m.nbQ():])

    QDDot = biorbd.s2mMusculoSkeletalModel.ForwardDynamics(m, X[:m.nbQ()], X[m.nbQ():], Tau).get_array()
    rsh = np.ndarray(m.nbQ() + m.nbQdot())
    for i in range(m.nbQ()):

        rsh[i] = X[m.nbQ()+i]
        rsh[i + m.nbQ()] = QDDot[i]

    return rsh

for interval in range(0, len(t)-1):                 # integration between each point
    print(interval)
    u = all_U[:, interval]
    X = np.concatenate((all_Q[:, interval], all_Qdot[:, interval]))
    print("Solution initiale: ", X)
    print("Control: ", u)
    print("dynamique: ", dyn(0, X))
    print("\n")



