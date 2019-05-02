"""
Example script for animating a model
"""

from pathlib import Path
import copy

import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate

from pyoviz.BiorbdViz import BiorbdViz
import biorbd

import matplotlib.pyplot as plt

# Load the biorbd model

m = biorbd.s2mMusculoSkeletalModel("Modeles/ModeleSansMuscle.s2mMod")

nFrame = 500            # number of frames
nNoeuds = 30                # number of points
nMuscle = m.nbMuscleTotal()
nPhase = 1
nPoints = (nPhase*nNoeuds)+1

##### State from the file

fichierS = open("Resultats/StatesSansMuscle.txt", "r")

i = 0
t = []                                      # initialization of the time
nQtotal = m.nbQ()+m.nbQdot()

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

fichierU = open("Resultats/ControlsSansMuscle.txt", "r")

i = 0
all_U = np.ones((m.nbQdot(), nPoints))          # initialization of the controls

for l in range(nNoeuds):
    ligne = fichierU.readline()
    lin = ligne.split('\t')
    lin[:1] = []
    lin[(2*nMuscle) + 1:] = []

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

#dynamiaue avec activation des muscles
# def dyn(t_int, X):
#     states_actual = biorbd.VecS2mMuscleStateActual(nMuscle)
#     for i in range(len(states_actual)):
#         states_actual[i] = biorbd.s2mMuscleStateActual(0, u[i])
#
#     Tau = biorbd.s2mMusculoSkeletalModel.muscularJointTorque(m, states_actual, X[:m.nbQ()], X[m.nbQ():])
#
#     QDDot = biorbd.s2mMusculoSkeletalModel.ForwardDynamics(m, X[:m.nbQ()], X[m.nbQ():], Tau).get_array()
#     rsh = np.ndarray(m.nbQ() + m.nbQdot())
#     for i in range(m.nbQ()):
#         rsh[i] = X[m.nbQ()+i]
#         rsh[i + m.nbQ()] = QDDot[i]
#
#     return rsh

#dynamique sans muscle
def dyn(t_int, X):

    QDDot = biorbd.s2mMusculoSkeletalModel.ForwardDynamics(m, X[:m.nbQ()], X[m.nbQ():], u).get_array()
    rsh = np.ndarray(m.nbQ() + m.nbQdot())
    for i in range(m.nbQ()):
        rsh[i] = X[m.nbQ()+i]
        rsh[i + m.nbQ()] = QDDot[i]

    return rsh


Y = list()                                  # interpolated states list
for q in range(m.nbQ()):
    T = []                                # time list
    I = []
    X = np.concatenate((all_Q[:, 0], all_Qdot[:, 0]))                                    # integrated states list
    for interval in range(len(t)-1):                                                    # integration between each point
        u = all_U[:, interval]
        LI = integrate.solve_ivp(dyn, (t[interval], t[interval+1]), X)
        X = LI.y[:, -1]

        for i in range(len(LI.t)-1):
            I.append(LI.y[q, :].tolist()[i])
            T.append(LI.t.tolist()[i])

    #### Interpolation

    tck = interpolate.splrep(T, I, s=0)
    time = np.linspace(0, T[len(T)-1], nFrame)
    Y.append(interpolate.splev(time, tck, der=0))               # liste des positions à chaque temp: nQ listes de nFrames elements

###### visualisation

plt.figure(1)
for i in range(m.nbQ()):
    plt.subplot(m.nbQ(), 2, 1+(2*i))
    plt.plot(all_Q[i])
    plt.title("états %i" %i)

    plt.subplot(m.nbQ(), 2, 2+(2*i))
    plt.plot(all_Qdot[i])
    plt.title("états dérivés %i" %i)

plt.figure(2)
for j in range(nMuscle):
    plt.subplot(nMuscle, 1, i+1)
    plt.plot(all_U[i])
    plt.title("contrôles %i" %j)

plt.show()

# Animation

b = BiorbdViz(loaded_model=m, show_muscles=False)

Q = np.ndarray((nFrame, m.nbQ()))
for i in range(m.nbQ()):
    Q[:, i] = Y[i]

i = 0
while b.vtk_window.is_active:
    b.set_q(Q[i, :])
    i = (i+1) % nFrame

