from scipy import integrate, interpolate
from matplotlib import pyplot as plt
import numpy as np
import utils

# Options
nb_nodes = 30
nb_phases = 4
nb_frame_inter = 500
nb_dim = 3
output_files = "Eocar"


# read states
nb_points = (nb_phases * nb_nodes) + 1

i = 0
t = np.ndarray(nb_points)  # initialization of the time

# initialization of the derived states
all_q = np.ndarray((nb_dim, nb_points))
all_qdot = np.ndarray((nb_dim, nb_points))
with open(f"../optimal_control/Results/States{output_files}.txt", "r") as data:
    # Nodes first lines
    for l in range(nb_nodes):
        line = data.readline()
        lin = line.split('\t')  # separation of the line in element
        lin[:1] = []  # remove the first element ( [ )
        lin[(nb_phases * nb_dim) + (nb_phases * nb_dim) + 1:] = []  # remove the last ]

        t[i] = float(lin[0])  # complete the time with the first column
        for p in range(nb_phases):
            all_q[:, i + p * nb_nodes] = [float(j) for j in lin[
                                                            1 + p * 2*nb_dim:nb_dim + p * 2*nb_dim + 1]]  # complete the states with the nQ next columns
            all_qdot[:, i + p * nb_nodes] = [float(k) for k in
                                             lin[nb_dim + 1 + p * 2*nb_dim:2*nb_dim * (p + 1) + 1]]
        i += 1
    # Last line
    line = data.readline()
    lin = line.split('\t')  # separation of the line in element
    lin[:1] = []  # remove the first element ( [ )
    lin[(nb_phases * nb_dim) + (nb_phases * nb_dim) + 1:] = []  # remove the last ( ] )
    t[i] = float(lin[0])
    all_q[:, -1] = [float(j) for j in lin[1 + (nb_phases - 1) * 2*nb_dim:nb_dim + (nb_phases - 1) * 2*nb_dim + 1]]
    all_qdot[:, -1] = [float(k) for k in
                       lin[nb_dim + 1 + (nb_phases - 1) * 2*nb_dim:2*nb_dim * nb_phases + 1]]
t_final = t
for p in range(1, nb_phases):
    for j in range(nb_nodes + 1):
        t_final[(nb_nodes * p) + j] = t_final[(nb_nodes * p) - 1] + t[j + 1]

# read controls

i = 0
all_u = np.ndarray((nb_dim, nb_points))
with open(f"../optimal_control/Results/Controls{output_files}.txt", "r") as fichier_u:

    for l in range(nb_nodes):
        line = fichier_u.readline()
        lin = line.split('\t')
        lin[:1] = []
        lin[(nb_phases*nb_dim) + 1:] = []

        for p in range(nb_phases):
            all_u[:, i+p*nb_nodes] = [float(i) for i in lin[1+p*nb_dim:nb_dim*(p+1)+1]]

        i += 1

    line = fichier_u.readline()
    lin = line.split('\t')
    lin[:1] = []
    lin[(nb_phases*nb_dim) + 1:] = []

    all_u[:, -1] = [float(i) for i in lin[1+(nb_phases-1)*nb_dim:nb_dim*nb_phases+1]]


# Show data
plt.figure("Eocar")
for i in range(nb_dim):
    plt.subplot(nb_dim, 3, 1+(3*i))
    plt.plot(t_final, all_q[i, :])
    plt.title("Q %i" % i)

    plt.subplot(nb_dim, 3, 2+(3*i))
    plt.plot(t_final, all_qdot[i, :])
    plt.title("Qdot %i" % i)

for i in range(nb_dim):
    plt.subplot(nb_dim, 3, 3 + (3 * i))
    utils.plot_piecewise_constant(t_final, all_u[i, :])
    plt.title("Torques %i" % i)


# plt.ion()  # Non blocking plt.show
plt.show()

