import scipy.io as sio
import matplotlib.pyplot as plt

a = sio.loadmat('/home/lim/Documents/Preadapt_MOD1100_leftHanded_GenderH_VicFou_1g_.Q2')

# for i in range(44):
#     print(a['Q2'][i, :])

# ------- Clavicule ------- #
plt.figure("Clavicule")

plt.subplot(3, 2, 1)
plt.plot(a['Q2'][12, :])
plt.title("Droit z")

plt.subplot(3, 2, 3)
plt.plot(a['Q2'][13, :])
plt.title("Droit y")

plt.subplot(3, 2, 5)
plt.plot(a['Q2'][14, :])
plt.title("Droit x")

plt.subplot(3, 2, 2)
plt.plot(a['Q2'][28, :])
plt.title("Gauche z")

plt.subplot(3, 2, 4)
plt.plot(a['Q2'][29, :])
plt.title("Gauche y")

plt.subplot(3, 2, 6)
plt.plot(a['Q2'][30, :])
plt.title("Gauche x")

# ------- Scapula ------- #
plt.figure("Scaoula")

plt.subplot(3, 2, 1)
plt.plot(a['Q2'][15, :])
plt.title("Droit z")

plt.subplot(3, 2, 3)
plt.plot(a['Q2'][16, :])
plt.title("Droit y")

plt.subplot(3, 2, 5)
plt.plot(a['Q2'][17, :])
plt.title("Droit x")

plt.subplot(3, 2, 2)
plt.plot(a['Q2'][31, :])
plt.title("Gauche z")

plt.subplot(3, 2, 4)
plt.plot(a['Q2'][32, :])
plt.title("Gauche y")

plt.subplot(3, 2, 6)
plt.plot(a['Q2'][33, :])
plt.title("Gauche x")

# ------- Arm ------- #
plt.figure("Arm")

plt.subplot(3, 2, 1)
plt.plot(a['Q2'][21, :])
plt.title("Droit z")

plt.subplot(3, 2, 3)
plt.plot(a['Q2'][22, :])
plt.title("Droit y")

plt.subplot(3, 2, 5)
plt.plot(a['Q2'][23, :])
plt.title("Droit z")

plt.subplot(3, 2, 2)
plt.plot(a['Q2'][37, :])
plt.title("Gauche z")

plt.subplot(3, 2, 4)
plt.plot(a['Q2'][38, :])
plt.title("Gauche y")

plt.subplot(3, 2, 6)
plt.plot(a['Q2'][39, :])
plt.title("Gauche z")

# ------- Lower Arm 1 ------- #
plt.figure("Lower Arm 1")

plt.subplot(1, 2, 1)
plt.plot(a['Q2'][24, :])
plt.title("Droit x")

plt.subplot(1, 2, 2)
plt.plot(a['Q2'][40, :])
plt.title("Gauche x")

# ------- Lower Arm 2 ------- #
plt.figure("Lower Arm 2")

plt.subplot(1, 2, 1)
plt.plot(a['Q2'][25, :])
plt.title("Droit x")

plt.subplot(1, 2, 2)
plt.plot(a['Q2'][41, :])
plt.title("Gauche x")

plt.show()
