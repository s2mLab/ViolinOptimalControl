from matplotlib import pyplot as plt
from violin_ocp import Bow, BowTrajectory

n_points = 200
x = BowTrajectory(Bow().hair_limits, n_points=n_points)

plt.title("Speed and position of the virtual contact on the bow during the up and down movement")
plt.plot(x.t, x.target.T)
plt.show()
