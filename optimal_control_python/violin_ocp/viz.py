from matplotlib import pyplot as plt


def display_graphics_x_est(target, x_est):
    plt.suptitle("X_est")
    for dof in range(10):
        plt.subplot(2, 5, int(dof + 1))
        if dof == 9:
            plt.plot(target[: x_est.shape[1]], color="red")
        plt.plot(x_est[dof, :], color="blue")
        plt.title(f"dof {dof}")
        plt.show()


def display_x_est(target, x_est, bow):
    plt.suptitle("X_est and target")
    plt.plot(target[: x_est.shape[1]], color="red")
    plt.title(f"target")
    plt.plot(x_est[bow.hair_idx, :], color="blue")
    plt.title(f"dof {bow.hair_idx}")
    plt.show()


def compare_target(target, target_curve):
    plt.suptitle("target_curve et target modulo")
    plt.subplot(2, 1, 1)
    plt.plot(target, color="blue")
    plt.title(f"target")
    plt.subplot(2, 1, 2)
    plt.plot(target_curve, color="red")
    plt.title(f"target_curve")
    plt.show()


