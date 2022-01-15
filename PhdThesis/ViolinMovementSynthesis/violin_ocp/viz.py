import biorbd_casadi as biorbd
from matplotlib import pyplot as plt
from bioptim import PlotType, OptimalControlProgram
from casadi import MX, Function


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


def online_muscle_torque(ocp: OptimalControlProgram):
    return
    nlp = ocp.nlp[0]

    states = MX.sym("x", nlp.nx, 1)
    controls = MX.sym("u", nlp.nu, 1)
    parameters = MX.sym("u", nlp.np, 1)

    nq = len(nlp.variable_mappings["q"].to_first)
    q = nlp.variable_mappings["q"].to_second.map(states[:nq])
    qdot = nlp.variable_mappings["qdot"].to_second.map(states[nq:])

    muscles_states = nlp.model.stateSet()
    muscles_activations = controls[nlp.shape["tau"] :]
    for k in range(nlp.shape["muscle"]):
        muscles_states[k].setActivation(muscles_activations[k])
    muscles_tau = nlp.model.muscularJointTorque(muscles_states, q, qdot).to_mx()
    muscle_tau_func = Function("muscle_tau", [states, controls, parameters], [muscles_tau]).expand()

    biorbd.to_casadi_func("")

    def muscle_tau_callback(t, s, c, p):
        return muscle_tau_func(s, c, p)

    ocp.add_plot(
        "muscle_torque",
        muscle_tau_callback,
        plot_type=PlotType.STEP,
    )
