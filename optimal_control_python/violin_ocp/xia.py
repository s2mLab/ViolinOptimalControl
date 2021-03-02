from casadi import MX, vertcat, if_else, lt, gt
import biorbd
from biorbd_optim import Problem, CustomPlot, PlotType


class Xia:
    def __init__(self):
        self.R = 0.0085  # 0.002 * 0.5 + 0.01 * 0.25 + 0.02 * 0.25
        self.F = 0.0425  # 0.01 * 0.5 + 0.05 * 0.25 + 0.1 * 0.25
        self.r = 1

    @staticmethod
    def xia_model_dynamic(states, controls, parameters, nlp):
        nbq = nlp["model"].nbQ()
        nbqdot = nlp["model"].nbQdot()
        nb_q_qdot = nbq + nbqdot

        q = states[:nbq]
        qdot = states[nbq:nb_q_qdot]
        active_fibers = states[nb_q_qdot : nb_q_qdot + nlp["nbMuscle"]]
        fatigued_fibers = states[nb_q_qdot + nlp["nbMuscle"] : nb_q_qdot + 2 * nlp["nbMuscle"]]
        resting_fibers = states[nb_q_qdot + 2 * nlp["nbMuscle"] :]

        residual_tau = controls[: nlp["nbTau"]]
        activation = controls[nlp["nbTau"] :]
        command = MX()

        comp = 0
        for i in range(nlp["model"].nbMuscleGroups()):
            for k in range(nlp["model"].muscleGroup(i).nbMuscles()):
                develop_factor = (
                    nlp["model"].muscleGroup(i).muscle(k).characteristics().fatigueParameters().developFactor().to_mx()
                )
                recovery_factor = (
                    nlp["model"].muscleGroup(i).muscle(k).characteristics().fatigueParameters().recoveryFactor().to_mx()
                )

                command = vertcat(
                    command,
                    if_else(
                        lt(active_fibers[comp], activation[comp]),
                        (
                            if_else(
                                gt(resting_fibers[comp], activation[comp] - active_fibers[comp]),
                                develop_factor * (activation[comp] - active_fibers[comp]),
                                develop_factor * resting_fibers[comp],
                            )
                        ),
                        recovery_factor * (activation[comp] - active_fibers[comp]),
                    ),
                )
                comp += 1
        restingdot = -command + Muscles.r * Muscles.R * fatigued_fibers  # todo r=r when activation=0
        activatedot = command - Muscles.F * active_fibers
        fatiguedot = Muscles.F * active_fibers - Muscles.R * fatigued_fibers

        muscles_states = biorbd.VecBiorbdMuscleState(nlp["nbMuscle"])
        for k in range(nlp["nbMuscle"]):
            muscles_states[k].setActivation(active_fibers[k])
        # todo fix force max

        muscles_tau = nlp["model"].muscularJointTorque(muscles_states, q, qdot).to_mx()
        # todo get muscle forces and multiply them by activate [k] and same as muscularJointTorque
        tau = muscles_tau + residual_tau
        dxdt = MX(nlp["nx"], nlp["ns"])

        if "external_forces" in nlp:
            for i, f_ext in enumerate(nlp["external_forces"]):
                qddot = biorbd.Model.ForwardDynamics(nlp["model"], q, qdot, tau, f_ext).to_mx()
                dxdt[:, i] = vertcat(qdot, qddot, activatedot, fatiguedot, restingdot)
        else:
            qddot = biorbd.Model.ForwardDynamics(nlp["model"], q, qdot, tau).to_mx()
            dxdt = vertcat(qdot, qddot, activatedot, fatiguedot, restingdot)

        return dxdt

    @staticmethod
    def xia_model_configuration(ocp, nlp):
        Problem.configure_q_qdot(nlp, True, False)
        Problem.configure_tau(nlp, False, True)
        Problem.configure_muscles(nlp, False, True)

        x = MX()
        for i in range(nlp["nbMuscle"]):
            x = vertcat(x, MX.sym(f"Muscle_{nlp['muscleNames']}_active_{nlp['phase_idx']}", 1, 1))
        for i in range(nlp["nbMuscle"]):
            x = vertcat(x, MX.sym(f"Muscle_{nlp['muscleNames']}_fatigue_{nlp['phase_idx']}", 1, 1))
        for i in range(nlp["nbMuscle"]):
            x = vertcat(x, MX.sym(f"Muscle_{nlp['muscleNames']}_resting_{nlp['phase_idx']}", 1, 1))

        nlp["x"] = vertcat(nlp["x"], x)
        nlp["var_states"]["muscles_active"] = nlp["nbMuscle"]
        nlp["var_states"]["muscles_fatigue"] = nlp["nbMuscle"]
        nlp["var_states"]["muscles_resting"] = nlp["nbMuscle"]
        nlp["nx"] = nlp["x"].rows()

        nb_q_qdot = nlp["nbQ"] + nlp["nbQdot"]
        nlp["plot"]["muscles_active"] = CustomPlot(
            lambda x, u, p: x[nb_q_qdot : nb_q_qdot + nlp["nbMuscle"]],
            plot_type=PlotType.INTEGRATED,
            legend=nlp["muscleNames"],
            color="r",
            ylim=[0, 1],
        )

        combine = "muscles_active"
        nlp["plot"]["muscles_fatigue"] = CustomPlot(
            lambda x, u, p: x[nb_q_qdot + nlp["nbMuscle"] : nb_q_qdot + 2 * nlp["nbMuscle"]],
            plot_type=PlotType.INTEGRATED,
            legend=nlp["muscleNames"],
            combine_to=combine,
            color="g",
            ylim=[0, 1],
        )
        nlp["plot"]["muscles_resting"] = CustomPlot(
            lambda x, u, p: x[nb_q_qdot + 2 * nlp["nbMuscle"] : nb_q_qdot + 3 * nlp["nbMuscle"]],
            plot_type=PlotType.INTEGRATED,
            legend=nlp["muscleNames"],
            combine_to=combine,
            color="b",
            ylim=[0, 1],
        )

        Problem.configure_forward_dyn_func(ocp, nlp, xia_model_dynamic)

    @staticmethod
    def xia_initialize_fatigue_at_zero(ocp, nlp, t, x, u, p):
        offset = nlp["nbQ"] + nlp["nbQdot"] + nlp["nbMuscle"]
        val = []
        for k in range(nlp["nbMuscle"]):
            val = vertcat(val, x[0][offset + k])
        return val

    @staticmethod
    def xia_model_fibers(ocp, nlp, t, x, u, p):
        offset = nlp["model"].nbQ() + nlp["model"].nbQdot()
        val = []
        for k in range(nlp["nbMuscle"]):
            val = vertcat(
                val, 1 - x[0][offset + k] - x[0][offset + k + nlp["nbMuscle"]] - x[0][offset + k + 2 * nlp["nbMuscle"]]
            )
        return val
