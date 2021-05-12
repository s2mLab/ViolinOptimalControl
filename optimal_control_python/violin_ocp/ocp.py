import os
import time
from typing import Any, Union

import biorbd
import numpy as np
from casadi import MX, SX, if_else, vertcat, horzcat, lt, gt
from bioptim import (
    Solver,
    MovingHorizonEstimator,
    OptimalControlProgram,
    Objective,
    ObjectiveFcn,
    ObjectiveList,
    DynamicsFcn,
    DynamicsFunctions,
    Dynamics,
    Constraint,
    ConstraintFcn,
    ConstraintList,
    Bounds,
    QAndQDotBounds,
    InitialGuess,
    Node,
    NonLinearProgram,
    PlotType,
    PenaltyNode,
    InterpolationType,
    Solution,
    Problem,
)

from .violin import Violin
from .bow import Bow, BowPosition
from .viz import online_muscle_torque


class ViolinOcp:

    # TODO Get these values from a better method
    tau_min, tau_max, tau_init = -100, 100, 0
    LD, LR, F, R = 100, 100, 0.9, 0.01

    # TODO add external forces?

    # TODO Warm starting when updating the objective_bow_target

    # TODO All the logic from NMPC

    # TODO include the muscle fatigue dynamics, constraints and objectives
    # dynamics.add(xia.xia_model_configuration, dynamic_function=xia.xia_model_dynamic)

    def __init__(
            self,
            model_path: str,
            violin: Violin,
            bow: Bow,
            n_cycles: int,
            bow_starting: BowPosition.TIP,
            init_file: str = None,
            use_muscles: bool = True,
            fatigable: bool = False,
            time_per_cycle: float = 1,
            n_shooting_per_cycle: int = 30,
            solver: Solver = Solver.IPOPT,
            n_threads: int = 8,
    ):
        self.model_path = model_path
        self.model = biorbd.Model(self.model_path)
        self.n_q = self.model.nbQ()
        self.n_tau = self.model.nbGeneralizedTorque()
        self.use_muscles = use_muscles
        self.fatigable = fatigable
        self.n_mus = self.model.nbMuscles() if self.use_muscles else 0

        self.violin = violin
        self.bow = bow
        self.bow_starting = bow_starting

        self.n_cycles = n_cycles
        self.n_shooting_per_cycle = n_shooting_per_cycle
        self.n_shooting = self.n_shooting_per_cycle * self.n_cycles
        self.time_per_cycle = time_per_cycle
        self.time = self.time_per_cycle * self.n_cycles

        self.solver = solver
        self.n_threads = n_threads
        if self.use_muscles:
            self.dynamics = Dynamics(DynamicsFcn.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN)
        else:
            if self.fatigable:
                self.dynamics = Dynamics(self.fatigue_configure, dynamic_function=self.fatigue_dynamics)
            else:
                self.dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

        self.x_bounds = Bounds()
        self.u_bounds = Bounds()
        self._set_bounds()

        self.x_init = InitialGuess()
        self.u_init = InitialGuess()
        self._set_initial_guess(init_file)

        self.objective_functions = ObjectiveList()
        self._set_generic_objective_functions()

        self.constraints = ConstraintList()
        self._set_generic_constraints()

        self._set_generic_ocp()
        if use_muscles:
            online_muscle_torque(self.ocp)

    @staticmethod
    def minimize_fatigue(pn: PenaltyNode) -> MX:
        nq = pn.nlp.shape["q"]
        nqdot = pn.nlp.shape["qdot"]
        fatigable_states = pn.x[nq + nqdot + 2::3, :]
        return fatigable_states

    def _set_generic_objective_functions(self):
        # Regularization objectives
        self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, weight=0.01, list_index=0)
        self.objective_functions.add(self.minimize_fatigue, custom_type=ObjectiveFcn.Lagrange, weight=10, list_index=1)

        if self.use_muscles:
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_TORQUE,
                index=self.violin.virtual_tau,
                weight=0.01,
                list_index=2
            )
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=10, list_index=3)
        else:
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=0.01, list_index=2)
        self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_QDDOT, weight=0.01, list_index=4)

        # Keep the bow align at 90 degrees with the violin
        self.objective_functions.add(
            ObjectiveFcn.Lagrange.TRACK_SEGMENT_WITH_CUSTOM_RT,
            weight=1000,
            segment_idx=self.bow.segment_idx,
            rt_idx=self.violin.rt_on_string,
            list_index=5
        )

    def _set_generic_constraints(self):
        # Keep the bow in contact with the violin
        if self.solver == Solver.IPOPT:
            self.constraints.add(
                ConstraintFcn.SUPERIMPOSE_MARKERS,
                node=Node.ALL,
                first_marker_idx=self.bow.contact_marker,
                second_marker_idx=self.violin.bridge_marker,
                list_index=0,
            )
        else:
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.SUPERIMPOSE_MARKERS,
                node=Node.ALL,
                first_marker_idx=self.bow.contact_marker,
                second_marker_idx=self.violin.bridge_marker,
                list_index=6,
                weight=1000,
            )

    def _set_bounds(self):
        self.x_bounds = QAndQDotBounds(self.model)
        self.x_bounds[:self.n_q, 0] = self.violin.q(self.bow_starting)
        self.x_bounds[self.n_q:, 0] = 0

        if self.fatigable:
            ma_bounds = [[0, 0, 0], [1, 1, 1]]
            mr_bounds = [[0, 0, 0], [1, 1, 1]]
            mf_bounds = [[0, 0, 0], [1, 1, 1]]
            for dof in range(self.n_tau * 2):
                self.x_bounds.concatenate(Bounds([ma_bounds[0]], [ma_bounds[1]]))
                self.x_bounds.concatenate(Bounds([mr_bounds[0]], [mr_bounds[1]]))
                self.x_bounds.concatenate(Bounds([mf_bounds[0]], [mf_bounds[1]]))

        if self.fatigable:
            u_bounds_min = [self.tau_min] * self.n_tau + [0] * self.n_tau + [0] * self.n_mus
            u_bounds_max = [0] * self.n_tau + [self.tau_max] * self.n_tau + [1] * self.n_mus
            self.u_bounds = Bounds(u_bounds_min, u_bounds_max)

        else:
            u_min = [self.tau_min] * self.n_tau + [0] * self.n_mus
            u_max = [self.tau_max] * self.n_tau + [1] * self.n_mus
            self.u_bounds = Bounds(u_min, u_max)

    def _set_initial_guess(self, init_file):
        if init_file is None:
            if self.fatigable:
                x_init = np.zeros((self.n_q * 2 + 6 * self.n_tau, 1))
                x_init[2 * self.n_q:, 0] = [0, 1, 0, 0, 1, 0] * self.n_tau
                u_init = np.zeros((self.n_tau * 2 + self.n_mus, 1))
            else:
                x_init = np.zeros((self.n_q * 2, 1))
                u_init = np.zeros((self.n_tau + self.n_mus, 1))
            x_init[:self.n_q, 0] = self.violin.q(self.bow_starting)
            self.x_init = InitialGuess(x_init)
            self.u_init = InitialGuess(u_init)

        else:
            _, sol = ViolinOcp.load(init_file)
            self.x_init = InitialGuess(sol.states["all"], interpolation=InterpolationType.EACH_FRAME)
            self.u_init = InitialGuess(sol.controls["all"][:, :-1], interpolation=InterpolationType.EACH_FRAME)

    def set_bow_target_objective(self, bow_target: np.ndarray, weight: float = 10000, sol: Solution = None):
        new_objectives = Objective(
            ObjectiveFcn.Lagrange.TRACK_STATE,
            node=Node.ALL,
            weight=weight,
            target=bow_target,
            index=self.bow.hair_idx,
            list_index=7,
        )
        self.ocp.update_objectives(new_objectives)

        if self.solver == Solver.IPOPT:
            new_constraint = Constraint(
                ConstraintFcn.TRACK_STATE,
                node=Node.ALL,
                target=bow_target,
                min_bound=-0.05,
                max_bound=0.05,
                index=self.bow.hair_idx,
                list_index=1,
            )
            self.ocp.update_constraints(new_constraint)

    def _set_generic_ocp(self):
        self.ocp = OptimalControlProgram(
                biorbd_model=self.model,
                dynamics=self.dynamics,
                n_shooting=self.n_shooting,
                phase_time=self.time,
                x_init=self.x_init,
                u_init=self.u_init,
                x_bounds=self.x_bounds,
                u_bounds=self.u_bounds,
                objective_functions=self.objective_functions,
                constraints=self.constraints,
                use_sx=self.solver == Solver.ACADOS,
                n_threads=self.n_threads,
            )

    def solve(self, **opts: Any) -> Solution:
        return self.ocp.solve(solver=self.solver, **opts)

    @staticmethod
    def fatigue_dynamics(states: Union[MX, SX], controls: Union[MX, SX], parameters: Union[MX, SX], nlp: NonLinearProgram
                       ) -> tuple:

        DynamicsFunctions.apply_parameters(parameters, nlp)
        q, qdot = DynamicsFunctions.dispatch_q_qdot_data(states, controls, nlp)
        n_tau = int(nlp.shape["tau"] / 2)
        tau = controls

        tau_bounds = [[], []]
        for i in range(n_tau):
            tau_bounds[0].append(ViolinOcp.tau_min)
            tau_bounds[1].append(ViolinOcp.tau_max)

        LD, LR, F, R = ViolinOcp.LD, ViolinOcp.LR, ViolinOcp.F, ViolinOcp.R

        fatigue = []
        for i in range(n_tau):  # Get fatigable states
            fatigue.append(states[2 * n_tau + 6 * i: 2 * n_tau + 6 * (i + 1)])

        def fatigue_dot_func(TL, param):
            # Implementation of Xia dynamics
            ma = param[0]
            mr = param[1]
            mf = param[2]
            c = if_else(lt(ma, TL), if_else(gt(mr, TL - ma), LD * (TL - ma), LD * mr), LR * (TL - ma))
            madot = c - F * ma
            mrdot = -c + R * mf + 100 * (1 - (ma + mr + mf))
            mfdot = F * ma - R * mf
            return vertcat(madot, mrdot, mfdot)

        fatigue_dot = []
        tau_current = []
        n_fatigue_param = 3
        for i in range(n_tau):
            TL_neg = tau[i] / tau_bounds[0][i]
            fatigue_dot.append(fatigue_dot_func(TL_neg, fatigue[i][:n_fatigue_param]))

            TL_pos = tau[i + n_tau] / tau_bounds[1][i]
            fatigue_dot.append(fatigue_dot_func(TL_pos, fatigue[i][n_fatigue_param:]))

            ma_neg, ma_pos = fatigue[i][0], fatigue[i][n_fatigue_param]
            tau_current.append(ma_neg * tau_bounds[0][i] + ma_pos * tau_bounds[1][i])

        qddot = nlp.model.ForwardDynamics(q, qdot, vertcat(*tau_current)).to_mx()

        return qdot, qddot, vertcat(*fatigue_dot)

    @staticmethod
    def fatigue_configure(ocp: OptimalControlProgram, nlp: NonLinearProgram):
        Problem.configure_q_qdot(nlp, as_states=True, as_controls=False)

        # Configure fatigable states
        dof_names = [n.to_string() for n in nlp.model.nameDof()]
        n_q = nlp.model.nbQ()
        nlp.shape["tau"] = nlp.model.nbGeneralizedTorque() * 2
        n_tau = int(nlp.shape["tau"] / 2)
        m_names = ["ma", "mr", "mf"]
        m_sides = ["neg", "pos"]
        name_fatigable_states = []
        for i in range(n_tau):
            for side in m_sides:
                for name in m_names:
                    name_fatigable_states.append(f"{name}_{side}_{dof_names[i]}")
        legend_fatigable_states = []
        for name in dof_names:
             legend_fatigable_states.append(f"fatigable_{name}")

        fatigable = []
        for name in name_fatigable_states:
            fatigable.append(nlp.cx.sym(name, 1, 1))
        nlp.x = vertcat(nlp.x, *fatigable)

        nlp.var_states["fatigue"] = len(name_fatigable_states)

        def plot_fatigue(x, u, p, mul, offset):
            return mul * x[2*n_q + offset::6, :]

        ocp.add_plot("states ma mr mf", plot_fatigue, mul=0, offset=0, color='black', legend=legend_fatigable_states)
        ocp.add_plot("states ma mr mf", plot_fatigue, mul=-1, offset=0, plot_type=PlotType.INTEGRATED, color='blue')
        ocp.add_plot("states ma mr mf", plot_fatigue, mul=-1, offset=1, plot_type=PlotType.INTEGRATED, color='green')
        ocp.add_plot("states ma mr mf", plot_fatigue, mul=-1, offset=2, plot_type=PlotType.INTEGRATED, color='red')
        ocp.add_plot("states ma mr mf", plot_fatigue, mul=1, offset=3, plot_type=PlotType.INTEGRATED, color='blue')
        ocp.add_plot("states ma mr mf", plot_fatigue, mul=1, offset=4, plot_type=PlotType.INTEGRATED, color='green')
        ocp.add_plot("states ma mr mf", plot_fatigue, mul=1, offset=5, plot_type=PlotType.INTEGRATED, color='red')

        # Configure controls (tau)
        tau_mx = MX()
        all_tau = [nlp.cx() for _ in range(nlp.control_type.value)]

        for i in range(n_tau):
            for side in m_sides:
                for j in range(len(all_tau)):
                    all_tau[j] = vertcat(all_tau[j], nlp.cx.sym(f"Tau_{side}_{dof_names[i]}_{j}", 1, 1))

        for i, _ in enumerate(nlp.mapping["q"].to_second.map_idx):
            for side in m_sides:
                tau_mx = vertcat(tau_mx, MX.sym(f"Tau_{side}_{dof_names[i]}", 1, 1))

        nlp.tau = MX()
        for i in range(n_tau):
            nlp.tau = vertcat(nlp.tau, tau_mx[i])

        nlp.u = vertcat(nlp.u, horzcat(*all_tau))
        nlp.var_controls["tau"] = nlp.shape["tau"]

        def tau_plot(x, u, p, direction):
            if direction < 0:
                return u[:n_tau, :]
            elif direction > 0:
                return u[n_tau:, :]
            else:
                return np.sum((u[:n_tau, :], u[n_tau:, :]), axis=0)

        legend_fatigable_tau = []
        for name in dof_names:
            legend_fatigable_tau.append(f"Tau_{name}")

        ocp.add_plot("tau", tau_plot, direction=0, plot_type=PlotType.STEP, legend=legend_fatigable_tau, color='black')
        ocp.add_plot("tau", tau_plot, direction=-1, plot_type=PlotType.STEP, color='red')
        ocp.add_plot("tau", tau_plot, direction=1, plot_type=PlotType.STEP, color='green')

        nlp.nx = nlp.x.rows()
        nlp.nu = nlp.u.rows()

        Problem.configure_dynamics_function(ocp, nlp, ViolinOcp.fatigue_dynamics)

    @staticmethod
    def load(file_path: str):
        return MovingHorizonEstimator.load(file_path)

    def save(self, sol: Solution, stand_alone: bool = False):
        try:
            os.mkdir("results")
        except FileExistsError:
            pass

        t = time.localtime(time.time())
        if stand_alone:
            self.ocp.save(sol, f"results/{t.tm_year}_{t.tm_mon}_{t.tm_mday}_out.bo", stand_alone=True)
        else:
            self.ocp.save(sol, f"results/{t.tm_year}_{t.tm_mon}_{t.tm_mday}.bo", stand_alone=False)


class ViolinNMPC(ViolinOcp):
    def __init__(
        self,
        model_path: str,
        violin: Violin,
        bow: Bow,
        bow_starting: BowPosition.TIP,
        use_muscles: bool = False,
        fatigable=True,
        window_duration: float = 1,
        window_len: int = 30,
        solver: Solver = Solver.ACADOS,
        n_threads: int = 8,
    ):
        super(ViolinNMPC, self).__init__(
            model_path=model_path,
            violin=violin,
            bow=bow,
            n_cycles=1,
            bow_starting=bow_starting,
            use_muscles=use_muscles,
            fatigable=fatigable,
            time_per_cycle=window_duration,
            n_shooting_per_cycle=window_len,
            solver=solver,
            n_threads=n_threads,
        )

    def _set_generic_ocp(self):
        self.ocp = MovingHorizonEstimator(
            biorbd_model=self.model,
            dynamics=self.dynamics,
            window_len=self.n_shooting,
            window_duration=self.time,
            x_init=self.x_init,
            u_init=self.u_init,
            x_bounds=self.x_bounds,
            u_bounds=self.u_bounds,
            objective_functions=self.objective_functions,
            constraints=self.constraints,
            use_sx=self.solver == Solver.ACADOS,
            n_threads=self.n_threads,
        )

    def solve(self, update_function, **opts: Any) -> Solution:
        return self.ocp.solve(update_function, solver=self.solver, **opts)
