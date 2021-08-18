import os
import time
from typing import Any

import biorbd_casadi as biorbd
import numpy as np
from bioptim import (
    Solver,
    MovingHorizonEstimator,
    OptimalControlProgram,
    Objective,
    ObjectiveFcn,
    ObjectiveList,
    DynamicsFcn,
    Dynamics,
    Constraint,
    ConstraintFcn,
    ConstraintList,
    Bounds,
    QAndQDotBounds,
    InitialGuess,
    Node,
    InterpolationType,
    Solution,
    OdeSolver,
    FatigueList,
    XiaFatigue,
    XiaTauFatigue,
    FatigueBounds,
    FatigueInitialGuess,
    VariableType,
)

from .violin import Violin
from .bow import Bow, BowPosition
from .viz import online_muscle_torque


class ViolinOcp:

    # TODO Get these values from a better method
    tau_min, tau_max, tau_init = -30, 30, 0

    # TODO add external forces?

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
        ode_solver: OdeSolver = OdeSolver.RK4(),
        multi_thread_objective: bool = True,
    ):
        self.ode_solver = ode_solver
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
        self.multi_thread_obj = multi_thread_objective

        self.fatigue_dynamics = None
        if self.fatigable:
            self.fatigue_dynamics = FatigueList()
            for i in range(self.n_tau):
                self.fatigue_dynamics.add(
                    XiaTauFatigue(
                        XiaFatigue(LD=300, LR=300, F=0.05, R=10, scale=self.tau_min),
                        XiaFatigue(LD=300, LR=300, F=0.05, R=10, scale=self.tau_max),
                    ),
                    state_only=True,
                )
            for i in range(self.n_mus):
                self.fatigue_dynamics.add(XiaFatigue(LD=10, LR=10, F=0.01, R=0.002), state_only=True)

        if self.use_muscles:
            self.dynamics = Dynamics(DynamicsFcn.MUSCLE_DRIVEN, with_torque=True, fatigue=self.fatigue_dynamics)
        else:
            self.dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN, fatigue=self.fatigue_dynamics)

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

    def _set_generic_objective_functions(self):
        # Regularization objectives
        self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=0.01, list_index=0, multi_thread=self.multi_thread_obj)
        self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=0.01, list_index=1, multi_thread=self.multi_thread_obj)

        if self.fatigable:
            # self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_FATIGUE, key="tau_minus", weight=10, list_index=2,
            #                              multi_thread=self.multi_thread_obj)
            # self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_FATIGUE, key="tau_plus", weight=10,
            #                              list_index=3,
            #                              multi_thread=self.multi_thread_obj)
            if self.use_muscles:
                self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_FATIGUE, key="muscles", weight=10, list_index=4, multi_thread=self.multi_thread_obj)

        if self.use_muscles:
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau",
                index=self.violin.virtual_tau,
                weight=0.01,
                list_index=5, multi_thread=self.multi_thread_obj
            )
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=10, list_index=6, multi_thread=self.multi_thread_obj)
        else:
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, list_index=7, multi_thread=self.multi_thread_obj)
        self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_QDDOT, weight=10, list_index=8, multi_thread=self.multi_thread_obj)
        self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, list_index=9, multi_thread=self.multi_thread_obj, derivative=True)

    def _set_generic_constraints(self):
        # Keep the bow in contact with the violin
        if self.solver == Solver.IPOPT:
            # Keep the bow align at 90 degrees with the violin
            self.constraints.add(
                ConstraintFcn.TRACK_SEGMENT_WITH_CUSTOM_RT,
                node=Node.ALL,
                segment=self.bow.segment_idx,
                rt=self.violin.rt_on_string,
                list_index=0, multi_thread=self.multi_thread_obj
            )

            self.constraints.add(
                ConstraintFcn.SUPERIMPOSE_MARKERS,
                node=Node.ALL,
                first_marker=self.bow.contact_marker,
                second_marker=self.violin.bridge_marker,
                list_index=1, multi_thread=self.multi_thread_obj
            )
        else:
            # Keep the bow align at 90 degrees with the violin
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.TRACK_SEGMENT_WITH_CUSTOM_RT,
                node=Node.ALL,
                segment=self.bow.segment_idx,
                rt=self.violin.rt_on_string,
                list_index=10, multi_thread=self.multi_thread_obj
            )

            self.objective_functions.add(
                ObjectiveFcn.Lagrange.SUPERIMPOSE_MARKERS,
                node=Node.ALL,
                first_marker=self.bow.contact_marker,
                second_marker=self.violin.bridge_marker,
                list_index=11,
                weight=1000, multi_thread=self.multi_thread_obj
            )

    def _set_bounds(self):
        self.x_bounds = QAndQDotBounds(self.model)
        self.x_bounds[:self.n_q, 0] = self.violin.q(self.bow_starting)
        self.x_bounds[self.n_q:, 0] = 0

        if self.fatigable:
            self.x_bounds.concatenate(FatigueBounds(self.fatigue_dynamics))

        if self.fatigable:
            self.u_bounds = FatigueBounds(self.fatigue_dynamics, variable_type=VariableType.CONTROLS)
        else:
            u_min = [self.tau_min] * self.n_tau + [0] * self.n_mus
            u_max = [self.tau_max] * self.n_tau + [1] * self.n_mus
            self.u_bounds = Bounds(u_min, u_max)

    def _set_initial_guess(self, init_file):
        if init_file is None:
            self.x_init = InitialGuess(np.concatenate((self.violin.q(self.bow_starting), np.zeros(self.n_q))))
            if self.fatigable:
                self.x_init.concatenate(FatigueInitialGuess(self.fatigue_dynamics))

            if self.fatigable:
                self.u_init = FatigueInitialGuess(self.fatigue_dynamics, variable_type=VariableType.CONTROLS)
            else:
                self.u_init = InitialGuess(np.zeros((self.n_tau + self.n_mus, 1)))

        else:
            _, sol = ViolinOcp.load(init_file)
            self.x_init = InitialGuess(sol.states["all"], interpolation=InterpolationType.EACH_FRAME)
            self.u_init = InitialGuess(sol.controls["all"][:, :-1], interpolation=InterpolationType.EACH_FRAME)

    def set_bow_target_objective(self, bow_target: np.ndarray, weight: float = 10000, sol: Solution = None):
        new_objectives = Objective(
            ObjectiveFcn.Lagrange.TRACK_STATE,
            key="q",
            node=Node.ALL,
            weight=weight,
            target=bow_target,
            index=self.bow.hair_idx,
            list_index=12, multi_thread=self.multi_thread_obj
        )
        self.ocp.update_objectives(new_objectives)

        if self.solver == Solver.IPOPT:
            new_constraint = Constraint(
                ConstraintFcn.TRACK_STATE,
                key="q",
                node=Node.ALL,
                target=bow_target,
                min_bound=-0.05,
                max_bound=0.05,
                index=self.bow.hair_idx,
                list_index=2, multi_thread=self.multi_thread_obj
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
            ode_solver=self.ode_solver,
        )

    def solve(self, limit_memory_max_iter, exact_max_iter, load_path=None, force_no_graph=False):

        sol = None
        if limit_memory_max_iter > 0:
            sol = self.ocp.solve(
                show_online_optim=exact_max_iter == 0 and not force_no_graph,
                solver_options={
                    "hessian_approximation": "limited-memory",
                    "max_iter": limit_memory_max_iter,
                    "linear_solver": "ma57"
                },
            )
        if limit_memory_max_iter > 0 and exact_max_iter > 0:
            self.ocp.set_warm_start(sol)
        if exact_max_iter > 0:
            sol = self.ocp.solve(
                show_online_optim=True and not force_no_graph,
                solver_options={
                    "hessian_approximation": "exact",
                    "max_iter": exact_max_iter,
                    "warm_start_init_point": "yes",
                    "linear_solver": "ma57",
                },
            )

        return sol

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
            multi_thread_objective=False
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
