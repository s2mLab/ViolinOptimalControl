import os
import time
from typing import Union

import biorbd_casadi as biorbd
import numpy as np
from bioptim import (
    Solver,
    MultiCyclicNonlinearModelPredictiveControl,
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
    FatigueBounds,
    FatigueInitialGuess,
    VariableType,
)

from .enums import FatigueType, StructureType
from .violin import Violin
from .bow import Bow, BowPosition
from .viz import online_muscle_torque


class ViolinOcp:

    # TODO add external forces?

    def __init__(
        self,
        model_path: str,
        violin: Violin,
        bow: Bow,
        n_cycles: int,
        bow_starting: BowPosition,
        structure_type: StructureType,
        fatigue_type: FatigueType = FatigueType.NO_FATIGUE,
        init_file: str = None,
        split_tau: bool = False,
        minimize_fatigue: bool = True,
        time_per_cycle: float = 1,
        n_shooting_per_cycle: int = 30,
        solver: Solver.Generic = Solver.IPOPT(),
        ode_solver: Union[OdeSolver.RK4, OdeSolver.COLLOCATION] = OdeSolver.RK4(),
        n_threads: int = 8,
    ):
        self.ode_solver = ode_solver
        self.model_path = model_path
        self.model = biorbd.Model(self.model_path)
        self.n_q = self.model.nbQ()
        self.n_tau = self.model.nbGeneralizedTorque()
        self.structure_type = structure_type
        self.split_tau = split_tau
        self.fatigue_model = fatigue_type
        self.n_mus = self.model.nbMuscles() if self.structure_type == StructureType.MUSCLE else 0

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
        self.expand = True

        self.fatigue_dynamics = None
        self.minimize_fatigue = minimize_fatigue
        if self.fatigue_model != FatigueType.NO_FATIGUE:
            self.fatigue_dynamics = FatigueList()
            if self.structure_type == StructureType.TAU:
                for i in range(self.n_tau):
                    self.fatigue_dynamics.add(violin.fatigue_model(self.fatigue_model, self.structure_type, i, self.split_tau))
            for i in range(self.n_mus):
                self.fatigue_dynamics.add(violin.fatigue_model(self.fatigue_model, self.structure_type))

        if self.structure_type == StructureType.MUSCLE:
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
        if self.structure_type == StructureType.MUSCLE:
            online_muscle_torque(self.ocp)

        if self.ode_solver == OdeSolver.RK4:
            self.ocp.add_plot_penalty()

    def _set_generic_objective_functions(self):
        # Regularization objectives
        self.objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=0.01, list_index=0, expand=self.expand
        )
        self.objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=0.01, list_index=1, expand=self.expand
        )
        self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_QDDOT, weight=10, list_index=13, expand=self.expand)
        if self.fatigue_model != FatigueType.NO_FATIGUE and self.structure_type == StructureType.TAU and self.split_tau:
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
                key="tau_minus",
                weight=100,
                list_index=6,
                expand=self.expand,
                derivative=True,
            )
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
                key="tau_plus",
                weight=100,
                list_index=7,
                expand=self.expand,
                derivative=True,
            )
        else:
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
                key="tau",
                weight=200,
                list_index=10,
                expand=self.expand,
                derivative=True,
            )

        # Minimize efforts
        if self.structure_type == StructureType.MUSCLE:
            if self.fatigue_model != FatigueType.NO_FATIGUE:
                raise NotImplementedError("Objective function for use_muscles and fatigable is not ready yet")
            else:
                # self.objective_functions.add(
                #     ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
                #     key="tau",
                #     index=self.violin.residual_tau,
                #     weight=1000,
                #     list_index=11,
                #     expand=self.expand,
                # )
                self.objective_functions.add(
                    ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=1, list_index=12, expand=self.expand
                )

        # Fatigue objectives
        if self.fatigue_model != FatigueType.NO_FATIGUE and self.minimize_fatigue:
            if self.structure_type == StructureType.TAU:
                self.objective_functions.add(
                    ObjectiveFcn.Lagrange.MINIMIZE_FATIGUE,
                    key="tau_minus",
                    weight=100_000,
                    list_index=2,
                    expand=self.expand,
                )
                self.objective_functions.add(
                    ObjectiveFcn.Lagrange.MINIMIZE_FATIGUE,
                    key="tau_plus",
                    weight=100_000,
                    list_index=3,
                    expand=self.expand,
                )
            elif self.structure_type == StructureType.MUSCLE:
                raise NotImplementedError("Objective function for use_muscles and fatigable is not ready yet")
                # self.objective_functions.add(
                #     ObjectiveFcn.Lagrange.MINIMIZE_FATIGUE,
                #     key="muscles",
                #     weight=100_000,
                #     list_index=8,
                #     expand=self.expand,
                # )
            else:
                raise NotImplementedError("Structure type not implemented yet")

    def _set_generic_constraints(self):
        # Keep the bow in contact with the violin
        if isinstance(self.solver, Solver.IPOPT):
            # Keep the bow align at 90 degrees with the violin
            self.constraints.add(
                ConstraintFcn.TRACK_SEGMENT_WITH_CUSTOM_RT,
                node=Node.ALL,
                segment=self.bow.segment_idx,
                rt=self.violin.rt_on_string,
                min_bound=-0.03,
                max_bound=0.03,
                list_index=0,
                expand=self.expand,
            )

            self.constraints.add(
                ConstraintFcn.SUPERIMPOSE_MARKERS,
                node=Node.ALL,
                first_marker=self.bow.contact_marker,
                second_marker=self.violin.bridge_marker,
                list_index=1,
                expand=self.expand,
            )
        else:
            # Keep the bow align at 90 degrees with the violin
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.TRACK_SEGMENT_WITH_CUSTOM_RT,
                node=Node.ALL,
                segment=self.bow.segment_idx,
                rt=self.violin.rt_on_string,
                list_index=14,
                expand=self.expand,
            )

            self.objective_functions.add(
                ObjectiveFcn.Lagrange.SUPERIMPOSE_MARKERS,
                node=Node.ALL,
                first_marker=self.bow.contact_marker,
                second_marker=self.violin.bridge_marker,
                list_index=15,
                weight=1000,
                expand=self.expand,
            )

    def _set_bounds(self):
        self.x_bounds = QAndQDotBounds(self.model)
        self.x_bounds[: self.n_q, 0] = self.violin.q(self.model, self.bow, self.bow_starting)
        self.x_bounds[self.n_q :, 0] = 0

        if self.fatigue_model != FatigueType.NO_FATIGUE:
            self.x_bounds.concatenate(FatigueBounds(self.fatigue_dynamics, fix_first_frame=True))

            # Limit the fatigue below the max level
            if self.fatigue_model == FatigueType.EFFORT_PERCEPTION:
                self.x_bounds.max[self.model.nbQ() * 2:, 1:] = 0.90
            elif self.fatigue_model == FatigueType.QCC:
                if self.structure_type == StructureType.TAU:
                    self.x_bounds.max[self.n_q * 2 + self.n_q * 3 : self.n_q * 2 + self.n_q * 4, 1:] = 0.90
                    self.x_bounds.max[self.n_q * 2 + self.n_q * 7 : self.n_q * 2 + self.n_q * 8, 1:] = 0.90
                elif self.structure_type == StructureType.MUSCLE:
                    self.x_bounds.max[self.n_q * 2 + self.n_mus * 3 : self.n_q * 2 + self.n_mus * 4, 1:] = 0.90
                else:
                    raise NotImplementedError("Fatigue type not implemented yet")
            else:
                raise NotImplementedError("Structure type not implemented yet")

        if self.fatigue_model != FatigueType.NO_FATIGUE:
            self.u_bounds = FatigueBounds(self.fatigue_dynamics, variable_type=VariableType.CONTROLS)
        else:
            u_min = self.violin.tau_min + [0] * self.n_mus  # n_mus == 0 if not self.use_muscle
            u_max = self.violin.tau_max + [1] * self.n_mus  # n_mus == 0 if not self.use_muscle
            self.u_bounds = Bounds(u_min, u_max)

    def _set_initial_guess(self, init_file):
        if init_file is None:
            self.x_init = InitialGuess(np.concatenate((self.violin.q(self.model, self.bow, self.bow_starting), np.zeros(self.n_q))))
            if self.fatigue_model != FatigueType.NO_FATIGUE:
                self.x_init.concatenate(FatigueInitialGuess(self.fatigue_dynamics))

            if self.fatigue_model != FatigueType.NO_FATIGUE:
                self.u_init = FatigueInitialGuess(self.fatigue_dynamics, variable_type=VariableType.CONTROLS)
            else:
                u_init = self.violin.tau_init + [0] * self.n_mus  # n_mus == 0 if not self.use_muscle
                self.u_init = InitialGuess(u_init)

        else:
            _, sol = ViolinOcp.load(init_file)
            self.x_init = InitialGuess(sol.states["all"], interpolation=InterpolationType.EACH_FRAME)
            self.u_init = InitialGuess(sol.controls["all"][:, :-1], interpolation=InterpolationType.EACH_FRAME)

    def set_cyclic_bound(self, slack: float = 0):
        """
        Add a cyclic bound constraint

        Parameters
        ----------
        slack: float
            The slack to the bound constraint, based on the range of motion
        """

        range_of_motion = self.ocp.nlp[0].x_bounds.max[:self.n_q * 2, 1] - self.ocp.nlp[0].x_bounds.min[:self.n_q * 2, 1]
        self.ocp.nlp[0].x_bounds.min[:self.n_q * 2, 2] = self.ocp.nlp[0].x_bounds.min[:self.n_q * 2, 0] - range_of_motion * slack
        self.ocp.nlp[0].x_bounds.max[:self.n_q * 2, 2] = self.ocp.nlp[0].x_bounds.max[:self.n_q * 2, 0] + range_of_motion * slack
        self.ocp.update_bounds(self.ocp.nlp[0].x_bounds)

    def set_bow_target_objective(self, bow_target: np.ndarray, weight: float = 10000, sol: Solution = None):
        if isinstance(self.solver, Solver.IPOPT):
            new_constraint = Constraint(
                ConstraintFcn.TRACK_STATE,
                key="q",
                node=Node.ALL,
                target=bow_target,
                min_bound=-0.05,
                max_bound=0.05,
                index=self.bow.hair_idx,
                list_index=2,
                expand=self.expand,
            )
            self.ocp.update_constraints(new_constraint)
        else:
            new_objectives = Objective(
                ObjectiveFcn.Lagrange.TRACK_STATE,
                key="q",
                node=Node.ALL,
                weight=weight,
                target=bow_target,
                index=self.bow.hair_idx,
                list_index=16,
                expand=self.expand,
            )
            self.ocp.update_objectives(new_objectives)

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
            use_sx=isinstance(self.solver, Solver.ACADOS),
            n_threads=self.n_threads,
            ode_solver=self.ode_solver,
        )

    def solve(self, limit_memory_max_iter, exact_max_iter, load_path=None, force_no_graph=False):

        sol = None
        if limit_memory_max_iter > 0:
            if isinstance(self.solver, Solver.IPOPT):
                self.solver.show_online_optim = exact_max_iter == 0 and not force_no_graph
                self.solver.set_hessian_approximation("limited-memory")
                self.solver.set_maximum_iterations(limit_memory_max_iter)
                self.solver.set_linear_solver("ma57")
            sol = self.ocp.solve(self.solver)

        if limit_memory_max_iter > 0 and exact_max_iter > 0:
            self.ocp.set_warm_start(sol)

        if exact_max_iter > 0:
            if isinstance(self.solver, Solver.IPOPT):
                self.solver.show_online_optim = not force_no_graph
                self.solver.set_hessian_approximation("exact")
                self.solver.set_maximum_iterations(exact_max_iter)
                self.solver.set_linear_solver("ma57")
            sol = self.ocp.solve(self.solver)

        return sol

    @staticmethod
    def load(file_path: str):
        return MultiCyclicNonlinearModelPredictiveControl.load(file_path)

    def save(self, sol: Solution, ext: str = "", stand_alone: bool = False):
        try:
            os.mkdir("results")
        except FileExistsError:
            pass

        ext = "_" + ext if ext else ""
        t = time.localtime(time.time())
        if stand_alone:
            self.ocp.save(sol, f"results/{t.tm_year}_{t.tm_mon}_{t.tm_mday}{ext}_out.bo", stand_alone=True)
        else:
            self.ocp.save(sol, f"results/{t.tm_year}_{t.tm_mon}_{t.tm_mday}{ext}.bo")


class MultiCyclicWithFatigue(MultiCyclicNonlinearModelPredictiveControl):
    def advance_window_initial_guess_states(self, sol, **advance_options):
        super(MultiCyclicWithFatigue, self).advance_window_initial_guess_states(sol, **advance_options)

        if "tau_minus_mf" in self.nlp[0].states:
            x_init = InitialGuess(sol.states["all"][:, self.initial_guess_frames[0]])
            u_init = InitialGuess(sol.controls["all"][:, self.initial_guess_frames], InterpolationType.EACH_FRAME)
            s = Solution(self, [x_init, u_init])
            integrated_sol = s.integrate()
            self.nlp[0].x_init.init[self.nlp[0].states["tau_minus_mf"].index, :] = integrated_sol.states["tau_minus_mf"]
            self.nlp[0].x_init.init[self.nlp[0].states["tau_plus_mf"].index, :] = integrated_sol.states["tau_plus_mf"]


class ViolinNMPC(ViolinOcp):
    def __init__(
        self,
        model_path: str,
        violin: Violin,
        bow: Bow,
        n_cycles_simultaneous,
        n_cycles_to_advance,
        bow_starting: BowPosition,
        structure_type: StructureType,
        fatigue_type: FatigueType = FatigueType.NO_FATIGUE,
        minimize_fatigue=True,
        window_duration: float = 1,
        window_len: int = 30,
        solver: Solver.Generic = Solver.ACADOS(),
        ode_solver: Union[OdeSolver.RK4, OdeSolver.COLLOCATION] = OdeSolver.RK4(),
        n_threads: int = 8,
    ):
        self.n_cycles_simultaneous = n_cycles_simultaneous
        self.n_cycles_to_advance = n_cycles_to_advance
        super(ViolinNMPC, self).__init__(
            model_path=model_path,
            violin=violin,
            bow=bow,
            n_cycles=1,
            bow_starting=bow_starting,
            structure_type=structure_type,
            fatigue_type=fatigue_type,
            minimize_fatigue=minimize_fatigue,
            time_per_cycle=window_duration,
            n_shooting_per_cycle=window_len,
            solver=solver,
            ode_solver=ode_solver,
            n_threads=n_threads,
        )

    def _set_generic_ocp(self):
        self.ocp = MultiCyclicWithFatigue(
            biorbd_model=self.model,
            dynamics=self.dynamics,
            n_cycles_simultaneous=self.n_cycles_simultaneous,
            n_cycles_to_advance=self.n_cycles_to_advance,
            cycle_len=self.n_shooting,
            cycle_duration=self.time,
            x_init=self.x_init,
            u_init=self.u_init,
            x_bounds=self.x_bounds,
            u_bounds=self.u_bounds,
            objective_functions=self.objective_functions,
            constraints=self.constraints,
            use_sx=isinstance(self.solver, Solver.ACADOS),
            n_threads=self.n_threads,
        )

    def solve(self, update_function, warm_start_solution: Solution = None, cycle_from=-1, show_online: bool = False) -> Solution:
        """

        Parameters
        ----------
        update_function
            The function to update between optimizations
        warm_start_solution: Solution
            A solution to warm start from
        cycle_from
            The cycle from which to start the next iteration
        Returns
        -------
        The solution
        """

        if isinstance(self.solver, Solver.IPOPT):
            self.solver.set_linear_solver("ma57")
            self.solver.set_hessian_approximation("limited-memory")
            self.solver.set_c_compile(False)
            self.solver.show_online_optim = show_online

        self.solver.set_maximum_iterations(1000)
        cyclic_options = {"states": ["q", "qdot"]}

        return self.ocp.solve(update_function, solver=self.solver, cyclic_options=cyclic_options, max_consecutive_failing=3, warm_start=warm_start_solution)
