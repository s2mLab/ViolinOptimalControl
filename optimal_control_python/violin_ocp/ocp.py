import os
import time
from typing import Any

import biorbd
import numpy as np
from bioptim import (
    Solver,
    OptimalControlProgram,
    Objective,
    ObjectiveFcn,
    ObjectiveList,
    DynamicsFcn,
    Dynamics,
    ConstraintFcn,
    Constraint,
    ConstraintList,
    Bounds,
    QAndQDotBounds,
    InitialGuess,
    Node,
    InterpolationType,
    Solution
)

from .violin import Violin
from .bow import Bow, BowTrajectory, BowPosition


class ViolinOcp:

    # TODO Get these values from a better method
    tau_min, tau_max, tau_init = -100, 100, 0

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
            bow_starting: BowPosition.FROG,
            init_file: str = None,
            use_muscles: bool = True,
            time_per_cycle: float = 1,
            n_shooting_per_cycle: int = 30,
            solver: Solver = Solver.IPOPT,
            n_threads: int = 8,
    ):
        self.model_path = model_path
        self.model = biorbd.Model(self.model_path)
        self.n_q = self.model.nbQ()
        self.n_tau = self.model.nbGeneralizedTorque()
        self.n_mus = self.model.nbMuscles() if use_muscles else 0

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
        if use_muscles:
            self.dynamics = Dynamics(DynamicsFcn.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN)
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

    def _set_generic_objective_functions(self):
        weight_align_rt = 1000  # 1000 if self.solver == Solver.ACADOS else 100

        # Regularization objectives
        self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_ALL_CONTROLS, weight=0.01, list_index=0)
        if self.n_mus == 0:
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=1, list_index=1)
        else:
            self.objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_TORQUE,
                index=range(0, 6),
                weight=1,
                list_index=1
            )
            self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_MUSCLES_CONTROL, weight=1, list_index=2)
        self.objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE_DERIVATIVE, weight=1, list_index=3)

        # Keep the bow align at 90 degrees with the violin
        self.objective_functions.add(
            ObjectiveFcn.Lagrange.TRACK_SEGMENT_WITH_CUSTOM_RT,
            weight=weight_align_rt,
            segment_idx=self.bow.segment_idx,
            rt_idx=self.violin.rt_on_string,
            list_index=4
        )

    def set_bow_target_objective(self, bow_target: np.ndarray, weight: float = 100):
        new_objectives = Objective(
            ObjectiveFcn.Lagrange.TRACK_STATE,
            node=Node.ALL,
            weight=weight,
            target=bow_target,
            index=self.bow.hair_idx,
            list_index=5,
        )
        self.ocp.update_objectives(new_objectives)
        new_constraint = Constraint(
            ConstraintFcn.TRACK_STATE,
            node=Node.ALL,
            target=bow_target,
            min_bound=-0.05,
            max_bound=0.05,
            index=self.bow.hair_idx,
        )
        self.ocp.update_constraints(new_constraint)

    def _set_generic_constraints(self):
        # Keep the bow in contact with the violin
        self.constraints.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS,
            node=Node.ALL,
            first_marker_idx=self.bow.contact_marker,
            second_marker_idx=self.violin.bridge_marker,
        )

        # Keep the bow in contact with the violin, but allow for prediction error
        # for j in range(1, 5):
        #     constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS,
        #                     node=j,
        #                     min_bound=0,
        #                     max_bound=0,
        #                     first_marker_idx=Bow.contact_marker,
        #                     second_marker_idx=violin.bridge_marker, list_index=j)
        # for j in range(5, nb_shooting + 1):
        #     constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS,
        #                     node=j,
        #                     min_bound=-10**(j-14), #-10**(j-14) donne 25 itérations
        #                     max_bound=10**(j-14), # (j-4)/10 donne 21 itérations
        #                     first_marker_idx=Bow.contact_marker,
        #                     second_marker_idx=violin.bridge_marker, list_index=j)
        if self.n_mus != 0:
            self.constraints.add(
                ConstraintFcn.TRACK_TORQUE,
                index=range(0, 6),
                node=Node.ALL,
            )

    def _set_bounds(self):
        self.x_bounds = QAndQDotBounds(self.model)
        self.x_bounds[:self.n_q, 0] = self.violin.q(self.bow_starting)
        self.x_bounds[self.n_q:, 0] = 0

        u_min = [self.tau_min] * self.n_tau + [0] * self.n_mus
        u_max = [self.tau_max] * self.n_tau + [1] * self.n_mus
        self.u_bounds = Bounds(u_min, u_max)

    def _set_initial_guess(self, init_file):
        if init_file is None:
            x_init = np.zeros((self.n_q * 2, 1))
            x_init[:self.n_q, 0] = self.violin.q(self.bow_starting)
            u_init = np.zeros((self.n_tau + self.n_mus, 1))
            self.x_init = InitialGuess(x_init)
            self.u_init = InitialGuess(u_init)

        else:
            _, sol = ViolinOcp.load(init_file)
            self.x_init = InitialGuess(sol.states["all"], interpolation=InterpolationType.EACH_FRAME)
            self.u_init = InitialGuess(sol.controls["all"][:, :-1], interpolation=InterpolationType.EACH_FRAME)

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
                use_sx=False,
                n_threads=self.n_threads,
            )

    def solve(self, **opts: Any) -> Solution:
        return self.ocp.solve(solver=self.solver, **opts)

    @staticmethod
    def load(file_path: str):
        return OptimalControlProgram.load(file_path)

    def save(self, sol: Solution):
        try:
            os.mkdir("results")
        except FileExistsError:
            pass

        t = time.localtime(time.time())
        self.ocp.save(sol, f"results/{t.tm_year}_{t.tm_mon}_{t.tm_mday}.bo")

    #
    # @staticmethod
    # def warm_start_nmpc(sol, ocp, window_len, n_q, n_qdot, n_tau, biorbd_model, acados, shift=1):
    #     x = sol.states["all"]
    #     u = sol.controls["all"][:, :-1]
    #
    #     x_init_np = np.ndarray(x.shape)
    #     x_init_np[:, :-shift] = x[:, shift:]
    #     x_init_np[:, -shift:] = np.array(x[:, -1])[:, np.newaxis]
    #     x_init = InitialGuess(x_init_np, interpolation=InterpolationType.EACH_FRAME)
    #
    #     x_bounds = QAndQDotBounds(biorbd_model)
    #     x_bounds[:, 0] = x_init_np[:, 0]
    #
    #     u_init_np = np.ndarray(u.shape)
    #     u_init_np[:, :-shift] = u[:, shift:]
    #     u_init_np[:, -shift:] = u[:, -1][:, np.newaxis]
    #     u_init = InitialGuess(u_init_np, interpolation=InterpolationType.EACH_FRAME)
    #
    #     ocp.update_initial_guess(x_init, u_init)
    #     ocp.update_bounds(x_bounds=x_bounds)
    #     return x_init, u_init, x[:, 0], u[:, 0], x_bounds
