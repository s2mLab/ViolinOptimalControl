from typing import Union

from bioptim import (
    FatigueList,
    Dynamics,
    ObjectiveList,
    ConstraintList,
    ConstraintFcn,
    Node,
    Axis,
    QAndQDotBounds,
    FatigueBounds,
    Bounds,
    VariableType,
    InitialGuess,
    FatigueInitialGuess,
    OptimalControlProgram,
    Solver,
)
from bioptim.dynamics.ode_solver import OdeSolverBase
import biorbd_casadi as biorbd

from .enums import DynamicsFcn
from .fatigue_model import FatigueModel


class OcpConfiguration:
    def __init__(
        self,
        name: str,
        model_path: str,
        n_shoot: int,
        final_time: float,
        x0: tuple[float, ...],
        tau_limits: tuple[float, float],
        dynamics: DynamicsFcn,
        fatigue_model: FatigueModel,
        objectives: ObjectiveList,
        constraints: ConstraintList,
        use_sx: bool,
        ode_solver: OdeSolverBase,
        solver: Union[Solver, Solver.Generic],
        n_threads: int,
    ):
        # Initialize the meta ocp parameters
        self.name = name
        self.save_name = name.replace("$", "")
        self.save_name = self.save_name.replace("\\", "")
        self.n_shoot = n_shoot
        self.final_time = final_time
        self.use_sx = use_sx
        self.ode_solver = ode_solver
        self.solver: Solver = solver
        self.n_threads = n_threads

        # Initializing model
        self.model = biorbd.Model(model_path)
        self.n_q = self.model.nbQ()
        self.n_tau = self.model.nbGeneralizedTorque()
        self.n_muscles = self.model.nbMuscleTotal()
        self.tau_min, self.tau_max = tau_limits

        # Initialize objectives of the problem
        self.objectives = objectives

        # Initialize constraints of the problem
        self.constraints = constraints
        self.constraints.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS,
            first_marker="target",
            second_marker="COM_hand",
            node=Node.END,
            axes=[Axis.X, Axis.Y],
        )

        # Initializing dynamics
        self.fatigue = FatigueList()
        if dynamics == DynamicsFcn.TORQUE_DRIVEN:
            if fatigue_model is not None:
                for _ in range(self.n_tau):
                    self.fatigue.add(fatigue_model.model)
            self.dynamics = Dynamics(dynamics.value, expand=False, fatigue=self.fatigue)
        elif dynamics == DynamicsFcn.MUSCLE_DRIVEN:
            if fatigue_model is not None:
                for _ in range(self.n_muscles):
                    self.fatigue.add(fatigue_model.model)
            self.dynamics = Dynamics(dynamics.value, expand=False, fatigue=self.fatigue, with_torque=True)
        else:
            raise NotImplementedError("Dynamics not implemented yet")

        # Initialize path constraints and initial guesses for x
        self.x_bounds = QAndQDotBounds(self.model)
        self.x_bounds[:, 0] = x0
        self.x_bounds[self.n_q :, -1] = 0  # Final velocities are null
        self.x_bounds.concatenate(FatigueBounds(self.fatigue, fix_first_frame=True))

        self.x_init = InitialGuess(x0)
        self.x_init.concatenate(FatigueInitialGuess(self.fatigue))

        # Initialize path constraints and initial guesses for u
        self.u_bounds = Bounds()
        self.u_init = InitialGuess()
        if dynamics == DynamicsFcn.TORQUE_DRIVEN:
            if fatigue_model is None:
                self.u_bounds.concatenate(Bounds([self.tau_min] * self.n_tau, [self.tau_max] * self.n_tau))
                self.u_init.concatenate(InitialGuess([0] * self.n_tau))
            else:
                self.u_bounds.concatenate(FatigueBounds(self.fatigue, variable_type=VariableType.CONTROLS))
                self.u_init.concatenate(FatigueInitialGuess(self.fatigue, variable_type=VariableType.CONTROLS))
        elif dynamics == DynamicsFcn.MUSCLE_DRIVEN:
            self.u_bounds.concatenate(Bounds([self.tau_min] * self.n_tau, [self.tau_max] * self.n_tau))
            self.u_init.concatenate(InitialGuess([0] * self.n_tau))
            if fatigue_model is None:
                self.u_bounds.concatenate(Bounds([0] * self.n_muscles, [1] * self.n_muscles))
                self.u_init.concatenate(InitialGuess([0] * self.n_muscles))
            else:
                self.u_bounds.concatenate(FatigueBounds(self.fatigue, variable_type=VariableType.CONTROLS))
                self.u_init.concatenate(FatigueInitialGuess(self.fatigue, variable_type=VariableType.CONTROLS))
        else:
            raise NotImplementedError("Dynamics not implemented yet")

        # Initialize the actual OCP
        self.ocp = OptimalControlProgram(
            biorbd_model=self.model,
            dynamics=self.dynamics,
            n_shooting=self.n_shoot,
            phase_time=self.final_time,
            x_init=self.x_init,
            u_init=self.u_init,
            x_bounds=self.x_bounds,
            u_bounds=self.u_bounds,
            objective_functions=self.objectives,
            constraints=self.constraints,
            ode_solver=self.ode_solver,
            use_sx=self.use_sx,
            n_threads=self.n_threads,
        )

    def perform(self):
        return self.ocp.solve(self.solver)
