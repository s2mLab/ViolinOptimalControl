from bioptim import ObjectiveList, ConstraintList, OdeSolver, Solver, ObjectiveFcn

from feasability_study_ocp import DynamicsFcn, FatigableStructure, OcpConfiguration, FatigueModels, FatigueParameters


class StudySetup:
    def __init__(
        self,
        n_shoot: int = 50,
        final_time: float = 1,
        tau_limits_no_muscles: tuple[float, float] = (-100, 100),
        tau_limits_with_muscles: tuple[float, float] = (-1, 1),
        split_controls: bool = False,
        ode_solver: OdeSolver = None,
        solver: Solver = None,
        use_sx: bool = False,
        n_thread: int = 8,
    ):
        self.n_shoot = n_shoot
        self.final_time = final_time
        self.tau_limits_no_muscles = tau_limits_no_muscles
        self.tau_limits_with_muscles = tau_limits_with_muscles
        self.split_controls = split_controls
        self.ode_solver = OdeSolver.RK4() if ode_solver is None else ode_solver
        self.solver = solver
        if self.solver is None:
            self.solver = Solver.IPOPT(show_online_optim=False, _print_level=5, _linear_solver="ma57")
        self.use_sx = use_sx
        self.n_thread = n_thread


class StudyInternal:
    @staticmethod
    def torque_driven_no_fatigue(study_setup: StudySetup):
        fatigue_model = FatigueModels.NONE

        objectives = ObjectiveList()
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)

        return OcpConfiguration(
            model_path="models/arm26.bioMod",
            n_shoot=study_setup.n_shoot,
            final_time=study_setup.final_time,
            tau_limits=study_setup.tau_limits_no_muscles,
            dynamics=DynamicsFcn.TORQUE_DRIVEN,
            fatigue_model=fatigue_model,
            objectives=objectives,
            constraints=ConstraintList(),
            use_sx=study_setup.use_sx,
            ode_solver=study_setup.ode_solver,
            solver=study_setup.solver,
            n_threads=study_setup.n_thread,
        )

    @staticmethod
    def muscles_driven_no_fatigue(study_setup: StudySetup):
        fatigue_model = FatigueModels.NONE

        objectives = ObjectiveList()
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100)
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=1)

        return OcpConfiguration(
            model_path="models/arm26.bioMod",
            n_shoot=study_setup.n_shoot,
            final_time=study_setup.final_time,
            tau_limits=study_setup.tau_limits_with_muscles,
            dynamics=DynamicsFcn.MUSCLE_DRIVEN,
            fatigue_model=fatigue_model,
            objectives=objectives,
            constraints=ConstraintList(),
            use_sx=study_setup.use_sx,
            ode_solver=study_setup.ode_solver,
            solver=study_setup.solver,
            n_threads=study_setup.n_thread,
        )

    @staticmethod
    def torque_driven_michaud(study_setup: StudySetup):
        fatigue_model = FatigueModels.MICHAUD(
            FatigableStructure.JOINTS,
            FatigueParameters(scaling=study_setup.tau_limits_no_muscles[1], split_controls=study_setup.split_controls)
        )

        objectives = ObjectiveList()
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_FATIGUE, key="tau_minus", weight=100)
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_FATIGUE, key="tau_plus", weight=100)

        return OcpConfiguration(
            model_path="models/arm26.bioMod",
            n_shoot=study_setup.n_shoot,
            final_time=study_setup.final_time,
            tau_limits=study_setup.tau_limits_no_muscles,
            dynamics=DynamicsFcn.TORQUE_DRIVEN,
            fatigue_model=fatigue_model,
            objectives=objectives,
            constraints=ConstraintList(),
            use_sx=study_setup.use_sx,
            ode_solver=study_setup.ode_solver,
            solver=study_setup.solver,
            n_threads=study_setup.n_thread,
        )

    @staticmethod
    def muscle_driven_michaud(study_setup: StudySetup):
        fatigue_model = FatigueModels.MICHAUD(FatigableStructure.MUSCLES, FatigueParameters())

        objectives = ObjectiveList()
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100)
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=1)
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_FATIGUE, key="muscles", weight=100)

        return OcpConfiguration(
            model_path="models/arm26.bioMod",
            n_shoot=study_setup.n_shoot,
            final_time=study_setup.final_time,
            tau_limits=study_setup.tau_limits_with_muscles,
            dynamics=DynamicsFcn.MUSCLE_DRIVEN,
            fatigue_model=fatigue_model,
            objectives=objectives,
            constraints=ConstraintList(),
            use_sx=study_setup.use_sx,
            ode_solver=study_setup.ode_solver,
            solver=study_setup.solver,
            n_threads=study_setup.n_thread,
        )

    @staticmethod
    def torque_driven_effort_perception(study_setup: StudySetup):
        fatigue_model = FatigueModels.EFFORT_PERCEPTION(
            FatigableStructure.JOINTS,
            FatigueParameters(scaling=study_setup.tau_limits_no_muscles[1], split_controls=study_setup.split_controls)
        )

        objectives = ObjectiveList()
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_FATIGUE, key="tau_minus", weight=100)
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_FATIGUE, key="tau_plus", weight=100)

        return OcpConfiguration(
            model_path="models/arm26.bioMod",
            n_shoot=study_setup.n_shoot,
            final_time=study_setup.final_time,
            tau_limits=study_setup.tau_limits_no_muscles,
            dynamics=DynamicsFcn.TORQUE_DRIVEN,
            fatigue_model=fatigue_model,
            objectives=objectives,
            constraints=ConstraintList(),
            use_sx=study_setup.use_sx,
            ode_solver=study_setup.ode_solver,
            solver=study_setup.solver,
            n_threads=study_setup.n_thread,
        )

    @staticmethod
    def muscle_driven_effort_perception(study_setup: StudySetup):
        fatigue_model = FatigueModels.EFFORT_PERCEPTION(FatigableStructure.MUSCLES, FatigueParameters())

        objectives = ObjectiveList()
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100)
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="muscles", weight=1)
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_FATIGUE, key="muscles", weight=100)

        return OcpConfiguration(
            model_path="models/arm26.bioMod",
            n_shoot=study_setup.n_shoot,
            final_time=study_setup.final_time,
            tau_limits=study_setup.tau_limits_with_muscles,
            dynamics=DynamicsFcn.MUSCLE_DRIVEN,
            fatigue_model=fatigue_model,
            objectives=objectives,
            constraints=ConstraintList(),
            use_sx=study_setup.use_sx,
            ode_solver=study_setup.ode_solver,
            solver=study_setup.solver,
            n_threads=study_setup.n_thread,
        )


class Study:
    TORQUE_DRIVEN_NO_FATIGUE = StudyInternal.torque_driven_no_fatigue(StudySetup())
    MUSCLE_DRIVEN_NO_FATIGUE = StudyInternal.muscles_driven_no_fatigue(StudySetup())

    TORQUE_DRIVEN_MICHAUD = StudyInternal.torque_driven_michaud(StudySetup(split_controls=True))
    TORQUE_DRIVEN_MICHAUD_NON_SPLIT = StudyInternal.torque_driven_michaud(StudySetup(split_controls=False))
    MUSCLE_DRIVEN_MICHAUD = StudyInternal.muscle_driven_michaud(StudySetup())

    TORQUE_DRIVEN_EFFORT_PERCEPTION = StudyInternal.torque_driven_effort_perception(StudySetup(split_controls=True))
    TORQUE_DRIVEN_EFFORT_PERCEPTION_NON_SPLIT = StudyInternal.torque_driven_effort_perception(
        StudySetup(split_controls=False)
    )
    MUSCLE_DRIVEN_EFFORT_PERCEPTION = StudyInternal.muscle_driven_effort_perception(StudySetup())
