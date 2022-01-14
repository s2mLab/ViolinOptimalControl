import numpy as np
from bioptim import Solver, OdeSolver, Solution

from violin_ocp import Violin, ViolinString, ViolinOcp, Bow, BowTrajectory, BowPosition, FatigueType, StructureType, DataType


class StudyInternal:
    def __init__(self, name: str, structure_type: StructureType, fatigue_type: FatigueType, rmse_index: int):
        self.name = name
        self.rmse_index = rmse_index

        self.model_name: str = "WuViolin"
        self.violin: Violin = Violin(self.model_name, ViolinString.G)
        self.bow: Bow = Bow(self.model_name)
        self.solver = Solver.IPOPT()
        self.ode_solver = OdeSolver.RK4(n_integration_steps=5)
        self.n_threads = 8

        self.starting_position = BowPosition.TIP
        self.n_shoot_per_cycle = 50
        self.n_cycles = 1
        self.cycle_time = 1

        # Generate a full cycle target
        lim = self.bow.hair_limits if self.starting_position == BowPosition.FROG else [self.bow.hair_limits[1], self.bow.hair_limits[0]]
        self.bow_trajectory = BowTrajectory(lim, self.n_shoot_per_cycle + 1)
        self.bow_trajectory.target = np.tile(self.bow_trajectory.target[:, :-1], self.n_cycles)
        self.bow_trajectory.target = np.concatenate((self.bow_trajectory.target, self.bow_trajectory.target[:, -1][:, np.newaxis]), axis=1)

        self.ocp = ViolinOcp(
            model_path=f"models/{self.model_name}.bioMod",
            violin=self.violin,
            bow=self.bow,
            n_cycles=self.n_cycles,
            bow_starting=self.starting_position,
            structure_type=structure_type,
            fatigue_type=fatigue_type,
            init_file=None,
            minimize_fatigue=True,
            time_per_cycle=self.cycle_time,
            n_shooting_per_cycle=self.n_shoot_per_cycle,
            solver=self.solver,
            ode_solver=self.ode_solver,
            n_threads=self.n_threads,
        )

        self.ocp.set_bow_target_objective(self.bow_trajectory.target)
        self.ocp.set_cyclic_bound(0.01)

    def perform(self, limit_memory_max_iter: int, exact_max_iter: int, force_no_graph: bool):
        return self.ocp.solve(
            limit_memory_max_iter=limit_memory_max_iter,
            exact_max_iter=exact_max_iter,
            force_no_graph=force_no_graph
        )


class StudiesInternal:
    def __init__(self, studies: tuple[StudyInternal, ...]):
        self._has_run = False
        self.studies = studies
        self.solutions: list[Solution, ...] = []

    def perform(self, limit_memory_max_iter: int = 100, exact_max_iter: int = 1000, force_no_graph: bool = True):
        self.solutions: list[Solution, ...] = []
        for study in self.studies:
            self.solutions.append(study.perform(limit_memory_max_iter, exact_max_iter, force_no_graph))
        self._has_run = True

    def generate_latex_table(self):
        if not self._has_run:
            raise RuntimeError("run() must be called before generating the latex table")

        table = \
            f"% These commented lines should be added to the preamble\n" \
            f"% \\usepackage[table]{{xcolor}}\n" \
            f"% \\usepackage{{makecell}}\n" \
            f"% \\definecolor{{lightgray}}{{gray}}{{0.91}}\n" \
            f"\\begin{{table}}[!ht]\n" \
            f" \\rowcolors{{1}}{{}}{{lightgray}}\n" \
            f" \\caption{{Comparaison des métriques d'efficacité entre les modèles de fatigue " \
            f"appliqués sur une dynamique musculaire ou articulaire lors de la résolution d'un \\ocp{{}}}}\n" \
            f" \\label{{table:faisabilite}}\n" \
            f" \\begin{{tabular}}{{lccc}}\n" \
            f"  \\hline\n" \
            f"  \\bfseries Condition & " \
            f"\\bfseries\\makecell[c]{{Nombre\\\\d'itération}} & " \
            f"\\bfseries\\makecell[c]{{Temps\\\\d'optimisation\\\\(s)}} & " \
            f"\\bfseries\\makecell[c]{{Temps moyen\\\\par itération\\\\(s/iteration)}}\\\\ \n" \
            f"  \\hline\n"

        for study, sol in zip(self.studies, self.solutions):
            table += \
                f"  {study.name} " \
                f"& {sol.iterations} " \
                f"& {sol.real_time_to_optimize:0.3f} " \
                f"& {sol.real_time_to_optimize / sol.iterations:0.3f} \\\\\n"

        table += \
            f"  \\hline\n" \
            f" \\end{{tabular}}\n" \
            f"\\end{{table}}"

        print("\n\nThis can be copy pasted to latex to generate the table from the thesis")
        print("**************")
        print(table)
        print("**************")
        print("\n")

    def _rmse(self, data_type, key, idx_ref: int, sol: Solution):
        data_ref = getattr(self.solutions[idx_ref], data_type.value)[key]
        data = getattr(sol, data_type.value)[key]

        e = data_ref - data
        se = e**2
        mse = np.sum(se, axis=1) / data_ref.shape[1]
        rmse = np.sqrt(mse)
        return rmse


class StudyConfig:
    # Debug studies
    DEBUG_TAU_NO_FATIGUE: StudiesInternal = StudiesInternal(
        studies=(
            StudyInternal(
                name=r"$\condTauNf$",
                structure_type=StructureType.TAU,
                fatigue_type=FatigueType.NO_FATIGUE,
                rmse_index=0,
            ),
        )
    )

    # Actual Studies
    STUDY1_OCP: StudiesInternal = StudiesInternal(
        (
            StudyInternal(
                name=r"$\condTauNf$",
                structure_type=StructureType.TAU,
                fatigue_type=FatigueType.NO_FATIGUE,
                rmse_index=0,
            ),
            StudyInternal(
                name=r"$\condTauQcc$",
                structure_type=StructureType.TAU,
                fatigue_type=FatigueType.QCC,
                rmse_index=0,
            ),
            StudyInternal(
                name=r"$\condTauPe$",
                structure_type=StructureType.TAU,
                fatigue_type=FatigueType.EFFORT_PERCEPTION,
                rmse_index=0,
            ),
            StudyInternal(
                name=r"$\condAlphaNf$",
                structure_type=StructureType.MUSCLE,
                fatigue_type=FatigueType.NO_FATIGUE,
                rmse_index=3,
            ),
            StudyInternal(
                name=r"$\condAlphaQcc$",
                structure_type=StructureType.MUSCLE,
                fatigue_type=FatigueType.QCC,
                rmse_index=3,
            ),
            StudyInternal(
                name=r"$\condAlphaPe$",
                structure_type=StructureType.MUSCLE,
                fatigue_type=FatigueType.EFFORT_PERCEPTION,
                rmse_index=3,
            ),
        )
    )


