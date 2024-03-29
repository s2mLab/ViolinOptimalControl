import os
from typing import Union
import pickle

import numpy as np
from bioptim import Solver, OdeSolver, Solution

from violin_ocp import (
    Violin,
    ViolinString,
    ViolinNMPC,
    ViolinOcp,
    Bow,
    BowTrajectory,
    BowPosition,
    FatigueType,
    StructureType,
    LatexTable,
    LatexAnalysesFcn,
    Figures,
    FiguresFcn,
    DataType,
    FigureOptions,
    CustomAnalysesFcn,
    CustomAnalyses,
    CustomAnalysesOption,
    Videos,
    ExtraFigures,
    ExtraFiguresFcn,
    ExtraFigureOption,
)


class StudyInternal:
    def __init__(
        self,
        name: str,
        structure_type: StructureType,
        fatigue_type: FatigueType,
        n_shoot_per_cycle: int,
        n_integration_steps: int,
        n_cycles_total: int,
        n_cycles_simultaneous: int,
        rmse_index: int,
        save_name: str = None,
        fatigue_max_threshold: float = 0.8,
        plot_options: dict = None,
    ):
        self.name = name
        save_name = name if save_name is None else save_name
        self.save_name = save_name.replace("$", "").replace("\ ", "_").replace("\\", "").replace("(", "").replace(")", "").replace("%", "pct")
        self.rmse_index = rmse_index
        self.plot_options = plot_options if plot_options is not None else {}

        self.model_name: str = "WuViolin"
        self.violin: Violin = Violin(self.model_name, ViolinString.G, fatigue_max_threshold)
        self.bow: Bow = Bow(self.model_name)
        self.solver = Solver.IPOPT()
        self.ode_solver = OdeSolver.RK4(n_integration_steps=n_integration_steps)
        self.n_threads = 8

        self.starting_position = BowPosition.TIP
        self.n_shoot_per_cycle = n_shoot_per_cycle
        self.n_cycles_simultaneous = n_cycles_simultaneous
        self.n_cycles_to_advance = 0 if self.n_cycles_simultaneous == 1 else 1
        self.n_cycles_total = n_cycles_total
        self.cycle_time = 1

        self.structure_type = structure_type
        self.fatigue_type = fatigue_type

        self._is_initialized: bool = False
        self.nmpc: Union[ViolinNMPC, None] = None

    @staticmethod
    def nmpc_update_function(ocp, t, sol, n_cycles_total):
        if t >= n_cycles_total:
            print("Finished optimizing!")
            return False

        print(f"\n\nOptimizing cycle #{t + 1}..")
        return True

    def _initialize_nmpc(self):
        self.nmpc = ViolinNMPC(
            model_path=f"models/{self.model_name}.bioMod",
            violin=self.violin,
            bow=self.bow,
            n_cycles_simultaneous=self.n_cycles_simultaneous,
            n_cycles_to_advance=self.n_cycles_to_advance,
            bow_starting=self.starting_position,
            structure_type=self.structure_type,
            fatigue_type=self.fatigue_type,
            minimize_fatigue=True,
            solver=self.solver,
            ode_solver=self.ode_solver,
            window_len=self.n_shoot_per_cycle,
            window_duration=self.cycle_time,
            n_threads=self.n_threads,
        )
        if self.n_cycles_to_advance == 0:
            # This is a hack to transform the multi cyclic into a normal cyclic
            self.nmpc.ocp.time_idx_to_cycle = self.n_shoot_per_cycle
        self.set_target(self.nmpc, self.get_bow_trajectory().target)

    def get_ocp(self) -> ViolinOcp:
        ocp = ViolinOcp(
            model_path=f"models/{self.model_name}.bioMod",
            violin=self.violin,
            bow=self.bow,
            n_cycles=self.n_cycles_simultaneous,
            bow_starting=self.starting_position,
            structure_type=self.structure_type,
            fatigue_type=self.fatigue_type,
            init_file=None,
            minimize_fatigue=True,
            time_per_cycle=self.cycle_time,
            n_shooting_per_cycle=self.n_shoot_per_cycle,
            solver=self.solver,
            ode_solver=self.ode_solver,
            n_threads=self.n_threads,
        )
        self.set_target(ocp, self.get_bow_trajectory().target)
        return ocp

    def get_bow_trajectory(self) -> BowTrajectory:
        # Generate a full cycle target
        lim = (
            self.bow.hair_limits if self.starting_position == BowPosition.FROG else list(reversed(self.bow.hair_limits))
        )
        bow_trajectory = BowTrajectory(lim, self.n_shoot_per_cycle + 1)
        bow_trajectory.target = np.tile(bow_trajectory.target[:, :-1], self.n_cycles_simultaneous)
        bow_trajectory.target = np.concatenate(
            (bow_trajectory.target, bow_trajectory.target[:, -1][:, np.newaxis]), axis=1
        )
        return bow_trajectory

    @staticmethod
    def set_target(optimal_program: Union[ViolinOcp, ViolinNMPC], target: np.ndarray):
        optimal_program.set_bow_target_objective(target)
        optimal_program.set_cyclic_bound(0.01)

    def initialize(self):
        if not self._is_initialized:
            self._initialize_nmpc()

    def perform(self, limit_memory_max_iter: int, exact_max_iter: int, show_graphs: bool):
        self.initialize()

        ocp = self.get_ocp()
        pre_sol = ocp.solve(limit_memory_max_iter=limit_memory_max_iter, exact_max_iter=0, force_no_graph=True)

        return self.nmpc.solve(
            update_function=self.nmpc_update_function,
            max_iter=exact_max_iter,
            warm_start_solution=pre_sol,
            show_online=show_graphs,
            update_function_extra_params={"n_cycles_total": self.n_cycles_total},
        )


class StudiesInternal:
    def __init__(
        self,
        name: str,
        studies: tuple[StudyInternal, ...],
        latex_table: LatexTable = None,
        figures: Figures = None,
        custom_analyses: CustomAnalyses = None,
        videos: Videos = None,
        snapshots: Videos = None,
        extra_figures: ExtraFigures = None
    ):
        self.name = name
        self._has_run = False
        self.studies = studies
        self.solutions: list[tuple[Solution, list[Solution, ...]], ...] = []
        self.latex_table = latex_table
        self.figures = figures
        self.videos = videos
        self.snapshots = snapshots
        self.custom_analyses = custom_analyses
        self.extra_figures = extra_figures

    def perform(
        self,
        reload_if_exists: bool,
        limit_memory_max_iter: int = 100,
        exact_max_iter: int = 1000,
        show_graphs: bool = False,
        save_solutions: bool = True,
    ):
        perform = not reload_if_exists
        if reload_if_exists:
            try:
                self.load_solutions()
            except FileNotFoundError:
                perform = True

        if perform:
            self.solutions: list[tuple[Solution, list[Solution]], ...] = []
            for study in self.studies:
                self.solutions.append(study.perform(limit_memory_max_iter, exact_max_iter, show_graphs))
        self._has_run = True

        if perform and save_solutions:
            self.save_solutions()

    def load_solutions(self):
        print("Loading data, this may take some time...")
        self.solutions: list[tuple[Solution, list[Solution, ...]], ...] = []
        for study in self.studies:
            study.initialize()
            _, sol = study.nmpc.load(f"{self.prepare_and_get_results_dir()}/{study.save_name}.bo")
            all_iterations = []
            for i in range(study.n_cycles_total):
                file_path = f"{self.prepare_and_get_results_dir()}/{study.save_name}_iterations/iteration_{i:04d}.bo"
                with open(file_path, "rb") as file:
                    data = pickle.load(file)
                all_iterations.append(data["sol"])
            self.solutions.append((sol, all_iterations))

    def save_solutions(self):
        for study, (sol, all_iterations) in zip(self.studies, self.solutions):
            study.nmpc.save(sol, save_path=f"{self.prepare_and_get_results_dir()}/{study.save_name}")
            study.nmpc.save(sol, save_path=f"{self.prepare_and_get_results_dir()}/{study.save_name}", stand_alone=True)

            for i, iterations in enumerate(all_iterations):
                study.nmpc.save(
                    iterations,
                    save_path=f"{self.prepare_and_get_results_dir()}/{study.save_name}_iterations/iteration_{i:04d}",
                )
                study.nmpc.save(
                    iterations,
                    save_path=f"{self.prepare_and_get_results_dir()}/{study.save_name}_iterations/iteration_{i:04d}",
                    stand_alone=True,
                )

    def generate_latex_table(self):
        if not self._has_run:
            raise RuntimeError("run() must be called before generating the latex table")

        if self.latex_table is None:
            return

        table = self.latex_table.get_table_text(self, self.solutions)

        save_path = f"{self.prepare_and_get_results_dir()}/results.tex"
        with open(save_path, "w", encoding="utf8") as file:
            file.write(table)
        print("\n\nTex file generated in the results folder")

    def perform_custom_analyses(self):
        if not self._has_run:
            raise RuntimeError("run() must be called before performing the custom analyses")

        if self.custom_analyses is None:
            return

        self.custom_analyses.perform(self)

    def generate_videos(self):
        if not self._has_run:
            raise RuntimeError("run() must be called before generating videos")

        if self.videos is None:
            return

        self.videos.generate_video(self, self.solutions, save_folder=self.prepare_and_get_results_dir())

    def generate_snapshots(self):
        if not self._has_run:
            raise RuntimeError("run() must be called before generating snapshots")

        if self.snapshots is None:
            return

        save_folder = self.prepare_and_get_results_dir()
        save_folder += "/snapshots"
        try:
            os.mkdir(save_folder)
        except FileExistsError:
            pass

        self.snapshots.generate_snapshot(self, self.solutions, save_folder=save_folder)

    def generate_figures(self):
        if not self._has_run:
            raise RuntimeError("run() must be called before generating the figures")

        if self.figures is None:
            return

        self.figures.generate_figure(self, self.solutions, save_folder=self.prepare_and_get_results_dir())

    def generate_extra_figures(self):
        if not self._has_run:
            raise RuntimeError("run() must be called before generating the figures")

        if self.extra_figures is None:
            return

        self.extra_figures.generate_extra_figures(self, save_folder=self.prepare_and_get_results_dir())

    def prepare_and_get_results_dir(self):
        try:
            os.mkdir("results")
        except FileExistsError:
            pass

        try:
            os.mkdir(f"results/{self.name}")
        except FileExistsError:
            pass
        return f"results/{self.name}"


class StudyConfig:
    # Debug studies
    DEBUG_TAU_NO_FATIGUE: StudiesInternal = StudiesInternal(
        name="DEBUG_TAU_NO_FATIGUE",
        studies=(
            StudyInternal(
                name=r"$\condTauNf$",
                structure_type=StructureType.TAU,
                fatigue_type=FatigueType.NO_FATIGUE,
                n_cycles_total=1,
                n_shoot_per_cycle=50,
                n_integration_steps=5,
                n_cycles_simultaneous=1,
                rmse_index=0,
            ),
        ),
    )

    DEBUG_TAU_PE_FATIGUE_3_CYCLES: StudiesInternal = StudiesInternal(
        name="DEBUG_TAU_PE_FATIGUE_3_CYCLES",
        studies=(
            StudyInternal(
                name=r"$\condTauPe$",
                structure_type=StructureType.TAU,
                fatigue_type=FatigueType.EFFORT_PERCEPTION,
                n_cycles_total=3,
                n_shoot_per_cycle=50,
                n_integration_steps=5,
                n_cycles_simultaneous=1,
                rmse_index=0,
            ),
        ),
    )

    DEBUG_ALL_TAU: StudiesInternal = StudiesInternal(
        name="DEBUG_ALL_TAU",
        studies=(
            StudyInternal(
                name=r"$\condTauNf$",
                structure_type=StructureType.TAU,
                fatigue_type=FatigueType.NO_FATIGUE,
                n_cycles_total=1,
                n_shoot_per_cycle=50,
                n_integration_steps=5,
                n_cycles_simultaneous=1,
                rmse_index=0,
            ),
            StudyInternal(
                name=r"$\condTauQcc$",
                structure_type=StructureType.TAU,
                fatigue_type=FatigueType.QCC,
                n_cycles_total=1,
                n_shoot_per_cycle=50,
                n_integration_steps=5,
                n_cycles_simultaneous=1,
                rmse_index=0,
            ),
            StudyInternal(
                name=r"$\condTauPe$",
                structure_type=StructureType.TAU,
                fatigue_type=FatigueType.EFFORT_PERCEPTION,
                n_cycles_total=1,
                n_shoot_per_cycle=50,
                n_integration_steps=5,
                n_cycles_simultaneous=1,
                rmse_index=0,
            ),
        ),
    )

    DEBUG_ALL_MUSCLE: StudiesInternal = StudiesInternal(
        name="DEBUG_ALL_MUSCLE",
        studies=(
            StudyInternal(
                name=r"$\condAlphaNf$",
                structure_type=StructureType.MUSCLE,
                fatigue_type=FatigueType.NO_FATIGUE,
                n_cycles_total=1,
                n_shoot_per_cycle=50,
                n_integration_steps=5,
                n_cycles_simultaneous=1,
                rmse_index=0,
            ),
            StudyInternal(
                name=r"$\condAlphaQcc$",
                structure_type=StructureType.MUSCLE,
                fatigue_type=FatigueType.QCC,
                n_cycles_total=1,
                n_shoot_per_cycle=50,
                n_integration_steps=5,
                n_cycles_simultaneous=1,
                rmse_index=0,
            ),
            StudyInternal(
                name=r"$\condAlphaPe$",
                structure_type=StructureType.MUSCLE,
                fatigue_type=FatigueType.EFFORT_PERCEPTION,
                n_cycles_total=1,
                n_shoot_per_cycle=50,
                n_integration_steps=5,
                n_cycles_simultaneous=1,
                rmse_index=0,
            ),
        ),
    )

    # Actual Studies
    STUDY1_OCP: StudiesInternal = StudiesInternal(
        name="STUDY1_OCP",
        studies=(
            StudyInternal(
                name=r"$\condTauNf$",
                structure_type=StructureType.TAU,
                fatigue_type=FatigueType.NO_FATIGUE,
                n_cycles_total=1,
                n_shoot_per_cycle=50,
                n_integration_steps=5,
                n_cycles_simultaneous=1,
                rmse_index=0,
            ),
            StudyInternal(
                name=r"$\condTauQcc$",
                structure_type=StructureType.TAU,
                fatigue_type=FatigueType.QCC,
                n_cycles_total=1,
                n_shoot_per_cycle=50,
                n_integration_steps=5,
                n_cycles_simultaneous=1,
                rmse_index=0,
            ),
            StudyInternal(
                name=r"$\condTauPe$",
                structure_type=StructureType.TAU,
                fatigue_type=FatigueType.EFFORT_PERCEPTION,
                n_cycles_total=1,
                n_shoot_per_cycle=50,
                n_integration_steps=5,
                n_cycles_simultaneous=1,
                rmse_index=0,
            ),
            StudyInternal(
                name=r"$\condAlphaNf$",
                structure_type=StructureType.MUSCLE,
                fatigue_type=FatigueType.NO_FATIGUE,
                n_cycles_total=1,
                n_shoot_per_cycle=50,
                n_integration_steps=5,
                n_cycles_simultaneous=1,
                rmse_index=3,
            ),
            StudyInternal(
                name=r"$\condAlphaQcc$",
                structure_type=StructureType.MUSCLE,
                fatigue_type=FatigueType.QCC,
                n_cycles_total=1,
                n_shoot_per_cycle=50,
                n_integration_steps=5,
                n_cycles_simultaneous=1,
                rmse_index=3,
            ),
            StudyInternal(
                name=r"$\condAlphaPe$",
                structure_type=StructureType.MUSCLE,
                fatigue_type=FatigueType.EFFORT_PERCEPTION,
                n_cycles_total=1,
                n_shoot_per_cycle=50,
                n_integration_steps=5,
                n_cycles_simultaneous=1,
                rmse_index=3,
            ),
        ),
        figures=Figures(
            figures=(
                FigureOptions(
                    title=r"Évolution au cours du temps de la cible pour le degré de liberté $q_{13}$",
                    fcn=FiguresFcn.PLOT_TARGET_FUNCTION,
                    save_name="bow_target",
                    show_legend=False,
                ),
            ),
        ),
        latex_table=LatexTable(
            table_caption=(
                f"Comparaison des métriques d'efficacité entre les modèles de fatigue "
                f"appliqués sur une dynamique musculaire ou articulaire lors de la résolution d'un \\ocp{{}}"
            ),
            add_non_converged_notice=True,
            add_bfgs_dagger_notice=True,
            table_label="table:aller_retour_ocp",
            analyses=(
                LatexAnalysesFcn.NUMBER_OF_VAR_AND_CONSTRAINTS,
                LatexAnalysesFcn.MEAN_OPTIMIZATION_TIME,
                LatexAnalysesFcn.MEAN_NUMBER_ITERATIONS,
                LatexAnalysesFcn.MEAN_ITERATION_TIME,
                LatexAnalysesFcn.RMSE_Q,
            ),
        ),
    )

    STUDY2_TAU_10_CYCLES: StudiesInternal = StudiesInternal(
        name="STUDY2_TAU_10_CYCLES",
        studies=(
            StudyInternal(
                name=r"$\condTauNf$",
                structure_type=StructureType.TAU,
                fatigue_type=FatigueType.NO_FATIGUE,
                n_cycles_total=10,
                n_shoot_per_cycle=50,
                n_integration_steps=5,
                n_cycles_simultaneous=1,
                rmse_index=0,
                plot_options={"linewidth": 5}
            ),
            StudyInternal(
                name=r"$\condTauPe$",
                structure_type=StructureType.TAU,
                fatigue_type=FatigueType.EFFORT_PERCEPTION,
                n_cycles_total=10,
                n_shoot_per_cycle=50,
                n_integration_steps=5,
                n_cycles_simultaneous=1,
                rmse_index=0,
            ),
        ),
        latex_table=LatexTable(
            table_caption=(
                f"Comparaison des métriques d'efficacité entre les conditions $\\condTauNf$ et $\\condTauPe$ "
                f"lors d'un \\cyclicNMPC{{}} à $10$~cycles"
            ),
            add_non_converged_notice=True,
            add_bfgs_dagger_notice=True,
            table_label="table:nmpc_cyclic",
            analyses=(
                LatexAnalysesFcn.TOTAL_OPTIMIZATION_TIME,
                LatexAnalysesFcn.MEAN_NUMBER_ITERATIONS,
                LatexAnalysesFcn.MEAN_ITERATION_TIME,
                LatexAnalysesFcn.RMSE_Q,
            ),
        ),
        figures=Figures(
            figures=(
                FigureOptions(
                    title="",
                    fcn=FiguresFcn.DATA_IN_ONE_GO,
                    save_name="study2_clav_elev",
                    params={"data_type": DataType.STATES, "key": "q", "index": 1, "to_degree": True},
                ),
                FigureOptions(
                    title="",
                    fcn=FiguresFcn.DATA_IN_ONE_GO,
                    save_name="study2_scap_rotLat",
                    params={"data_type": DataType.STATES, "key": "q", "index": 2, "to_degree": True},
                ),
                FigureOptions(
                    title="",
                    fcn=FiguresFcn.DATA_IN_ONE_GO,
                    save_name="study2_humerus_abd",
                    params={"data_type": DataType.STATES, "key": "q", "index": 4, "to_degree": True},
                ),
            ),
            font_size=30,
        ),
        custom_analyses=CustomAnalyses((CustomAnalysesOption(CustomAnalysesFcn.PRINT_NUMBER_OF_ITERATIONS),)),
        extra_figures=ExtraFigures(
            extra_figures=(ExtraFigureOption(
                ExtraFiguresFcn.INITIAL_GUESS_NMPC,
                data_path="condTauPe_iterations/iteration_0000.bo"),
            ),
        ),
    )

    STUDY3_TAU_10_CYCLES_3_AT_A_TIME: StudiesInternal = StudiesInternal(
        name="STUDY3_TAU_10_CYCLES_3_AT_A_TIME",
        studies=(
            StudyInternal(
                name=r"$\condTauNf$",
                structure_type=StructureType.TAU,
                fatigue_type=FatigueType.NO_FATIGUE,
                n_cycles_total=10,
                n_shoot_per_cycle=30,
                n_integration_steps=3,
                n_cycles_simultaneous=3,
                rmse_index=0,
                plot_options={"linewidth": 5}
            ),
            StudyInternal(
                name=r"$\condTauPe$",
                structure_type=StructureType.TAU,
                fatigue_type=FatigueType.EFFORT_PERCEPTION,
                n_cycles_total=10,
                n_shoot_per_cycle=30,
                n_integration_steps=3,
                n_cycles_simultaneous=3,
                rmse_index=0,
            ),
        ),
        latex_table=LatexTable(
            table_caption=(
                f"Comparaison des métriques d'efficacité entre les conditions $\\condTauNf$ et $\\condTauPe$ "
                f"lors d'un \\multiCyclicNMPC{{}} à $3$~cycles simultanés sur un total de $10$~allers-retours"
            ),
            add_non_converged_notice=False,
            add_bfgs_dagger_notice=True,
            table_label="table:multi_cyclic_nmpc",
            analyses=(
                LatexAnalysesFcn.TOTAL_OPTIMIZATION_TIME,
                LatexAnalysesFcn.MEAN_NUMBER_ITERATIONS,
                LatexAnalysesFcn.MEAN_ITERATION_TIME,
                LatexAnalysesFcn.RMSE_Q,
            ),
        ),
        figures=Figures(
            figures=(
                FigureOptions(
                    title="",
                    fcn=FiguresFcn.DATA_IN_ONE_GO,
                    save_name="study3_clav_elev",
                    params={"data_type": DataType.STATES, "key": "q", "index": 1, "to_degree": True},
                ),
                FigureOptions(
                    title="",
                    fcn=FiguresFcn.DATA_IN_ONE_GO,
                    save_name="study3_scap_rotLat",
                    params={"data_type": DataType.STATES, "key": "q", "index": 2, "to_degree": True},
                ),
                FigureOptions(
                    title="",
                    fcn=FiguresFcn.DATA_IN_ONE_GO,
                    save_name="study3_humerus_abd",
                    params={"data_type": DataType.STATES, "key": "q", "index": 4, "to_degree": True},
                ),
                FigureOptions(
                    title="$q_{4-2}$ phase diagram",
                    fcn=FiguresFcn.PHASE_DIAGRAM,
                    params={"data_meta": ((DataType.STATES, "q", 4), (DataType.STATES, "q", 2)), "to_degree": True},
                ),
                FigureOptions(
                    title="$q_{4-1}$ phase diagram",
                    fcn=FiguresFcn.PHASE_DIAGRAM,
                    params={"data_meta": ((DataType.STATES, "q", 4), (DataType.STATES, "q", 1)), "to_degree": True},
                ),
                FigureOptions(
                    title="$q_{2-1}$ phase diagram",
                    fcn=FiguresFcn.PHASE_DIAGRAM,
                    params={"data_meta": ((DataType.STATES, "q", 2), (DataType.STATES, "q", 1)), "to_degree": True},
                ),
            ),
        ),
        custom_analyses=CustomAnalyses((CustomAnalysesOption(CustomAnalysesFcn.PRINT_NUMBER_OF_ITERATIONS),)),
    )

    STUDY4A_VIOLIN: StudiesInternal = StudiesInternal(
        name="STUDY4_VIOLIN",
        studies=(
            StudyInternal(
                name=r"$\condTauNf$",
                structure_type=StructureType.TAU,
                fatigue_type=FatigueType.NO_FATIGUE,
                n_cycles_total=600,
                n_shoot_per_cycle=30,
                n_integration_steps=3,
                n_cycles_simultaneous=3,
                rmse_index=0,
                plot_options={"linewidth": 5}
            ),
            StudyInternal(
                name=r"$\condTauPe$",
                save_name=r"$\condTauPe\ (80\%)$",
                structure_type=StructureType.TAU,
                fatigue_type=FatigueType.EFFORT_PERCEPTION,
                n_cycles_total=600,
                n_shoot_per_cycle=30,
                n_integration_steps=3,
                n_cycles_simultaneous=3,
                rmse_index=0,
            ),
        ),
        latex_table=LatexTable(
            table_caption=(
                f"Comparaison des métriques d'efficacité entre les conditions $\\condTauNf$ et $\\condTauPe$ "
                f"lors d'un mouvement de violon synthétisé par \\multiCyclicNMPC{{}} à $3$~cycles simultanés "
                f"sur un total de $600$~allers-retours"
            ),
            add_non_converged_notice=False,
            add_bfgs_dagger_notice=True,
            table_label="table:violin_nmpc",
            analyses=(
                LatexAnalysesFcn.TOTAL_OPTIMIZATION_TIME,
                LatexAnalysesFcn.MEAN_NUMBER_ITERATIONS,
                LatexAnalysesFcn.MEAN_ITERATION_TIME,
                LatexAnalysesFcn.RMSE_Q,
            ),
        ),
        custom_analyses=CustomAnalyses(
            analyses=(
                CustomAnalysesOption(CustomAnalysesFcn.RMSE, last_cycle=450),
                CustomAnalysesOption(CustomAnalysesFcn.RMSE, first_cycle=510),
                CustomAnalysesOption(CustomAnalysesFcn.OBJECTIVE_FUNCTION, cycle=50),
                CustomAnalysesOption(CustomAnalysesFcn.OBJECTIVE_FUNCTION, cycle=550),
            ),
        ),
        figures=Figures(
            figures=(
                FigureOptions(
                    # title="Évolution au cours du temps pour le bassin\n"
                    #       "de fatigue positif de $\\tau_1$ pour tous les cycles",
                    title="",
                    fcn=FiguresFcn.INTEGRATION_FROM_ANOTHER_DYNAMICS,
                    save_name="study4_fatigue_m1_full",
                    params={
                        "dynamics_source_idx": 1,
                        "key": "tau_plus_mf",
                        "index": 1,
                        "is_fatigue": True,
                        "ylim": (0, 100),
                    },
                ),
                FigureOptions(
                    # title="Évolution au cours du temps pour le bassin\n"
                    #       "de fatigue positif de $\\tau_1$ pour les cycles de $450$ à la fin",
                    title="",
                    fcn=FiguresFcn.INTEGRATION_FROM_ANOTHER_DYNAMICS,
                    save_name="study4_fatigue_m1_from_450",
                    params={
                        "dynamics_source_idx": 1,
                        "key": "tau_plus_mf",
                        "index": 1,
                        "is_fatigue": True,
                        "first_cycle": 450,
                    },
                ),
                FigureOptions(
                    # title="Évolution au cours du temps pour le bassin\n"
                    #       "de fatigue négatif de $\\tau_2$ pour tous les cycles",
                    title="",
                    fcn=FiguresFcn.INTEGRATION_FROM_ANOTHER_DYNAMICS,
                    save_name="study4_fatigue_m2_full",
                    params={
                        "dynamics_source_idx": 1,
                        "key": "tau_minus_mf",
                        "index": 2,
                        "is_fatigue": True,
                        "ylim": (0, 100),
                    },
                ),
                FigureOptions(
                    # title="Évolution au cours du temps pour le bassin\n"
                    #       "de fatigue négatif de $\\tau_2$ pour les cycles de $450$ à la fin",
                    title="",
                    fcn=FiguresFcn.INTEGRATION_FROM_ANOTHER_DYNAMICS,
                    save_name="study4_fatigue_m2_from_450",
                    params={
                        "dynamics_source_idx": 1,
                        "key": "tau_minus_mf",
                        "index": 2,
                        "is_fatigue": True,
                        "first_cycle": 450,
                    },
                ),
                FigureOptions(
                    # title="Évolution au cours du temps pour le bassin\n"
                    #       "de fatigue négatif de $\\tau_5$ pour tous les cycles",
                    title="",
                    fcn=FiguresFcn.INTEGRATION_FROM_ANOTHER_DYNAMICS,
                    save_name="study4_fatigue_m5_full",
                    params={
                        "dynamics_source_idx": 1,
                        "key": "tau_minus_mf",
                        "index": 5,
                        "is_fatigue": True,
                        "ylim": (0, 100),
                    },
                ),
                FigureOptions(
                    # title="Évolution au cours du temps pour le bassin\n"
                    #       "de fatigue négatif de $\\tau_5$ pour les cycles de $450$ à la fin",
                    title="",
                    fcn=FiguresFcn.INTEGRATION_FROM_ANOTHER_DYNAMICS,
                    save_name="study4_fatigue_m5_from_450",
                    params={
                        "dynamics_source_idx": 1,
                        "key": "tau_minus_mf",
                        "index": 5,
                        "is_fatigue": True,
                        "first_cycle": 450,
                    },
                ),
                FigureOptions(
                    # title="Superposition des cycles $10$ à $500$ de l'évolution de $q_1$ au cours du temps",
                    title="",
                    fcn=FiguresFcn.DATA_STACKED_PER_CYCLE,
                    save_name="study4_q_1_from_10_to_500",
                    params={
                        "data_type": DataType.STATES,
                        "key": "q",
                        "index": 1,
                        "to_degree": True,
                        "first_cycle": 10,
                        "last_cycle": 500,
                    },
                    use_subplots=False,
                ),
                FigureOptions(
                    # title="Superposition des cycles $10$ à $500$ de l'évolution de $q_1$ au cours du temps",
                    title="",
                    fcn=FiguresFcn.DATA_STACKED_PER_CYCLE,
                    save_name="study4_tau_1_from_10_to_500",
                    params={
                        "data_type": DataType.CONTROLS,
                        "key": "tau",
                        "index": 1,
                        "to_degree": False,
                        "first_cycle": 10,
                        "last_cycle": 500,
                    },
                    use_subplots=False,
                ),
                FigureOptions(
                    # title="Superposition des cycles $500$ jusqu'à final de l'évolution de $q_1$ au cours du temps",
                    title="",
                    fcn=FiguresFcn.DATA_STACKED_PER_CYCLE,
                    save_name="study4_q_1_from_500",
                    params={
                        "data_type": DataType.STATES,
                        "key": "q",
                        "index": 1,
                        "to_degree": True,
                        "first_cycle": 500,
                    },
                    use_subplots=False,
                ),
                FigureOptions(
                    # title="Superposition des cycles $500$ jusqu'à final de la commande de $\\tau_1$ au cours du temps",
                    title="",
                    fcn=FiguresFcn.DATA_STACKED_PER_CYCLE,
                    save_name="study4_tau_1_from_500",
                    params={
                        "data_type": DataType.CONTROLS,
                        "key": "tau",
                        "index": 1,
                        "to_degree": False,
                        "first_cycle": 500,
                    },
                    use_subplots=False,
                ),
                FigureOptions(
                    # title="Superposition des cycles $500$ jusqu'à final de l'évolution de $q_2$ au cours du temps",
                    title="",
                    fcn=FiguresFcn.DATA_STACKED_PER_CYCLE,
                    save_name="study4_q_2_from_500",
                    params={
                        "data_type": DataType.STATES,
                        "key": "q",
                        "index": 2,
                        "to_degree": True,
                        "first_cycle": 500,
                    },
                    use_subplots=False,
                ),
                FigureOptions(
                    # title="Superposition des cycles $500$ jusqu'à final de la commande de $\\tau_2$ au cours du temps",
                    title="",
                    fcn=FiguresFcn.DATA_STACKED_PER_CYCLE,
                    save_name="study4_tau_2_from_500",
                    params={
                        "data_type": DataType.CONTROLS,
                        "key": "tau",
                        "index": 2,
                        "to_degree": False,
                        "first_cycle": 500,
                    },
                    use_subplots=False,
                ),
                FigureOptions(
                    ##title="Superposition des cycles $500$ jusqu'à final de l'évolution de $q_4$ au cours du temps",
                    title="",
                    fcn=FiguresFcn.DATA_STACKED_PER_CYCLE,
                    save_name="study4_q_4_from_500",
                    params={
                        "data_type": DataType.STATES,
                        "key": "q",
                        "index": 4,
                        "to_degree": True,
                        "first_cycle": 500,
                    },
                    use_subplots=False,
                ),
                FigureOptions(
                    # title="Superposition des cycles $500$ jusqu'à final de la commande de $\\tau_4$ au cours du temps",
                    title="",
                    fcn=FiguresFcn.DATA_STACKED_PER_CYCLE,
                    save_name="study4_tau_4_from_500",
                    params={
                        "data_type": DataType.CONTROLS,
                        "key": "tau",
                        "index": 4,
                        "to_degree": False,
                        "first_cycle": 500,
                    },
                    use_subplots=False,
                ),
                FigureOptions(
                    ##title="Superposition des cycles $500$ jusqu'à final de l'évolution de $q_5$ au cours du temps",
                    title="",
                    fcn=FiguresFcn.DATA_STACKED_PER_CYCLE,
                    save_name="study4_q_5_from_500",
                    params={
                        "data_type": DataType.STATES,
                        "key": "q",
                        "index": 5,
                        "to_degree": True,
                        "first_cycle": 500,
                    },
                    use_subplots=False,
                ),
                FigureOptions(
                    # title="Superposition des cycles $500$ jusqu'à final de la commande de $\\tau_5$ au cours du temps",
                    title="",
                    fcn=FiguresFcn.DATA_STACKED_PER_CYCLE,
                    save_name="study4_tau_5_from_500",
                    params={
                        "data_type": DataType.CONTROLS,
                        "key": "tau",
                        "index": 5,
                        "to_degree": False,
                        "first_cycle": 500,
                    },
                    use_subplots=False,
                ),
                FigureOptions(
                    ##title="Superposition des cycles $500$ jusqu'à final de l'évolution de $q_6$ au cours du temps",
                    title="",
                    fcn=FiguresFcn.DATA_STACKED_PER_CYCLE,
                    save_name="study4_q_6_from_500",
                    params={
                        "data_type": DataType.STATES,
                        "key": "q",
                        "index": 6,
                        "to_degree": True,
                        "first_cycle": 500,
                    },
                    use_subplots=False,
                ),
                FigureOptions(
                    # title="Superposition des cycles $500$ jusqu'à final de la commande de $\\tau_6$ au cours du temps",
                    title="",
                    fcn=FiguresFcn.DATA_STACKED_PER_CYCLE,
                    save_name="study4_tau_6_from_500",
                    params={
                        "data_type": DataType.CONTROLS,
                        "key": "tau",
                        "index": 6,
                        "to_degree": False,
                        "first_cycle": 500,
                    },
                    use_subplots=False,
                ),
            ),
        ),
        videos=Videos(
            cycle_in_and_out=((500, 507),),
            camera_name_pos_roll=(
                ("front", (3, 0, 0), 0),
                ("top", (0.5, 3, 0), 0),
                ("side", (2, 0, 3), 0),
            ),
        ),
        snapshots=Videos(
            cycle_in_and_out=((506, 507),),
            camera_name_pos_roll=(
                ("front", (3, 0, 0), 0),
            ),
        ),
        extra_figures=ExtraFigures(
            extra_figures=(
                ExtraFigureOption(
                    ExtraFiguresFcn.BOW,
                ),
                ExtraFigureOption(
                    ExtraFiguresFcn.VIOLIN,
                ),
                ExtraFigureOption(
                    ExtraFiguresFcn.SHOW_MODEL_TOGGLE_SEGMENT,
                    save_name="violin_tip_NoFatigue.png",
                    idx_solution=0,
                    cycle=550,
                    half_cycle=False,
                    toggle_idx=[],
                    camera_name_pos_roll=(
                        ("front", (3, 0, 0), 0),
                        ("top", (0.5, 3, 0), 0),
                        ("side", (2, 0, 3), 0),
                    ),
                ),
                ExtraFigureOption(
                    ExtraFiguresFcn.SHOW_MODEL_TOGGLE_SEGMENT,
                    save_name="violin_tip_Fatigue80.png",
                    idx_solution=1,
                    cycle=550,
                    half_cycle=False,
                    toggle_idx=[1, 0] + list(range(13, 29)),
                    camera_name_pos_roll=(
                        ("front", (3, 0, 0), 0),
                        ("top", (0.5, 3, 0), 0),
                        ("side", (2, 0, 3), 0),
                    ),
                ),
            ),
        ),
    )

    STUDY4B_VIOLIN: StudiesInternal = StudiesInternal(
        name="STUDY4_VIOLIN",
        studies=(
            StudyInternal(
                name=r"$\condTauNf$",
                structure_type=StructureType.TAU,
                fatigue_type=FatigueType.NO_FATIGUE,
                n_cycles_total=600,
                n_shoot_per_cycle=30,
                n_integration_steps=3,
                n_cycles_simultaneous=3,
                rmse_index=0,
                plot_options={"linewidth": 5}
            ),
            StudyInternal(
                name=r"$\condTauPe\ (80\%)$",
                structure_type=StructureType.TAU,
                fatigue_type=FatigueType.EFFORT_PERCEPTION,
                n_cycles_total=600,
                n_shoot_per_cycle=30,
                n_integration_steps=3,
                n_cycles_simultaneous=3,
                rmse_index=0,
                fatigue_max_threshold=0.8,
            ),
            StudyInternal(
                name=r"$\condTauPe\ (70\%)$",
                structure_type=StructureType.TAU,
                fatigue_type=FatigueType.EFFORT_PERCEPTION,
                n_cycles_total=600,
                n_shoot_per_cycle=30,
                n_integration_steps=3,
                n_cycles_simultaneous=3,
                rmse_index=0,
                fatigue_max_threshold=0.7,
            ),
        ),
        custom_analyses=CustomAnalyses(
            analyses=(
                CustomAnalysesOption(CustomAnalysesFcn.OBJECTIVE_FUNCTION, cycle=50),
                CustomAnalysesOption(CustomAnalysesFcn.OBJECTIVE_FUNCTION, cycle=550),
            ),
        ),
        figures=Figures(
            figures=(
                FigureOptions(
                    # title="Évolution au cours du temps pour le bassin\n"
                    #       "de fatigue négatif de $\\tau_5$ pour tous les cycles",
                    title="",
                    fcn=FiguresFcn.INTEGRATION_FROM_ANOTHER_DYNAMICS,
                    save_name="study4b_fatigue_m5_full",
                    params={
                        "dynamics_source_idx": 1,
                        "key": "tau_minus_mf",
                        "index": 5,
                        "is_fatigue": True,
                        "ylim": (0, 100),
                    },
                ),
                FigureOptions(
                    # title="Superposition des cycles $550$ jusqu'à final de l'évolution de $q_4$ au cours du temps",
                    title="",
                    fcn=FiguresFcn.DATA_STACKED_PER_CYCLE,
                    save_name="study4b_q_4_from_550",
                    params={
                        "data_type": DataType.STATES,
                        "key": "q",
                        "index": 4,
                        "to_degree": True,
                        "first_cycle": 550,
                    },
                    use_subplots=False,
                ),
                FigureOptions(
                    ##title="Superposition des cycles $550$ jusqu'à final de l'évolution de $q_5$ au cours du temps",
                    title="",
                    fcn=FiguresFcn.DATA_STACKED_PER_CYCLE,
                    save_name="study4b_q_5_from_550",
                    params={
                        "data_type": DataType.STATES,
                        "key": "q",
                        "index": 5,
                        "to_degree": True,
                        "first_cycle": 550,
                    },
                    use_subplots=False,
                ),
            ),
        ),
        videos=Videos(
            cycle_in_and_out=((550, 557),),
            camera_name_pos_roll=(
                ("front", (3, 0, 0), 0),
                ("top", (0.5, 3, 0), 0),
                ("side", (2, 0, 3), 0),
            ),
        ),
        extra_figures=ExtraFigures(
            extra_figures=(
                ExtraFigureOption(
                    ExtraFiguresFcn.SHOW_MODEL_TOGGLE_SEGMENT,
                    save_name="violin_frog_NoFatigue.png",
                    idx_solution=0,
                    cycle=550,
                    half_cycle=True,
                    toggle_idx=[],
                    camera_name_pos_roll=(
                        ("front", (3, 0, 0), 0),
                        ("top", (0.5, 3, 0), 0),
                        ("side", (2, 0, 3), 0),
                    ),
                ),
                ExtraFigureOption(
                    ExtraFiguresFcn.SHOW_MODEL_TOGGLE_SEGMENT,
                    save_name="violin_frog_Fatigue80.png",
                    idx_solution=1,
                    cycle=550,
                    half_cycle=True,
                    toggle_idx=[1, 0] + list(range(13, 29)),
                    camera_name_pos_roll=(
                        ("front", (3, 0, 0), 0),
                        ("top", (0.5, 3, 0), 0),
                        ("side", (2, 0, 3), 0),
                    ),
                ),
                ExtraFigureOption(
                    ExtraFiguresFcn.SHOW_MODEL_TOGGLE_SEGMENT,
                    save_name="violin_frog_Fatigue70.png",
                    idx_solution=2,
                    cycle=550,
                    half_cycle=True,
                    toggle_idx=[1, 0] + list(range(13, 29)),
                    camera_name_pos_roll=(
                        ("front", (3, 0, 0), 0),
                        ("top", (0.5, 3, 0), 0),
                        ("side", (2, 0, 3), 0),
                    ),
                ),
            ),
        ),
    )
