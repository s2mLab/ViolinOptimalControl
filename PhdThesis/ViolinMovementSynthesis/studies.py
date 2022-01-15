import os
import statistics
from typing import Union

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
    DataType,
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
    ):
        self.name = name
        self.save_name = name.replace("$", "")
        self.save_name = self.save_name.replace("\\", "")
        self.rmse_index = rmse_index

        self.model_name: str = "WuViolin"
        self.violin: Violin = Violin(self.model_name, ViolinString.G)
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
        # if sol is not None:
        #     nmpc_violin.save(sol, ext=f"tmp_{save_name}_{t}", stand_alone=True)
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

    def perform(self, limit_memory_max_iter: int, exact_max_iter: int, show_graphs: bool):
        if not self._is_initialized:
            self._initialize_nmpc()

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
    def __init__(self, name: str, studies: tuple[StudyInternal, ...]):
        self.name = name
        self._has_run = False
        self.studies = studies
        self.solutions: list[tuple[Solution, list[Solution]], ...] = []

    def perform(self, limit_memory_max_iter: int = 100, exact_max_iter: int = 1000, show_graphs: bool = False):
        self.solutions: list[tuple[Solution, list[Solution]], ...] = []
        for study in self.studies:
            self.solutions.append(study.perform(limit_memory_max_iter, exact_max_iter, show_graphs))
        self._has_run = True

    def save_solutions(self):
        for study, (sol, all_iterations) in zip(self.studies, self.solutions):
            study.nmpc.save(sol, save_path=f"{self._prepare_and_get_results_dir()}/{study.save_name}")
            study.nmpc.save(sol, save_path=f"{self._prepare_and_get_results_dir()}/{study.save_name}", stand_alone=True)

            for i, iterations in enumerate(all_iterations):
                study.nmpc.save(
                    iterations,
                    save_path=f"{self._prepare_and_get_results_dir()}/{study.save_name}_iterations/iteration_{i:04d}"
                )

    def generate_latex_table(self):
        if not self._has_run:
            raise RuntimeError("run() must be called before generating the latex table")

        table = (
            f"\\documentclass{{article}}\n"
            f"\n"
            f"\\usepackage{{amsmath}}\n"
            f"\\usepackage{{amssymb}}\n"
            f"\\usepackage[table]{{xcolor}}\n"
            f"\\usepackage{{makecell}}\n"
            f"\\definecolor{{lightgray}}{{gray}}{{0.91}}\n"
            f"\n\n"
            f"% Aliases\n"
            f"\\newcommand{{\\rmse}}{{RMSE}}\n"
            f"\\newcommand{{\\ocp}}{{OCP}}\n"
            f"\\newcommand{{\\controls}}{{\\mathbf{{u}}}}\n"
            f"\\newcommand{{\\states}}{{\\mathbf{{x}}}}\n"
            f"\\newcommand{{\\statesDot}}{{\\mathbf{{\\dot{{x}}}}}}\n"
            f"\\newcommand{{\\q}}{{\\mathbf{{q}}}}\n"
            f"\\newcommand{{\\qdot}}{{\\mathbf{{\\dot{{q}}}}}}\n"
            f"\\newcommand{{\\qddot}}{{\\mathbf{{\\ddot{{q}}}}}}\n"
            f"\\newcommand{{\\f}}{{\\mathbf{{f}}}}\n"
            f"\\newcommand{{\\taupm}}{{\\tau^{{\\pm}}}}\n"
            f"\\newcommand{{\\tauns}}{{\\tau^{{\\times}}}}\n"
            f"\n"
            f"\\newcommand{{\\condition}}{{C/}}\n"
            f"\\newcommand{{\\noFatigue}}{{\\varnothing}}\n"
            f"\\newcommand{{\\qcc}}{{4\\textsubscript{{CC}}}}\n"
            f"\\newcommand{{\\pe}}{{P\\textsubscript{{E}}}}\n"
            f"\\newcommand{{\\condTau}}{{{{\\condition}}{{\\tau}}{{}}}}\n"
            f"\\newcommand{{\\condTauNf}}{{{{\\condition}}{{\\tau}}{{\\noFatigue}}}}\n"
            f"\\newcommand{{\\condTauQcc}}{{{{\\condition}}{{\\tau}}{{\\qcc}}}}\n"
            f"\\newcommand{{\\condTauPe}}{{{{\\condition}}{{\\tau}}{{\\pe}}}}\n"
            f"\\newcommand{{\\condTaupm}}{{{{\\condition}}{{\\taupm}}{{}}}}\n"
            f"\\newcommand{{\\condTaupmQcc}}{{{{\\condition}}{{\\taupm}}{{\\qcc}}}}\n"
            f"\\newcommand{{\\condTaupmPe}}{{{{\\condition}}{{\\taupm}}{{\\pe}}}}\n"
            f"\\newcommand{{\\condTauns}}{{{{\\condition}}{{\\tauns}}{{}}}}\n"
            f"\\newcommand{{\\condTaunsQcc}}{{{{\\condition}}{{\\tauns}}{{\\qcc}}}}\n"
            f"\\newcommand{{\\condTaunsPe}}{{{{\\condition}}{{\\tauns}}{{\\pe}}}}\n"
            f"\\newcommand{{\\condAlpha}}{{{{\\condition}}{{\\alpha}}{{}}}}\n"
            f"\\newcommand{{\\condAlphaNf}}{{{{\\condition}}{{\\alpha}}{{\\noFatigue}}}}\n"
            f"\\newcommand{{\\condAlphaQcc}}{{{{\\condition}}{{\\alpha}}{{\\qcc}}}}\n"
            f"\\newcommand{{\\condAlphaPe}}{{{{\\condition}}{{\\alpha}}{{\\pe}}}}\n"
            f"\n\n"
            f"\\begin{{document}}\n"
            f"\n"
            f"\\begin{{table}}[!ht]\n"
            f" \\rowcolors{{1}}{{}}{{lightgray}}\n"
            f" \\caption{{Comparaison des métriques d'efficacité entre les modèles de fatigue "
            f"appliqués sur une dynamique musculaire ou articulaire lors de la résolution d'un \\ocp{{}}}}\n"
            f" \\label{{table:faisabilite}}\n"
            f" \\begin{{tabular}}{{lcccc}}\n"
            f"  \\hline\n"
            f"  \\bfseries Condition & "
            f"\\bfseries\\makecell[c]{{Nombre\\\\d'itération}} & "
            f"\\bfseries\\makecell[c]{{Temps\\\\d'optimisation\\\\(s)}} & "
            f"\\bfseries\\makecell[c]{{Temps moyen\\\\par itération\\\\(s/iteration)}} & "
            f"\\bfseries\\makecell[c]{{$\\sum\\text{{\\rmse{{}}}}$\\\\pour $\\q$\\\\(rad)}}\\\\ \n"
            f"  \\hline\n"
        )

        for study, (sol, all_iterations) in zip(self.studies, self.solutions):
            mean_iterations = statistics.mean([iteration.iterations for iteration in all_iterations])

            rmse = np.sum(self._rmse(DataType.STATES, "q", study.rmse_index, sol))
            rmse_str = f"{rmse:0.3e}" if rmse != 0 else "---"
            if rmse_str.find("e") >= 0:
                rmse_str = rmse_str.replace("e", "$\\times 10^{{")
                rmse_str += "}}$"
                rmse_str = rmse_str.replace("+0", "")
                rmse_str = rmse_str.replace("-0", "-")
                rmse_str = rmse_str.replace("$\\times 10^{{0}}$", "")
            table += (
                f"  {study.name} "
                f"& {mean_iterations} "
                f"& {sol.real_time_to_optimize:0.3f} "
                f"& {sol.real_time_to_optimize / mean_iterations:0.3f} "
                f"& {rmse_str} \\\\\n"
            )

        table += f"  \\hline\n" f" \\end{{tabular}}\n" f"\\end{{table}}\n\n"
        table += f"\\end{{document}}\n"

        save_path = f"{self._prepare_and_get_results_dir()}/results.tex"
        with open(save_path, "w", encoding='utf8') as file:
            file.write(table)
        print("\n\nTex file generated in the results folder")

    def _rmse(self, data_type, key, idx_ref: int, sol: Solution):
        data_ref = getattr(self.solutions[idx_ref][0], data_type.value)[key]
        data = getattr(sol, data_type.value)[key]

        e = data_ref - data
        se = e ** 2
        mse = np.sum(se, axis=1) / data_ref.shape[1]
        rmse = np.sqrt(mse)
        return rmse

    def _prepare_and_get_results_dir(self):
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
                name=r"$\condTauNf$",
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
    )

    STUDY2_TAU_10_CYCLES: StudiesInternal = StudiesInternal(
        name="DEBUG_TAU_PE_FATIGUE_3_CYCLES",
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
    )

    STUDY3_TAU_10_CYCLES_3_AT_A_TIME: StudiesInternal = StudiesInternal(
        name="STUDY2_TAU_10_CYCLES_3_AT_A_TIME",
        studies=(
            StudyInternal(
                name=r"$\condTauNf$",
                structure_type=StructureType.TAU,
                fatigue_type=FatigueType.NO_FATIGUE,
                n_cycles_total=4,
                n_shoot_per_cycle=30,
                n_integration_steps=3,
                n_cycles_simultaneous=3,
                rmse_index=0,
            ),
            StudyInternal(
                name=r"$\condTauPe$",
                structure_type=StructureType.TAU,
                fatigue_type=FatigueType.EFFORT_PERCEPTION,
                n_cycles_total=4,
                n_shoot_per_cycle=30,
                n_integration_steps=3,
                n_cycles_simultaneous=3,
                rmse_index=0,
            ),
        ),
    )

    STUDY4_VIOLIN: StudiesInternal = StudiesInternal(
        name="STUDY2_TAU_10_CYCLES_3_AT_A_TIME",
        studies=(
            StudyInternal(
                name=r"$\condTauNf$",
                structure_type=StructureType.TAU,
                fatigue_type=FatigueType.NO_FATIGUE,
                n_cycles_total=900,
                n_shoot_per_cycle=30,
                n_integration_steps=3,
                n_cycles_simultaneous=3,
                rmse_index=0,
            ),
            StudyInternal(
                name=r"$\condTauPe$",
                structure_type=StructureType.TAU,
                fatigue_type=FatigueType.EFFORT_PERCEPTION,
                n_cycles_total=900,
                n_shoot_per_cycle=30,
                n_integration_steps=3,
                n_cycles_simultaneous=3,
                rmse_index=0,
            ),
        ),
    )
