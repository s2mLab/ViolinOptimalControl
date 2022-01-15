import os
from typing import Union
from enum import Enum

from bioptim import ObjectiveList, ConstraintList, OdeSolver, Solver, ObjectiveFcn, Solution
import numpy as np

from matplotlib import pyplot as plt, colors as mcolors

from feasability_study_ocp import (
    DynamicsFcn,
    FatigableStructure,
    OcpConfiguration,
    FatigueModels,
    FatigueParameters,
    PlotOptions,
    DataType,
)


class StudySetup:
    def __init__(
        self,
        n_shoot: int = 50,
        final_time: float = 1,
        x0: tuple[float, ...] = (0.07, 1.4, 0, 0),
        tau_limits_no_muscles: tuple[float, float] = (-10, 10),
        tau_limits_with_muscles: tuple[float, float] = (-1, 1),
        weight_fatigue: float = 1_000,
        split_controls: bool = False,
        ode_solver: OdeSolver = None,
        solver: Solver = None,
        use_sx: bool = False,
        n_thread: int = 8,
    ):
        self.n_shoot = n_shoot
        self.final_time = final_time
        self.x0 = x0
        self.tau_limits_no_muscles = tau_limits_no_muscles
        self.tau_limits_with_muscles = tau_limits_with_muscles
        self.weight_fatigue = weight_fatigue
        self.split_controls = split_controls
        self.ode_solver = OdeSolver.RK4() if ode_solver is None else ode_solver
        self.solver = solver
        if self.solver is None:
            self.solver = Solver.IPOPT(
                show_online_optim=False,
                _print_level=5,
                _linear_solver="ma57",
                _hessian_approximation="exact",
                _max_iter=1000,
            )
        self.use_sx = use_sx
        self.n_thread = n_thread


class StudyInternal:
    @staticmethod
    def torque_driven_no_fatigue(study_setup: StudySetup):
        fatigue_model = FatigueModels.NONE

        objectives = ObjectiveList()
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)

        return OcpConfiguration(
            name=r"$\condTauNf$",
            model_path="models/arm26.bioMod",
            n_shoot=study_setup.n_shoot,
            final_time=study_setup.final_time,
            x0=study_setup.x0,
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
            name=r"$\condAlphaNf$",
            model_path="models/arm26.bioMod",
            n_shoot=study_setup.n_shoot,
            final_time=study_setup.final_time,
            x0=study_setup.x0,
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
            FatigueParameters(scaling=study_setup.tau_limits_no_muscles[1], split_controls=study_setup.split_controls),
        )

        objectives = ObjectiveList()
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="tau_minus_mf", weight=study_setup.weight_fatigue)
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="tau_plus_mf", weight=study_setup.weight_fatigue)
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="tau_minus_mf_xia", weight=study_setup.weight_fatigue)
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="tau_plus_mf_xia", weight=study_setup.weight_fatigue)

        return OcpConfiguration(
            name=r"$\condTaupmQcc$" if study_setup.split_controls else r"$\condTaunsQcc$",
            model_path="models/arm26.bioMod",
            n_shoot=study_setup.n_shoot,
            final_time=study_setup.final_time,
            x0=study_setup.x0,
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
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="muscles_mf", weight=study_setup.weight_fatigue)
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="muscles_mf_xia", weight=study_setup.weight_fatigue)

        return OcpConfiguration(
            name=r"$\condAlphaQcc$",
            model_path="models/arm26.bioMod",
            n_shoot=study_setup.n_shoot,
            final_time=study_setup.final_time,
            x0=study_setup.x0,
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
            FatigueParameters(scaling=study_setup.tau_limits_no_muscles[1], split_controls=study_setup.split_controls),
        )

        objectives = ObjectiveList()
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=1)
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_FATIGUE, key="tau_minus", weight=study_setup.weight_fatigue)
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_FATIGUE, key="tau_plus", weight=study_setup.weight_fatigue)

        return OcpConfiguration(
            name=r"$\condTaupmPe$" if study_setup.split_controls else r"$\condTaunsPe$",
            model_path="models/arm26.bioMod",
            n_shoot=study_setup.n_shoot,
            final_time=study_setup.final_time,
            x0=study_setup.x0,
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
        objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_FATIGUE, key="muscles", weight=study_setup.weight_fatigue)

        return OcpConfiguration(
            name=r"$\condAlphaPe$",
            model_path="models/arm26.bioMod",
            n_shoot=study_setup.n_shoot,
            final_time=study_setup.final_time,
            x0=study_setup.x0,
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


class StudyConfiguration:
    def __init__(
        self,
        studies: tuple[OcpConfiguration, ...],
        rmse_index: Union[tuple[int, ...], None],
        plot_options: PlotOptions,
    ):
        self.studies = studies
        self.rmse_index = rmse_index
        self.plot_options = plot_options if plot_options is None else plot_options
        if len(self.plot_options.options) < len(self.studies):
            raise ValueError("len(plot_options.options) must be >= len(studies)")


class Conditions(Enum):
    DEBUG_FAST = StudyConfiguration(
        studies=(StudyInternal.torque_driven_michaud(StudySetup(split_controls=False)),),
        rmse_index=None,
        plot_options=PlotOptions(
            title="%s pour les conditions $C/\\tau\\varnothing$  et $C/\\alpha\\varnothing$",
            legend_indices=None,
            options=({"linestyle": "-"}, {"linestyle": "--"}),
        ),
    )

    DEBUG_ALL_CONDITIONS = StudyConfiguration(
        studies=(
            StudyInternal.torque_driven_no_fatigue(StudySetup()),
            StudyInternal.muscles_driven_no_fatigue(StudySetup()),
            StudyInternal.torque_driven_michaud(StudySetup(split_controls=True)),
            StudyInternal.torque_driven_michaud(StudySetup(split_controls=False)),
            StudyInternal.muscle_driven_michaud(StudySetup()),
            StudyInternal.torque_driven_effort_perception(StudySetup(split_controls=True)),
            StudyInternal.torque_driven_effort_perception(StudySetup(split_controls=False)),
            StudyInternal.muscle_driven_effort_perception(StudySetup()),
        ),
        rmse_index=None,
        plot_options=PlotOptions(
            title="Fast debugger",
            legend_indices=None,
            options=(
                {"linestyle": "-"},
                {"linestyle": "--"},
                {"linestyle": "-"},
                {"linestyle": "--"},
                {"linestyle": "-"},
                {"linestyle": "--"},
                {"linestyle": "-"},
                {"linestyle": "--"},
                {"linestyle": "-"},
                {"linestyle": "--"},
            ),
        ),
    )

    STUDY1 = StudyConfiguration(
        studies=(
            StudyInternal.torque_driven_no_fatigue(StudySetup()),
            StudyInternal.torque_driven_michaud(StudySetup(split_controls=True)),
            StudyInternal.torque_driven_effort_perception(StudySetup(split_controls=True)),
            StudyInternal.torque_driven_michaud(StudySetup(split_controls=False)),
            StudyInternal.torque_driven_effort_perception(StudySetup(split_controls=False)),
            StudyInternal.muscles_driven_no_fatigue(StudySetup()),
            StudyInternal.muscle_driven_michaud(StudySetup()),
            StudyInternal.muscle_driven_effort_perception(StudySetup()),
        ),
        rmse_index=(0, 0, 0, 0, 0, 5, 5, 5),
        plot_options=PlotOptions(
            title="Degré de liberté %s en fonction du temps pour toutes les conditions",
            legend_indices=(
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ),
            options=(
                {"linestyle": "-", "color": mcolors.CSS4_COLORS["black"]},
                {"linestyle": "-", "color": mcolors.CSS4_COLORS["lightcoral"]},
                {"linestyle": "-", "color": mcolors.CSS4_COLORS["cornflowerblue"]},
                {"linestyle": "-", "color": mcolors.CSS4_COLORS["red"]},
                {"linestyle": "-", "color": mcolors.CSS4_COLORS["blue"]},
                {"linestyle": "--", "color": mcolors.CSS4_COLORS["black"]},
                {"linestyle": "--", "color": mcolors.CSS4_COLORS["red"]},
                {"linestyle": "--", "color": mcolors.CSS4_COLORS["blue"]},
            ),
            maximize=False,
            save_path=("feasibility_q0", "feasibility_q1"),
        ),
    )


class Study:
    def __init__(self, conditions: Conditions):
        self.name = conditions.name
        self._has_run: bool = False
        self._plots_are_prepared: bool = False
        self.conditions: StudyConfiguration = conditions.value
        self.solution: list[Solution, ...] = []

    def run(self):
        for condition in self.conditions.studies:
            self.solution.append(condition.perform())
        self._has_run = True

    def print_results(self):
        print("Number of iterations")
        for study, sol in zip(self.conditions.studies, self.solution):
            print(f"\t{study.name} = {sol.iterations}")

        print("Total time to optimize")
        for study, sol in zip(self.conditions.studies, self.solution):
            print(f"\t{study.name} = {sol.real_time_to_optimize:0.3f} second")

        print("Mean time per iteration to optimize")
        for study, sol in zip(self.conditions.studies, self.solution):
            print(f"\t{study.name} = {sol.real_time_to_optimize / sol.iterations:0.3f} second")

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
            f" \\caption{{Comparaison des métriques d'efficacité et de comportement entre les modèles de fatigue "
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

        for study, sol, rmse_index in zip(self.conditions.studies, self.solution, self.conditions.rmse_index):
            rmse = np.sum(self._rmse(DataType.STATES, "q", rmse_index, sol))
            rmse_str = f"{rmse:0.3e}" if rmse != 0 else "---"
            if rmse_str.find("e") >= 0:
                rmse_str = rmse_str.replace("e", "$\\times 10^{{")
                rmse_str += "}}$"
                rmse_str = rmse_str.replace("+0", "")
                rmse_str = rmse_str.replace("-0", "-")
                rmse_str = rmse_str.replace("$\\times 10^{{0}}$", "")
            table += (
                f"  {study.name} "
                f"& {sol.iterations} "
                f"& {sol.real_time_to_optimize:0.3f} "
                f"& {sol.real_time_to_optimize / sol.iterations:0.3f} "
                f"& {rmse_str} \\\\\n"
            )

        table += f"  \\hline\n" f" \\end{{tabular}}\n" f"\\end{{table}}\n\n"
        table += f"\\end{{document}}\n"

        save_path = f"{self._prepare_and_get_results_dir()}/results.tex"

        with open(save_path, "w", encoding='utf8') as file:
            file.write(table)
        print("\n\nTex file generated in the results folder")

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

    def prepare_plot_data(self, data_type: DataType, key: str, font_size: int = 20):
        if not self._has_run:
            raise RuntimeError("run() must be called before plotting the results")

        n_plots = getattr(self.solution[0], data_type.value)[key].shape[0]
        if sum(np.array([getattr(sol, data_type.value)[key].shape[0] for sol in self.solution]) != n_plots) != 0:
            raise RuntimeError("All the models must have the same number of dof to be plotted")
        t = np.linspace(self.solution[0].phase_time[0], self.solution[0].phase_time[1], self.solution[0].ns[0] + 1)

        plot_options = self.conditions.plot_options
        studies = self.conditions.studies

        for i in range(n_plots):
            fig = plt.figure()
            fig.set_size_inches(16, 9)
            plt.rcParams["text.usetex"] = True
            plt.rcParams["text.latex.preamble"] = (
                r"\usepackage{amssymb}"
                r"\newcommand{\condition}{C/}"
                r"\newcommand{\noFatigue}{\varnothing}"
                r"\newcommand{\qcc}{4\textsubscript{CC}}"
                r"\newcommand{\pe}{P\textsubscript{E}}"
                r"\newcommand{\taupm}{\tau^{\pm}}"
                r"\newcommand{\tauns}{\tau^{\times}}"
                r"\newcommand{\condTauNf}{{\condition}{\tau}{\noFatigue}}"
                r"\newcommand{\condTaupm}{{\condition}{\taupm}{}}"
                r"\newcommand{\condTaupmQcc}{{\condition}{\taupm}{\qcc}}"
                r"\newcommand{\condTaupmPe}{{\condition}{\taupm}{\pe}}"
                r"\newcommand{\condTauns}{{\condition}{\tauns}{}}"
                r"\newcommand{\condTaunsNf}{{\condition}{\tauns}{\noFatigue}}"
                r"\newcommand{\condTaunsQcc}{{\condition}{\tauns}{\qcc}}"
                r"\newcommand{\condTaunsPe}{{\condition}{\tauns}{\pe}}"
                r"\newcommand{\condAlpha}{{\condition}{\alpha}{}}"
                r"\newcommand{\condAlphaNf}{{\condition}{\alpha}{\noFatigue}}"
                r"\newcommand{\condAlphaQcc}{{\condition}{\alpha}{\qcc}}"
                r"\newcommand{\condAlphaPe}{{\condition}{\alpha}{\pe}}"
            )

            ax = plt.axes()
            ax.set_title(plot_options.title % f"{key}\\textsubscript{{{i}}}", fontsize=1.5 * font_size)
            ax.set_xlabel("Temps (s)", fontsize=font_size)
            ax.set_ylabel("Angle (rad)", fontsize=font_size)
            ax.tick_params(axis="both", labelsize=font_size)

            for sol, options in zip(self.solution, plot_options.options):
                data = getattr(sol, data_type.value)[key][i, :]
                plt.plot(t, data, **options)

            if plot_options.legend_indices is not None:
                legend = [study.name if idx else "_" for study, idx in zip(studies, plot_options.legend_indices)]
                ax.legend(legend, loc="lower right", fontsize=font_size, framealpha=0.9)

            if plot_options.maximize:
                plt.get_current_fig_manager().window.showMaximized()

            if plot_options.save_path is not None and plot_options.save_path[i] is not None:
                plt.show(block=False)
                plt.draw_all(True)
                plt.savefig(f"{self._prepare_and_get_results_dir()}/{plot_options.save_path[i]}", dpi=300)

        self._plots_are_prepared = True

    def _rmse(self, data_type, key, idx_ref: int, sol: Solution):
        data_ref = getattr(self.solution[idx_ref], data_type.value)[key]
        data = getattr(sol, data_type.value)[key]

        e = data_ref - data
        se = e ** 2
        mse = np.sum(se, axis=1) / data_ref.shape[1]
        rmse = np.sqrt(mse)
        return rmse

    def plot(self):
        if not self._plots_are_prepared:
            raise RuntimeError("At least one plot should be prepared before calling plot")

        plt.show()
