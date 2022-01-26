from enum import Enum
from typing import Union, Callable
from copy import copy

import numpy as np
from matplotlib import pyplot as plt
from bioptim import Solution, InitialGuess, InterpolationType

from .enums import DataType


class FiguresFunctionImplementation:
    @staticmethod
    def _get_data(
            studies,
            idx_study: int,
            solution: Solution,
            first_cycle: Union[int, None],
            last_cycle: Union[int, None],
            data_type: DataType,
            key: str,
            index: Union[int, tuple],
    ):
        n_cycles = studies.studies[idx_study].n_cycles_total
        if first_cycle is None:
            first_cycle = 0
        elif first_cycle < 0:
            first_cycle = n_cycles - first_cycle

        if last_cycle is None:
            last_cycle = n_cycles
        elif last_cycle < 0:
            last_cycle = n_cycles - last_cycle

        ns = solution.states["q"].shape[1]
        ns_per_cycle = int(ns / n_cycles)
        if ns_per_cycle != ns / n_cycles:
            raise RuntimeError("ns should be divisible by n_cycle")

        n_cycles = last_cycle - first_cycle
        first_frame = first_cycle * ns_per_cycle
        last_frame = last_cycle * ns_per_cycle
        t = np.linspace(first_cycle, last_cycle, n_cycles * ns_per_cycle)

        if isinstance(index, int):
            index = (index, )
        data = copy(getattr(solution, data_type.value)[key][index, first_frame:last_frame])

        return t, data, n_cycles

    @staticmethod
    def data_in_one_go(
            studies,
            idx_study: int,
            solution: Solution,
            all_iterations: tuple[Solution, ...],
            data_type: DataType,
            key: str,
            index: Union[int, tuple],
            first_cycle: int = None,
            last_cycle: int = None,
            to_degree: bool = False,
            is_fatigue: bool = False,
            ylim: tuple[float, float] = None
    ) -> tuple[np.ndarray, np.ndarray, list[dict, ...], dict]:
        t, data, _ = FiguresFunctionImplementation._get_data(studies, idx_study, solution, first_cycle, last_cycle, data_type, key, index)

        if is_fatigue:
            data *= 100
        if to_degree:
            data *= 180/np.pi

        if data_type == DataType.STATES:
            if is_fatigue:
                y_label = r"Niveau (\SI{}{\percent})"
            else:
                y_label = r"Angle (\SI{}{\degree})" if to_degree else r"Angle (\SI{}{\radian})"
        elif data_type == DataType.CONTROLS:
            y_label = r"Effort (\SI{}{\newton\meter})"
        else:
            raise NotImplementedError("data_type not implemented yet")

        plot_options = [
            {"color": plt.rcParams['axes.prop_cycle'].by_key()['color'][idx_study]}
        ]
        ax_options = {"xlabel": r"Temps (\SI{}{\second})", "ylabel": y_label, "ylim": ylim}
        return t, data, plot_options, ax_options

    @staticmethod
    def data_stacked_per_cycle(
            studies,
            idx_study: int,
            solution: Solution,
            all_iterations: tuple[Solution, ...],
            data_type: DataType,
            key: str,
            index: Union[int, tuple],
            first_cycle: int = None,
            last_cycle: int = None,
            to_degree: bool = False,
            is_fatigue: bool = False,
            ylim: tuple[float, float] = None
    ) -> tuple[np.ndarray, np.ndarray, list[dict, ...], dict]:
        t, data, n_cycles = FiguresFunctionImplementation._get_data(studies, idx_study, solution, first_cycle, last_cycle, data_type, key, index)
        t = t[:int(data.shape[1] / n_cycles)] - t[0]
        data = data.reshape((-1, int(data.shape[1] / n_cycles)), order="C")

        if is_fatigue:
            data *= 100
        if to_degree:
            data *= 180/np.pi

        if data_type == DataType.STATES:
            if is_fatigue:
                y_label = r"Niveau (\SI{}{\percent})"
            else:
                y_label = r"Angle (\SI{}{\degree})" if to_degree else "Angle (rad)"
        elif data_type == DataType.CONTROLS:
            y_label = r"Effort (\SI{}{\newton\meter})"
        else:
            raise NotImplementedError("data_type not implemented yet")

        plot_options = []
        alpha_range = (0.2, 1)
        for i in range(data.shape[0]):
            plot_options.append(
                {
                    "color": plt.rcParams['axes.prop_cycle'].by_key()['color'][idx_study],
                    "alpha": alpha_range[0] + (i / data.shape[0]) * (alpha_range[1] - alpha_range[0])
                }
            )
        ax_options = {"xlabel": r"Temps (\SI{}{\second})", "ylabel": y_label, "ylim": ylim}
        return t, data, plot_options, ax_options

    @staticmethod
    def phase_diagram(
            studies,
            idx_study: int,
            solution: Solution,
            all_iterations: tuple[Solution, ...],
            data_meta: tuple[tuple[DataType, str, int], tuple[DataType, str, int]],
            first_cycle: int = None,
            last_cycle: int = None,
            to_degree: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, list[dict, ...], dict]:
        (data_type_x, key_x, index_x), (data_type_y, key_y, index_y) = data_meta

        _, data_x, _ = FiguresFunctionImplementation._get_data(studies, idx_study, solution, first_cycle, last_cycle, data_type_x, key_x, index_x)
        _, data_y, _ = FiguresFunctionImplementation._get_data(studies, idx_study, solution, first_cycle, last_cycle, data_type_y, key_y, index_y)
        if to_degree:
            data_x *= 180/np.pi
            data_y *= 180 / np.pi

        if data_type_x == DataType.STATES:
            x_label = r"Angle (\SI{}{\degree})" if to_degree else r"Angle (\SI{}{\radian})"
        elif data_type_x == DataType.CONTROLS:
            x_label = r"Effort (\SI{}{\newton\meter})"
        else:
            raise NotImplementedError("data_type not implemented yet")
        if data_type_x == DataType.STATES:
            y_label = r"Angle (\SI{}{\degree})" if to_degree else r"Angle (\SI{}{\radian})"
        elif data_type_x == DataType.CONTROLS:
            y_label = r"Effort (\SI{}{\newton\meter})"
        else:
            raise NotImplementedError("data_type not implemented yet")

        plot_options = [
            {"color": plt.rcParams['axes.prop_cycle'].by_key()['color'][idx_study]}
        ]
        ax_options = {"xlabel": x_label, "ylabel": y_label, "aspect": "equal"}
        return data_x[0, :], data_y, plot_options, ax_options

    @staticmethod
    def integration_from_another_dynamics(
            studies,
            idx_study: int,
            solution: Solution,
            all_iterations: tuple[Solution, ...],
            dynamics_source_idx: int,
            key: str,
            index: Union[int, tuple],
            first_cycle: int = None,
            last_cycle: int = None,
            to_degree: bool = False,
            is_fatigue: bool = False,
            ylim: tuple[float, float] = None
    ):
        if not hasattr(studies, "integrated_solutions"):
            studies.integrated_solutions: list[Union[None, Solution], ...] = [None] * len(studies.studies)

        if idx_study == dynamics_source_idx:
            studies.integrated_solutions[idx_study] = studies.solutions[idx_study][0]

        elif not studies.integrated_solutions[idx_study]:
            print(f"Integrating {studies.studies[idx_study].name} using {studies.studies[dynamics_source_idx].name}"
                  f", this may take some time...")
            n_cycles = studies.studies[0].n_cycles_total
            ns = solution.states["q"].shape[1]
            ns_per_cycle = int(ns / n_cycles)
            d = dynamics_source_idx

            # Integrate using the PE model
            ocp = studies.solutions[d][0].ocp
            ocp.nlp[0].dynamics = [studies.studies[dynamics_source_idx].nmpc.ocp.nlp[0].dynamics[0]] * n_cycles * ns_per_cycle
            ocp.nlp[0].ns = n_cycles * ns_per_cycle - 1
            ocp.v.n_phase_x[0] = studies.solutions[d][0].states["all"].shape[0] * studies.solutions[d][0].states["all"].shape[1]
            ocp.v.n_phase_u[0] = studies.solutions[d][0].controls["all"].shape[0] * (studies.solutions[d][0].controls["all"].shape[1] - 1)
            ocp.v.n_all_x = sum(ocp.v.n_phase_x)
            ocp.v.n_all_u = sum(ocp.v.n_phase_u)

            x_init = InitialGuess(copy(studies.solutions[d][0].states["all"][:, 0]))
            u_init = InitialGuess(
                copy(studies.solutions[idx_study][0].controls["all"]),
                interpolation=InterpolationType.EACH_FRAME
            )
            sol = Solution(ocp, [x_init, u_init])
            studies.integrated_solutions[idx_study] = sol.integrate()

        integrated_solution = studies.integrated_solutions[idx_study]
        t, data, _ = FiguresFunctionImplementation._get_data(studies, idx_study, integrated_solution, first_cycle, last_cycle, DataType.STATES, key, index)

        if is_fatigue:
            data *= 100
        if to_degree:
            data *= 180/np.pi

        plot_options = [
            {"color": plt.rcParams['axes.prop_cycle'].by_key()['color'][idx_study]}
        ]
        ax_options = {"xlabel": r"Temps (\SI{}{\second})", "ylabel": r"Niveau (\SI{}{\percent})", "ylim": ylim}
        return t, data, plot_options, ax_options


class FiguresFcn(Enum):
    DATA_IN_ONE_GO = FiguresFunctionImplementation.data_in_one_go
    DATA_STACKED_PER_CYCLE = FiguresFunctionImplementation.data_stacked_per_cycle
    PHASE_DIAGRAM = FiguresFunctionImplementation.phase_diagram
    INTEGRATION_FROM_ANOTHER_DYNAMICS = FiguresFunctionImplementation.integration_from_another_dynamics


class FigureOptions:
    def __init__(
            self,
            title: str,
            fcn: Union[Callable, FiguresFcn],
            use_subplots: bool = False,
            params: dict = None,
            save_name: str = ""
    ):
        self.title = title
        self.fcn: Union[Callable, FiguresFcn] = fcn
        self.extra_params = {} if params is None else params
        self.use_subplots = use_subplots
        self.save_name = save_name


class Figures:
    def __init__(
            self,
            figures: tuple[FigureOptions, ...],
            font_size: int = 20,
    ):
        self.figure_options: tuple[FigureOptions, ...] = figures
        self.font_size = font_size

        self.preamble = (
            r"\usepackage{amsmath}"
            r"\usepackage{amssymb}"
            r"\usepackage[table]{xcolor}"
            r"\usepackage{threeparttable}"
            r"\usepackage{makecell}"
            r"\usepackage{siunitx}"
            r"\definecolor{lightgray}{gray}{0.91}"
            r"\newcommand{\rmse}{RMSE}"
            r"\newcommand{\ocp}{OCP}"
            r"\newcommand{\cyclicNMPC}{NMPC cyclique}"
            r"\newcommand{\multiCyclicNMPC}{NMPC multicyclique}"
            r"\newcommand{\controls}{\mathbf{u}}"
            r"\newcommand{\states}{\mathbf{x}}"
            r"\newcommand{\statesDot}{\mathbf{\dot{x}}}"
            r"\newcommand{\q}{\mathbf{q}}"
            r"\newcommand{\qdot}{\mathbf{\dot{q}}}"
            r"\newcommand{\qddot}{\mathbf{\ddot{q}}}"
            r"\newcommand{\f}{\mathbf{f}}"
            r"\newcommand{\taupm}{\tau^{\pm}}"
            r"\newcommand{\tauns}{\\tau^{\times}}"
            r"\newcommand{\condition}{C/}"
            r"\newcommand{\noFatigue}{\varnothing}"
            r"\newcommand{\qcc}{4\textsubscript{CC}}"
            r"\newcommand{\pe}{P\textsubscript{E}}"
            r"\newcommand{\condTau}{{\condition}{\\tau}{}}"
            r"\newcommand{\condTauNf}{{\condition}{\tau}{\noFatigue}}"
            r"\newcommand{\condTauQcc}{{\condition}{\tau}{\qcc}}"
            r"\newcommand{\condTauPe}{{\condition}{\tau}{\pe}}"
            r"\newcommand{\condTaupm}{{\condition}{\taupm}{}}"
            r"\newcommand{\condTaupmQcc}{{\condition}{\taupm}{\qcc}}"
            r"\newcommand{\condTaupmPe}{{\condition}{\taupm}{\pe}}"
            r"\newcommand{\condTauns}{{\condition}{\tauns}{}}"
            r"\newcommand{\condTaunsQcc}{{\condition}{\tauns}{\qcc}}"
            r"\newcommand{\condTaunsPe}{{\condition}{\tauns}{\pe}}"
            r"\newcommand{\condAlpha}{{\condition}{\alpha}{}}"
            r"\newcommand{\condAlphaNf}{{\condition}{\alpha}{\noFatigue}}"
            r"\newcommand{\condAlphaQcc}{{\condition}{\alpha}{\qcc}}"
            r"\newcommand{\condAlphaPe}{{\condition}{\alpha}{\pe}}"
        )

    def generate_figure(self, studies, all_solutions: list[tuple[Solution, list[Solution]], ...], save_folder: str = None):
        all_figures = []

        for figure in self.figure_options:
            all_figures.append(plt.figure())
            all_figures[-1].set_size_inches(16, 9)
            plt.rcParams["text.usetex"] = True
            plt.rcParams["text.latex.preamble"] = self.preamble

            ax = plt.axes()
            ax.set_title(figure.title, fontsize=1.5 * self.font_size)

            # Perform the computation
            axes_options_set = False
            legend = []
            for i, solutions in enumerate(all_solutions):
                try:
                    t, data, plot_options, ax_options = figure.fcn(studies, i, *solutions, **figure.extra_params)
                except KeyError:
                    continue
                for data_tp, options in zip(data, plot_options):
                    if "data_type" not in figure.extra_params or figure.extra_params["data_type"] == DataType.STATES:
                        plt.plot(t, data_tp, **options)
                    elif figure.extra_params["data_type"] == DataType.CONTROLS:
                        plt.step(t, data_tp, where="post", **options)
                    else:
                        raise ValueError(f"Wrong data_type ({figure.extra_params['data_type']})")

                legend.extend(["_"] * (data.shape[0] - 1))
                legend.append(studies.studies[i].name)

                ax.relim()
                ax.autoscale()
                if not axes_options_set:
                    ax.set(**ax_options)
                    axes_options_set = True

            ax.set_xlabel(ax.get_xlabel(), fontsize=self.font_size)
            ax.set_ylabel(ax.get_ylabel(), fontsize=self.font_size)
            ax.tick_params(axis="both", labelsize=self.font_size)
            ax.legend(legend, loc="upper left", fontsize=self.font_size, framealpha=0.9)

            if figure.save_name and save_folder is not None:
                plt.show(block=False)
                plt.draw_all(True)
                plt.savefig(f"{save_folder}/{figure.save_name}", dpi=300)
