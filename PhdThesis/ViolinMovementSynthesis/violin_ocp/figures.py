from enum import Enum
from typing import Union, Callable
from copy import copy

import numpy as np
from matplotlib import pyplot as plt
from bioptim import Solution

from .enums import DataType


class FiguresFunctionImplementation:
    @staticmethod
    def data_in_one_go(
            studies,
            idx_study: int,
            solution: Solution,
            all_iterations: tuple[Solution, ...],
            data_type: DataType,
            key: str,
            index: int,
            to_degree: bool = True,
            is_fatigue: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, list[dict, ...], dict]:
        ns = solution.states["q"].shape[1]

        t = np.linspace(*solution.phase_time, ns)
        data = copy(getattr(solution, data_type.value)[key][index, :])
        if to_degree:
            data *= 180/np.pi

        if is_fatigue:
            y_label = "Niveau (\%)"
        else:
            y_label = "Angle (degree)" if to_degree else "Angle (rad)"

        plot_options = [
            {"color": plt.rcParams['axes.prop_cycle'].by_key()['color'][idx_study]}
        ]
        ax_options = {"xlabel": "Temps (s)", "ylabel": y_label}
        return t, data[np.newaxis, :], plot_options, ax_options

    @staticmethod
    def data_stacked_per_cycle(
            studies,
            idx_study: int,
            solution: Solution,
            all_iterations: tuple[Solution, ...],
            data_type: DataType,
            key: str,
            index: int,
            to_degree: bool = True,
            is_fatigue: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, list[dict, ...], dict]:
        n_cycle = studies.studies[idx_study].n_cycles_total
        ns = int(solution.states["q"].shape[1] / n_cycle)
        if ns != solution.states["q"].shape[1] / n_cycle:
            raise RuntimeError("ns should be divisible by n_cycle")

        t0, tf = solution.phase_time[0], solution.phase_time[1] / n_cycle
        t = np.linspace(t0, tf, ns)

        data = copy(getattr(solution, data_type.value)[key][index, :])
        data = data.reshape((-1, ns), order="C")
        if is_fatigue:
            data *= 100
        elif to_degree:
            data *= 180/np.pi

        if is_fatigue:
            y_label = "Niveau (\%)"
        else:
            y_label = "Angle (degree)" if to_degree else "Angle (rad)"

        plot_options = []
        alpha_range = (0.2, 1)
        for i in range(data.shape[0]):
            plot_options.append(
                {
                    "color": plt.rcParams['axes.prop_cycle'].by_key()['color'][idx_study],
                    "alpha": alpha_range[0] + (i / data.shape[0]) * (alpha_range[1] - alpha_range[0])
                }
            )
        ax_options = {"xlabel": "Temps (s)", "ylabel": y_label}
        return t, data, plot_options, ax_options

    @staticmethod
    def phase_diagram(
            studies,
            idx_study: int,
            solution: Solution,
            all_iterations: tuple[Solution, ...],
            data_meta: tuple[tuple[DataType, str, int], tuple[DataType, str, int]],
            to_degree: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, list[dict, ...], dict]:
        (data_type_x, key_x, index_x), (data_type_y, key_y, index_y) = data_meta

        data_x = copy(getattr(solution, data_type_x.value)[key_x][index_x, :])
        data_y = copy(getattr(solution, data_type_y.value)[key_y][index_y, :])
        if to_degree:
            data_x *= 180/np.pi
            data_y *= 180 / np.pi

        x_label = "Angle (degree)" if to_degree else "Angle (rad)"
        y_label = "Angle (degree)" if to_degree else "Angle (rad)"

        plot_options = [
            {"color": plt.rcParams['axes.prop_cycle'].by_key()['color'][idx_study]}
        ]
        ax_options = {"xlabel": x_label, "ylabel": y_label, "aspect": "equal"}
        return data_x, data_y[np.newaxis, :], plot_options, ax_options


class FiguresFcn(Enum):
    DATA_IN_ONE_GO = FiguresFunctionImplementation.data_in_one_go
    DATA_STACKED_PER_CYCLE = FiguresFunctionImplementation.data_stacked_per_cycle
    PHASE_DIAGRAM = FiguresFunctionImplementation.phase_diagram


class FigureOptions:
    def __init__(
            self,
            title: str,
            fcn: Union[Callable, FiguresFcn],
            params: dict = None,
            save_name: str = ""
    ):
        self.title = title
        self.fcn: Union[Callable, FiguresFcn] = fcn
        self.extra_params = {} if params is None else params
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
                    plt.plot(t, data_tp, **options)

                legend.extend(["_"] * (data.shape[0] - 1))
                legend.append(studies.studies[i].name)

                if not axes_options_set:
                    ax.set(**ax_options)
                    axes_options_set = True

            ax.set_xlabel(ax.get_xlabel(), fontsize=self.font_size)
            ax.set_ylabel(ax.get_ylabel(), fontsize=self.font_size)
            ax.tick_params(axis="both", labelsize=self.font_size)
            ax.legend(legend, loc="upper right", fontsize=self.font_size, framealpha=0.9)

            if figure.save_name and save_folder is not None:
                plt.show(block=False)
                plt.draw_all(True)
                plt.savefig(f"{save_folder}/{figure.save_name}", dpi=300)
