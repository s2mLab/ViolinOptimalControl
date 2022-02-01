import os
from enum import Enum, auto
from copy import copy
from typing import Union, Callable

from bioviz import Viz
import numpy as np
from matplotlib import pyplot as plt

from bioptim import OptimalControlProgram, Shooting, Solution, InitialGuess, InterpolationType


class ExtraFiguresFunctionImplementation:
    @staticmethod
    def initial_guess_cyclic_nmpc(studies, save_folder: str, data_path: str):
        class From(Enum):
            START = 0
            END = -1
            NONE = auto()

        def plot_integrated(
            sol_to_plot: Solution,
            index: int,
            color_index: int,
            circle_on: From = From.NONE,
            title: str = "",
            save_path: str = None,
            legend: tuple[str, ...] = None,
            **kwargs
        ):
            t = np.linspace(*sol.phase_time, ns * n_steps + 1)
            color = plt.rcParams['axes.prop_cycle'].by_key()['color'][color_index]

            integrated = sol_to_plot.integrate(
                shooting_type=Shooting.MULTIPLE, keep_intermediate_points=True, continuous=False
            )
            for i in range(ns):
                frames_t = slice(i * n_steps, (i + 1) * n_steps + 1)
                frames_q = slice(i * (n_steps + 1), (i + 1) * (n_steps + 1))
                plt.plot(t[frames_t], integrated.states["q"][index, frames_q] * 180 / np.pi, color=color, **kwargs)
            plt.plot(t[::n_steps], sol_to_plot.states["q"][index, :] * 180 / np.pi, ".", color=color, markersize=15, **kwargs)

            frame = circle_on.value
            plt.scatter(t[frame], sol_to_plot.states["q"][index, frame] * 180 / np.pi, facecolors='none', s=5000, edgecolors="red")

            fig = plt.gcf()
            fig.set_size_inches(16, 9)
            ax = plt.gca()
            ax.set_title(title, fontsize=1.5 * font_size)
            ax.set_xlabel(r"Temps (\SI{}{\second})", fontsize=font_size)
            ax.set_ylabel(r"Angle (\SI{}{\degree})", fontsize=font_size)
            ax.tick_params(axis="both", labelsize=font_size)

            if legend is not None:
                ax.legend(legend, loc="upper left", fontsize=font_size, framealpha=0.9)

            if save_path is not None:
                folder_path = os.path.dirname(save_path)
                try:
                    os.mkdir(folder_path)
                except FileExistsError:
                    pass
                plt.savefig(save_path, dpi=300)

        # Alias
        font_size = 20
        abduction = 4
        plt.rcParams["text.usetex"] = True
        plt.rcParams["text.latex.preamble"] = r"\usepackage{siunitx}"

        # Get the data
        ocp, sol = OptimalControlProgram.load(f"{save_folder}/{data_path}")
        ns = ocp.nlp[0].ns
        n_steps = sol.ocp.nlp[0].ode_solver.steps

        # Integrate and show a final solution
        plt.figure()
        plot_integrated(sol, abduction, 0, circle_on=From.END, linewidth=3, save_path=f"{save_folder}/method_nmpc_cyclic_solution.png", legend=("Solution",))

        # Advance as a cyclic NMPC and integrate the initial guess
        x_init = InitialGuess(copy(sol.states["all"]), interpolation=InterpolationType.EACH_FRAME)
        x_init.init[:, 0] = x_init.init[:, -1]
        u_init = InitialGuess(copy(sol.controls["all"]), interpolation=InterpolationType.EACH_FRAME)
        sol_next = Solution(ocp, [x_init, u_init])

        plt.figure()
        plot_integrated(
            sol_next, abduction, 0, circle_on=From.START, save_path=f"{save_folder}/method_nmpc_cyclic_initial_guess.png", legend=("Solution initiale suivante",)
        )

    @staticmethod
    def bow_figure(studies, save_folder: str):
        bow_index = 13
        b = Viz(
            "models/WuViolin.bioMod",
            show_floor=False,
            show_global_ref_frame=False,
            show_gravity_vector=False,
            background_color=(1, 1, 1),
            show_global_center_of_mass=False,
            show_segments_center_of_mass=False,
            show_muscles=False,
        )

        b.toggle_segments(tuple(range(b.model.nbSegment())))
        b.toggle_segments(bow_index)
        b.toggle_segments(bow_index + 1)
        q = np.zeros(b.model.nbQ())
        q[-1] = -0.25
        b.set_q(q)

        b.resize(1920, 1080)
        b.set_camera_focus_point(0.12, -0.6, 0.3)
        b.set_camera_position(-0.45, 0.8, 0.3)
        b.set_camera_roll(80)
        b.set_camera_zoom(3.2)

        b.snapshot(f"{save_folder}/method_bow.png")
        b.quit()

    @staticmethod
    def violin(studies, save_folder: str):
        b = Viz(
            "models/violon_one_string.bioMod",
            show_floor=False,
            show_global_ref_frame=False,
            show_gravity_vector=False,
            background_color=(1, 1, 1),
            markers_size=0.003,
            show_global_center_of_mass=False,
            show_segments_center_of_mass=False,
        )

        b.resize(1920, 1080)
        b.set_camera_position(-0.35, -1.35, -0.4)
        b.set_camera_roll(-65)
        b.set_camera_zoom(5.5)

        b.snapshot(f"{save_folder}/method_violin.png")
        b.quit()


class ExtraFiguresFcn(Enum):
    VIOLIN = ExtraFiguresFunctionImplementation.violin
    BOW = ExtraFiguresFunctionImplementation.bow_figure
    INITIAL_GUESS_NMPC = ExtraFiguresFunctionImplementation.initial_guess_cyclic_nmpc


class ExtraFigureOption:
    def __init__(self, extra_figure: Union[ExtraFiguresFcn, Callable], **params):
        self.extra_figure = extra_figure
        self.params = params


class ExtraFigures:
    def __init__(self, extra_figures: tuple[ExtraFigureOption, ...]):
        self.extra_figures = extra_figures

    def generate_extra_figures(self, studies, save_folder):
        for figure in self.extra_figures:
            figure.extra_figure(studies, save_folder, **figure.params)
