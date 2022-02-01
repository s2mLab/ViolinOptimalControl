from enum import Enum, auto
from copy import copy

from bioviz import Viz
import numpy as np
from matplotlib import pyplot as plt

from bioptim import OptimalControlProgram, Shooting, Solution, InitialGuess, InterpolationType


def initial_guess_cyclic_nmpc():
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
            plt.savefig(save_path, dpi=300)

    # Alias
    font_size = 20
    abduction = 4
    plt.rcParams["text.usetex"] = True
    plt.rcParams["text.latex.preamble"] = r"\usepackage{siunitx}"

    # Get the data
    ocp, sol = OptimalControlProgram.load("cyclic_nmpc_results/iteration_0000.bo")
    ns = ocp.nlp[0].ns
    n_steps = sol.ocp.nlp[0].ode_solver.steps

    # Integrate and show a final solution
    plt.figure()
    plot_integrated(sol, abduction, 0, circle_on=From.END, linewidth=3, save_path="method_nmpc_cyclic_solution.png", legend=("Solution", ))

    # Advance as a cyclic NMPC and integrate the initial guess
    x_init = InitialGuess(copy(sol.states["all"]), interpolation=InterpolationType.EACH_FRAME)
    x_init.init[:, 0] = x_init.init[:, -1]
    u_init = InitialGuess(copy(sol.controls["all"]), interpolation=InterpolationType.EACH_FRAME)
    sol_next = Solution(ocp, [x_init, u_init])

    plt.figure()
    plot_integrated(
        sol_next, abduction, 0, circle_on=From.START, save_path="method_nmpc_cyclic_initial_guess.png", legend=("Solution initiale suivante", )
    )


def bow_figure():
    b = Viz(
        "models/bow.bioMod",
        show_floor=False,
        show_global_ref_frame=False,
        show_gravity_vector=False,
        background_color=(1, 1, 1),
        show_global_center_of_mass=False,
        show_segments_center_of_mass=False,
    )

    b.resize(1920, 1080)
    b.set_camera_position(1, 0.9, 0.2)
    b.set_camera_roll(-100)
    b.set_camera_zoom(4.5)

    b.snapshot("bow")
    b.quit()


def violin():
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

    b.snapshot("violin")
    b.quit()


if __name__ == "__main__":
    # bow_figure()
    # violin()

    initial_guess_cyclic_nmpc()