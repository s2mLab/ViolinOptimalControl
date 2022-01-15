import pickle
from enum import Enum

import bioviz
import numpy as np
from matplotlib import pyplot as plt
from bioptim import OptimalControlProgram


class DataType(Enum):
    STATE = 0
    CONTROL = 1
    PARAMETER = 2


def load_data(path: str, n_cycles: int, use_bo: bool = True):

    if use_bo:
        with open(path, "rb") as file:
            data = pickle.load(file)
    else:
        data_tp = OptimalControlProgram.load(path)[1]
        data = [data_tp.states, data_tp.controls]

    out = {}
    for data_type in [DataType.STATE, DataType.CONTROL]:
        data_per_type = {}
        for key in data[data_type.value].keys():
            n_frame_per_cycle = int(data[data_type.value][key].shape[1] / n_cycles)
            data_tp = data[data_type.value][key].reshape((-1, n_frame_per_cycle, n_cycles), order="F")

            mean_data = data_tp.mean(axis=2)
            rms_data = np.sqrt(((data_tp - mean_data[:, :, np.newaxis]) ** 2).mean(axis=2))
            data_per_type[key] = {"mean": mean_data, "rms": rms_data, "all_cycles": data_tp}

        out[data_type] = data_per_type
    return out


def prepare_subplots(n_elements):
    n_rows = int(np.ceil(np.sqrt(n_elements)))
    n_cols = int(np.ceil(n_elements / n_rows))
    fig, axs = plt.subplots(n_rows, n_cols)
    axs = np.array(axs)
    return axs.reshape((-1,))


def plot_shaded(axs, t, data, mod, **opts):
    for i in range(data["mean"].shape[0]):
        d = data["mean"][i, :] * mod
        r = data["rms"][i, :] * mod

        axs[i].plot(t, d, **opts)
        axs[i].fill_between(t, d - r, d + r, alpha=0.4, **opts)


def plot_all(axs, t, data, mod, skip_frame=None, **opts):
    n_dof = data["all_cycles"].shape[0]
    n_cycle = data["all_cycles"].shape[2]

    if not skip_frame:
        skip_frame = []
    cycle = [i for i in range(n_cycle) if i not in skip_frame]

    for i in range(n_dof):
        d = (data["all_cycles"][i, :, cycle].squeeze() * mod).T
        axs[i].plot(t, d, alpha=0.4, **opts)
        # axs[i].set_ylim([0, 1])


def main():
    # OPTIONS
    folder = "./results/900_cycles"
    files = ["cycles_non_fatigue", "cycles_with_fatigue"]
    use_bo = False
    mod = 1  # -180 / np.pi
    n_cycles = 900
    cycle_time = 1
    # data_type = DataType.STATE
    data_type = DataType.CONTROL
    elt = None
    # data_key = "tau_plus_mf"
    # data_key = "q"
    data_key = "tau"
    colors = ["tab:green", "tab:red"]
    show_shaded = False
    show_all = True
    animate = False
    model_path = "../models/WuViolin.bioMod"
    skip_frame = [0]

    plot_title = "Humerus abduction"  # "Humerus fatigue"  # "Humerus abduction"
    y_label = "Angle (°)"  # "Accumulated fatigue"  #

    # Get data
    all_data = [
        load_data(f"{folder}/{n_cycles}_{file}{'_out' if use_bo else ''}.bo", n_cycles, use_bo) for file in files
    ]

    # Show
    if show_shaded or show_all:
        axs_shaded = None
        axs_all = None

        for i, data in enumerate(all_data):
            data_to_plot = data[data_type][data_key]

            # Slice the data
            is_elt_int = False
            if elt:
                if isinstance(elt, int):
                    is_elt_int = True
                    elt = range(elt, elt + 1)
                n_elt = len(elt)
            else:
                n_elt = data_to_plot["all_cycles"].shape[0]
                elt = range(n_elt)
            for key in data_to_plot.keys():
                data_to_plot[key] = data_to_plot[key][elt, ...]

            # Prepare time vector
            n_frame_per_cycle = data_to_plot["all_cycles"].shape[1]
            t = np.linspace(0, cycle_time, n_frame_per_cycle)

            if show_shaded:
                if i == 0:
                    axs_shaded = prepare_subplots(n_elt)
                    if is_elt_int:
                        font_size = 45
                        axs_shaded[0].set_title(plot_title, fontsize=font_size)
                        axs_shaded[0].set_xlabel("Time (s)", fontsize=font_size)
                        axs_shaded[0].set_ylabel(y_label, fontsize=font_size)
                        axs_shaded[0].tick_params(labelsize=font_size)
                plot_shaded(axs_shaded, t, data_to_plot, mod=mod, color=colors[i])

            if show_all:
                if i == 0:
                    axs_all = prepare_subplots(n_elt)
                plot_all(axs_all, t, data_to_plot, mod=mod, color=colors[i], skip_frame=skip_frame)
        plt.show()

    if animate:
        viz = []
        for data in all_data:
            viz.append(bioviz.Viz(model_path, show_muscles=False, show_local_ref_frame=False))
            viz[-1].load_movement(data[DataType.STATE]["q"]["mean"])

        is_still_running = True
        while is_still_running:
            for v in viz:
                if not v.vtk_window.is_active:
                    is_still_running = False
                    break
                v.update()


if __name__ == "__main__":
    main()
