from time import time

from violin_ocp import Violin, ViolinString, ViolinNMPC, Bow, BowTrajectory, BowPosition
from bioptim import Solver


if __name__ == "__main__":
    model_name = "WuViolin"
    violin = Violin(model_name, ViolinString.E)
    bow = Bow(model_name)

    # --- Solve the program --- #
    window = 30
    full_cycle = 30
    cycle_time = 1
    n_cycles = 3
    solver = Solver.ACADOS
    ocp_violin = ViolinNMPC(
        model_path=f"../models/{model_name}.bioMod",
        violin=violin,
        bow=bow,
        bow_starting=BowPosition.TIP,
        use_muscles=False,
        solver=solver,
        window_len=window,
        window_duration=cycle_time,
    )

    # Generate a full cycle target
    lim = bow.hair_limits if ocp_violin.bow_starting == BowPosition.FROG else [bow.hair_limits[1], bow.hair_limits[0]]
    bow_trajectory = BowTrajectory(lim, full_cycle + 1)

    def nmpc_update_function(ocp, t, sol):
        print(t)
        target_time_index = [i % full_cycle for i in range(t, t + window + 1)]
        ocp_violin.set_bow_target_objective(bow_trajectory.target[:, target_time_index])
        return t < n_cycles * full_cycle - window

    tic = time()
    full_sol = ocp_violin.solve(nmpc_update_function)
    print(f"Running time: {time() - tic}")

    # sol.print()
    full_sol.graphs()
    full_sol.animate(show_meshes=False)
