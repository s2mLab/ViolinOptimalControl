from time import time

from violin_ocp import Violin, ViolinString, ViolinNMPC, ViolinOcp, Bow, BowTrajectory, BowPosition
from bioptim import Solver


def main():
    model_name = "WuViolin"
    violin = Violin(model_name, ViolinString.E)
    bow = Bow(model_name)

    # --- OPTIONS --- #
    starting_position = BowPosition.TIP
    window = 40
    full_cycle = 40
    cycle_time = 3
    n_cycles = 10
    n_threads = 8
    solver = Solver.IPOPT
    with_fatigue = True
    with_muscles = False
    pre_solve = True

    # --- Solve the program --- #
    tic = time()
    nmpc_violin = ViolinNMPC(
        model_path=f"../models/{model_name}.bioMod",
        violin=violin,
        bow=bow,
        bow_starting=starting_position,
        use_muscles=with_muscles,
        fatigable=with_fatigue,
        solver=solver,
        window_len=window,
        window_duration=cycle_time,
        n_threads=n_threads
    )

    # Generate a full cycle target
    lim = bow.hair_limits if nmpc_violin.bow_starting == BowPosition.FROG else [bow.hair_limits[1], bow.hair_limits[0]]
    bow_trajectory = BowTrajectory(lim, full_cycle + 1)

    if pre_solve:
        ocp_pre = ViolinOcp(
            model_path=f"../models/{model_name}.bioMod",
            violin=violin,
            bow=bow,
            n_cycles=1,
            bow_starting=starting_position,
            init_file=None,
            use_muscles=with_muscles,
            fatigable=with_fatigue,
            time_per_cycle=cycle_time,
            n_shooting_per_cycle=full_cycle,
            solver=solver,
            n_threads=n_threads
        )
        ocp_pre.set_bow_target_objective(bow_trajectory.target)
        ocp_pre.set_cyclic_bound()
        sol_pre = ocp_pre.solve(limit_memory_max_iter=50, exact_max_iter=1000, force_no_graph=True)
        nmpc_violin.ocp.set_warm_start(sol_pre)

    def nmpc_update_function(ocp, t, sol):
        if t >= n_cycles:
            print("Finished optimizing!")
            return False

        print(f"Optimizing cycle {t + 1}..")
        _t = 0  # Cyclic so t should always be the start
        target_time_index = [i % full_cycle for i in range(_t, _t + window + 1)]
        nmpc_violin.set_bow_target_objective(bow_trajectory.target[:, target_time_index])
        return True

    sol = nmpc_violin.solve(nmpc_update_function, show_online_optim=True)

    # Data output
    nmpc_violin.save(sol, ext=f"{n_cycles}_cycles{'_with_fatigue' if with_fatigue else ''}", stand_alone=True)
    print(f"Running time: {time() - tic} seconds")
    sol.print()
    sol.graphs()
    sol.animate()


if __name__ == "__main__":
    main()
