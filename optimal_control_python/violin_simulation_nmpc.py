from time import time

import numpy as np
from violin_ocp import Violin, ViolinString, ViolinNMPC, ViolinOcp, Bow, BowTrajectory, BowPosition
from bioptim import Solver, OdeSolver


def main():
    model_name = "WuViolin"
    violin = Violin(model_name, ViolinString.G)
    bow = Bow(model_name)

    # --- OPTIONS --- #
    starting_position = BowPosition.TIP
    full_cycle = 30
    cycle_time = 1
    cycle_from = 1
    n_cycles_simultaneous = 3
    n_cycles = 900
    n_threads = 32
    solver = Solver.IPOPT()
    ode_solver = OdeSolver.RK4(n_integration_steps=3)
    with_fatigue = True
    minimize_fatigue = True
    with_muscles = False
    pre_solve = True

    # Final save name
    save_name = f"{n_cycles}_cycles{'_with_fatigue' if with_fatigue else ''}"
    
    # Generate a full cycle target
    lim = bow.hair_limits if starting_position == BowPosition.FROG else [bow.hair_limits[1], bow.hair_limits[0]]
    bow_trajectory = BowTrajectory(lim, full_cycle + 1)
    bow_trajectory.target = np.tile(bow_trajectory.target[:, :-1], n_cycles_simultaneous)
    bow_trajectory.target = np.concatenate((bow_trajectory.target, bow_trajectory.target[:, 0][:, np.newaxis]), axis=1)

    # --- Solve the program --- #
    window = full_cycle
    tic = time()
    nmpc_violin = ViolinNMPC(
        model_path=f"../models/{model_name}.bioMod",
        violin=violin,
        bow=bow,
        n_cycles_simultaneous=n_cycles_simultaneous,
        n_cycles_to_advance=1,
        bow_starting=starting_position,
        use_muscles=with_muscles,
        fatigable=with_fatigue,
        minimize_fatigue=minimize_fatigue,
        solver=solver,
        ode_solver=ode_solver,
        window_len=window,
        window_duration=cycle_time,
        n_threads=n_threads,
    )
    nmpc_violin.set_bow_target_objective(bow_trajectory.target)
    nmpc_violin.set_cyclic_bound(0.01)

    sol_pre = None
    if pre_solve:
        ocp_pre = ViolinOcp(
            model_path=f"../models/{model_name}.bioMod",
            violin=violin,
            bow=bow,
            n_cycles=n_cycles_simultaneous,
            bow_starting=starting_position,
            init_file=None,
            use_muscles=with_muscles,
            fatigable=with_fatigue,
            minimize_fatigue=minimize_fatigue,
            time_per_cycle=cycle_time,
            n_shooting_per_cycle=full_cycle,
            solver=solver,
            ode_solver=ode_solver,
            n_threads=n_threads,
        )
        ocp_pre.set_bow_target_objective(bow_trajectory.target)
        ocp_pre.set_cyclic_bound(0.01)
        sol_pre = ocp_pre.solve(limit_memory_max_iter=50, exact_max_iter=0, force_no_graph=True)

    def nmpc_update_function(ocp, t, sol):
        if t >= n_cycles:
            print("Finished optimizing!")
            return False

        print(f"\n\nOptimizing cycle #{t + 1}..")
        if sol is not None:
            nmpc_violin.save(sol, ext=f"tmp_{save_name}_{t}", stand_alone=True)
        if window != full_cycle:
            _t = 0  # Cyclic so t should always be the start
            target_time_index = [i % full_cycle for i in range(_t, _t + window * n_cycles_simultaneous + 1)]
            nmpc_violin.set_bow_target_objective(bow_trajectory.target[:, target_time_index])
        return True

    save_name = f"{n_cycles}_cycles{'_with_fatigue' if with_fatigue else ''}"
    sol = nmpc_violin.solve(nmpc_update_function, sol_pre, show_online=False, cycle_from=cycle_from)

    # Data output
    nmpc_violin.save(sol, ext=save_name)
    nmpc_violin.save(sol, ext=save_name, stand_alone=True)
    print(f"Running time: {time() - tic} seconds")
    sol.print()


if __name__ == "__main__":
    main()
