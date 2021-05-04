from time import time

from violin_ocp import Violin, ViolinString, ViolinOcp, Bow, BowTrajectory, BowPosition
from bioptim import Solver


if __name__ == "__main__":
    model_name = "WuViolin"
    violin = Violin(model_name, ViolinString.E)
    bow = Bow(model_name)

    # --- Solve the program --- #
    # ocp, sol = ViolinOcp.load("results/2021_4_9.bo")
    window = 30
    full_cycle = 30
    cycle_time = 1
    n_cycles = 3
    solver = Solver.ACADOS
    ocp_violin = ViolinOcp(
        f"../models/{model_name}.bioMod",
        violin,
        bow,
        1,
        BowPosition.TIP,
        use_muscles=False,
        solver=solver,
        n_shooting_per_cycle=window-1,
        time_per_cycle=cycle_time * window / full_cycle
    )

    # Generate a full cycle target
    lim = bow.hair_limits if ocp_violin.bow_starting == BowPosition.FROG else [bow.hair_limits[1], bow.hair_limits[0]]
    bow_trajectory = BowTrajectory(lim, full_cycle + 1)
    # bow_target = np.tile(bow_trajectory.target[:, :-1], ocp.n_cycles)
    # bow_target = np.concatenate((bow_target, bow_trajectory.target[:, -1][:, np.newaxis]), axis=1)
    bow_trajectory.target = bow_trajectory.target[:, :-1]


    def mhe_update_function(ocp, sol, t):
        if t >= n_cycles * full_cycle - window + 1:
            return False
        target_time_index = [i % full_cycle for i in range(t, t + window)]
        ocp_violin.set_bow_target_objective(bow_trajectory.target[:, target_time_index])
        return True

    tic = time()
    full_sol = ocp_violin.ocp.solve_mhe(mhe_update_function, solver=solver)
    print(f"Running time: {time() - tic}")

    # ocp.save(sol)
    # ocp.save(sol, stand_alone=True)

    # sol.print()
    full_sol.graphs()
    full_sol.animate(show_meshes=False)
