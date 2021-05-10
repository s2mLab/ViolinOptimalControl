import numpy as np

from violin_ocp import Violin, ViolinString, ViolinOcp, Bow, BowTrajectory, BowPosition
from bioptim import Solver


if __name__ == "__main__":
    model_name = "WuViolin"
    violin = Violin(model_name, ViolinString.E)
    bow = Bow(model_name)

    # --- Solve the program --- #
    n_shoot_per_cycle = 30
    cycle_time = 1
    n_cycles = 1
    solver = Solver.IPOPT
    n_threads = 2
    ocp = ViolinOcp(
        model_path=f"../models/{model_name}.bioMod",
        violin=violin,
        bow=bow,
        n_cycles=n_cycles,
        bow_starting=BowPosition.TIP,
        init_file=None,
        use_muscles=False,
        fatigable=True,
        time_per_cycle=cycle_time,
        n_shooting_per_cycle=n_shoot_per_cycle,
        solver=solver,
        n_threads=n_threads
    )
    # ocp, sol = ViolinOcp.load("results/5_cycles_34_muscles/2021_3_12.bo")

    lim = bow.hair_limits if ocp.bow_starting == BowPosition.FROG else [bow.hair_limits[1], bow.hair_limits[0]]
    bow_trajectory = BowTrajectory(lim, ocp.n_shooting_per_cycle + 1)
    bow_target = np.tile(bow_trajectory.target[:, :-1], ocp.n_cycles)
    bow_target = np.concatenate((bow_target, bow_trajectory.target[:, -1][:, np.newaxis]), axis=1)
    ocp.set_bow_target_objective(bow_target)

    sol = ocp.solve(
        show_online_optim=True,
        solver_options={"max_iter": 1000, "hessian_approximation": "limited-memory", "linear_solver": "ma57"},
    )
    ocp.save(sol)
    ocp.save(sol, stand_alone=True)

    sol.print()
    sol.animate(show_meshes=False)
