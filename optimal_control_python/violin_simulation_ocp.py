import numpy as np

from violin_ocp import Violin, ViolinString, ViolinOcp, Bow, BowTrajectory, BowPosition
from bioptim import Solver, OdeSolver


def main():
    model_name = "WuViolin"
    violin = Violin(model_name, ViolinString.E)
    bow = Bow(model_name)

    # --- Solve the program --- #
    n_shoot_per_cycle = 40
    cycle_time = 1
    n_cycles = 1
    solver = Solver.IPOPT
    ode_solver = OdeSolver.RK4(n_integration_steps=5)  # OdeSolver.COLLOCATION(method="radau", polynomial_degree=8)  #
    n_threads = 8
    # ocp, sol = ViolinOcp.load("results/5_cycles_with_fatigue.bo")
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
        ode_solver=ode_solver,
        n_threads=n_threads,
    )

    lim = bow.hair_limits if ocp.bow_starting == BowPosition.FROG else [bow.hair_limits[1], bow.hair_limits[0]]
    bow_trajectory = BowTrajectory(lim, ocp.n_shooting_per_cycle + 1)
    bow_target = np.tile(bow_trajectory.target[:, :-1], ocp.n_cycles)
    bow_target = np.concatenate((bow_target, bow_trajectory.target[:, -1][:, np.newaxis]), axis=1)
    ocp.set_bow_target_objective(bow_target)

    sol = ocp.solve(limit_memory_max_iter=50, exact_max_iter=1000)

    #
    # ocp.save(sol)
    # ocp.save(sol, stand_alone=True)

    sol.print()
    # sol.graphs()
    sol.animate()


if __name__ == "__main__":
    main()
