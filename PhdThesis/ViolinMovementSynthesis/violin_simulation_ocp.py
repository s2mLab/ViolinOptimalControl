import numpy as np

from violin_ocp import Violin, ViolinString, ViolinOcp, Bow, BowTrajectory, BowPosition, FatigueType, StructureType
from bioptim import Solver, OdeSolver


def main():
    model_name = "WuViolin"
    violin = Violin(model_name, ViolinString.G)
    bow = Bow(model_name)

    # --- Solve the program --- #
    starting_position = BowPosition.TIP
    n_shoot_per_cycle = 30
    n_cycles = 1
    cycle_time = 1
    solver = Solver.IPOPT()
    ode_solver = OdeSolver.RK4(n_integration_steps=3)  # OdeSolver.COLLOCATION(method="radau", polynomial_degree=4)  #
    n_threads = 8

    # Generate a full cycle target
    lim = bow.hair_limits if starting_position == BowPosition.FROG else [bow.hair_limits[1], bow.hair_limits[0]]
    bow_trajectory = BowTrajectory(lim, n_shoot_per_cycle + 1)
    bow_trajectory.target = np.tile(bow_trajectory.target[:, :-1], n_cycles)
    bow_trajectory.target = np.concatenate((bow_trajectory.target, bow_trajectory.target[:, -1][:, np.newaxis]), axis=1)

    ocp = ViolinOcp(
        model_path=f"models/{model_name}.bioMod",
        violin=violin,
        bow=bow,
        n_cycles=n_cycles,
        bow_starting=starting_position,
        structure_type=StructureType.TAU,
        fatigue_type=FatigueType.EFFORT_PERCEPTION,
        init_file=None,
        minimize_fatigue=True,
        time_per_cycle=cycle_time,
        n_shooting_per_cycle=n_shoot_per_cycle,
        solver=solver,
        ode_solver=ode_solver,
        n_threads=n_threads,
    )

    ocp.set_bow_target_objective(bow_trajectory.target)
    ocp.set_cyclic_bound(0.01)

    sol = ocp.solve(limit_memory_max_iter=100, exact_max_iter=1000, force_no_graph=True)

    # ocp.save(sol)
    # ocp.save(sol, stand_alone=True)

    sol.print()
    # sol.graphs()
    sol.animate(show_muscles=False, show_floor=False)


if __name__ == "__main__":
    main()
