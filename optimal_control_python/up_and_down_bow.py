import numpy as np

from violin_ocp import Violin, ViolinString, ViolinOcp, Bow, BowTrajectory, BowPosition


if __name__ == "__main__":
    violin = Violin(ViolinString.E)
    bow = Bow()

    # --- Solve the program --- #
    ocp, sol = ViolinOcp.load("results/2021_3_1.bo")
    # ocp = ViolinOcp("../models/BrasViolon.bioMod", violin, bow, 3, BowPosition.TIP, use_muscles=True, init_file="results/2021_3_1.bo")
    #
    # lim = bow.hair_limits if ocp.bow_starting == BowPosition.FROG else [bow.hair_limits[1], bow.hair_limits[0]]
    # bow_trajectory = BowTrajectory(lim, ocp.n_shooting_per_cycle + 1)
    # bow_target = np.tile(bow_trajectory.target[:, :-1], ocp.n_cycles)
    # bow_target = np.concatenate((bow_target, bow_trajectory.target[:, -1][:, np.newaxis]), axis=1)
    # ocp.set_bow_target_objective(bow_target)
    #
    # sol = ocp.solve(
    #     show_online_optim=True,
    #     solver_options={
    #         "tol": 1e-6,
    #         "max_iter": 1000,
    #         "hessian_approximation": "exact",
    #         "linear_solver": "ma57"
    #     },
    # )
    # ocp.save(sol)

    sol.print()
    sol.graphs()
    sol.animate(show_meshes=True)
