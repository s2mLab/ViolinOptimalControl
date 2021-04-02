import numpy as np

from violin_ocp import Violin, ViolinString, ViolinOcp, Bow, BowTrajectory, BowPosition


if __name__ == "__main__":
    model_name = "WuViolin"
    violin = Violin(model_name, ViolinString.E)
    bow = Bow(model_name)

    # --- Solve the program --- #
    # ocp, sol = ViolinOcp.load("results/5_cycles_34_muscles/2021_3_12.bo")
    ocp = ViolinOcp(f"../models/{model_name}.bioMod", violin, bow, 3, BowPosition.TIP, use_muscles=True)

    lim = bow.hair_limits if ocp.bow_starting == BowPosition.FROG else [bow.hair_limits[1], bow.hair_limits[0]]
    bow_trajectory = BowTrajectory(lim, ocp.n_shooting_per_cycle + 1)
    bow_target = np.tile(bow_trajectory.target[:, :-1], ocp.n_cycles)
    bow_target = np.concatenate((bow_target, bow_trajectory.target[:, -1][:, np.newaxis]), axis=1)
    ocp.set_bow_target_objective(bow_target)

    sol = ocp.solve(
        show_online_optim=True,
        solver_options={
            "max_iter": 1000,
            "hessian_approximation": "exact",
            "linear_solver": "ma57"
        },
    )
    ocp.save(sol)
    ocp.save(sol, stand_alone=True)

    sol.print()
    sol.animate(show_meshes=False)
