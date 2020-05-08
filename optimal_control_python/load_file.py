from biorbd_optim import OptimalControlProgram, ShowResult

ocp, sol = OptimalControlProgram.load(biorbd_model_path="../models/BrasViolon.bioMod", name="results/2020_5_7_up_and_down_5_constraints.bo")
result = ShowResult(ocp, sol)
result.animate(show_meshes=False)
