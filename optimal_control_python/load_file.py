from biorbd_optim import OptimalControlProgram, ShowResult

ocp, sol = OptimalControlProgram.load(biorbd_model_path="../models/BrasViolon.bioMod", name="up_and_down_5_constraints")
result = ShowResult(ocp, sol)
result.graphs()
