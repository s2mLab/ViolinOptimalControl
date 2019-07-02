from pyoviz.BiorbdViz import BiorbdViz

model_name = "Model"

b = BiorbdViz(model_path=f"../models/{model_name}.bioMod", show_muscles=False, show_meshes=False)

b.exec()