from pyoviz.BiorbdViz import BiorbdViz

model_name = "eocar"

b = BiorbdViz(model_path=f"../models/{model_name}.bioMod", show_muscles=False)

b.exec()