import bioviz
import biorbd_casadi as biorbd
import numpy as np

from optimal_control_python.violin_ocp.violin import Violin, ViolinString
from optimal_control_python.violin_ocp.bow import Bow, BowPosition


def main():
    # Options
    model_path = "../models/"
    model_name = "WuViolin"
    model = biorbd.Model(f"{model_path}/{model_name}.bioMod")
    violin = Violin(model=model_name, string=ViolinString.E)
    bow = Bow(model_name)
    bow_position = BowPosition.TIP
    q = violin.q(model, bow, bow_position)
    b = bioviz.Viz(model.path().absolutePath().to_string(), markers_size=0.003, show_markers=True)
    b.set_q(q)
    b.exec()


if __name__ == "__main__":
    main()
