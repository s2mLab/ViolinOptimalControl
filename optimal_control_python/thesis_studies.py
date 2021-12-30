from enum import Enum, auto
import numpy as np

from violin_ocp.violin import Violin, ViolinString
from thesis_studies_misc.fatigue_integrator import FatigueIntegrator
from thesis_studies_misc.fatigue_models import FatigueModels
from thesis_studies_misc.target_functions import TargetFunctions


class Study(Enum):
    XIA_ONLY = auto()


def main():
    # Defines the global options
    violin = Violin("WuViolin", ViolinString.E)
    target = 0.2
    t_end = 60
    max_target = 1
    n_points = 100000

    # Define the specific options
    study = Study.XIA_ONLY

    # Declare some important variables for the study
    t = np.linspace(0, t_end, n_points)
    tf = TargetFunctions(t_end, target)
    fm = FatigueModels(violin, max_target)
    target_function, fatigue_models = study_chooser(tf, fm, study)

    # Prepare and run the integrator
    runner = FatigueIntegrator(t=t, target=target_function, fatigue_models=fatigue_models)
    runner.perform()

    # Print some results
    runner.print_integration_time()
    runner.plot_results()


def study_chooser(tf: TargetFunctions, fm: FatigueModels, study: Study):
    """
    Choose the study to perform

    Parameters
    ----------
    tf: TargetFunctions
        The instantiated TargetFunctions
    fm: FatigueModels
        The instantiated FatigueModels
    study: Study
        The study to perform
    """

    if study == Study.XIA_ONLY:
        target_func = tf.TARGET_UP_TO_MID_THEN_ZERO
        fatigue_models = fm.XIA,
    else:
        raise ValueError("Wrong choice of study")
    return target_func, fatigue_models


if __name__ == "__main__":
    main()
