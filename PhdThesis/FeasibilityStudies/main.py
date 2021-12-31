from enum import Enum

from feasibility_studies import FatigueIntegrator, StudyConfiguration, FatigueModels, TargetFunctions, FatigueParameters


class Study(Enum):

    XIA_ONLY = StudyConfiguration(
        fatigue_parameters=FatigueParameters(),
        t_end=600,
        fixed_target=0.2,
        target_function=TargetFunctions.TARGET_UP_TO_END,
        n_points=100000,
        fatigue_models=(FatigueModels.XIA,),
    )

    XIA_AND_STABILIZED = StudyConfiguration(
        fatigue_parameters=FatigueParameters(),
        t_end=600,
        fixed_target=0.2,
        target_function=TargetFunctions.TARGET_UP_TO_END,
        n_points=100000,
        fatigue_models=(FatigueModels.XIA, FatigueModels.XIA_STABILIZED),
    )


def main():

    # Define the study to perform
    study = Study.XIA_ONLY

    # Prepare and run the integrator
    runner = FatigueIntegrator(study.value)
    runner.perform()

    # Print some results
    runner.print_integration_time()
    runner.plot_results()


if __name__ == "__main__":
    main()
