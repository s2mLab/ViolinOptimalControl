from enum import Enum

from feasibility_studies import FatigueIntegrator, StudyConfiguration, FatigueModels, TargetFunctions, FatigueParameters


class Study(Enum):
    # DEBUG OPTIONS
    XIA_ONLY = StudyConfiguration(
        fatigue_parameters=FatigueParameters(),
        t_end=600,
        fixed_target=0.2,
        target_function=TargetFunctions.TARGET_UP_TO_END,
        n_points=100000,
        x0=((0, 0.8, 0, 0),),
        fatigue_models=(FatigueModels.XIA,),
    )

    XIA_STABILIZED_ONLY = StudyConfiguration(
        fatigue_parameters=FatigueParameters(),
        t_end=10,
        fixed_target=0.2,
        target_function=TargetFunctions.TARGET_UP_TO_END,
        n_points=100000,
        fatigue_models=(FatigueModels.XIA_STABILIZED,),
        x0=((0, 0.5, 0, 0),),
        plot_options=({"linestyle": "-"},),
    )

    # Actual studies from the thesis
    STUDY1_XIA_LONG = StudyConfiguration(
        fatigue_parameters=FatigueParameters(stabilization_factor=100),
        t_end=3600,
        fixed_target=1,
        target_function=TargetFunctions.TARGET_RANDOM_PER_10SECONDS,
        n_points=100000,
        fatigue_models=(FatigueModels.XIA, FatigueModels.XIA_STABILIZED),
        x0=((0, 1, 0), (0, 1, 0),),
        rms_indices=((0, 1, 2), (0, 1, 2)),
        plot_options=({"linestyle": "-"}, {"linestyle": "--"}),
    )

    STUDY1_XIA_VS_STABILIZED = StudyConfiguration(
        fatigue_parameters=FatigueParameters(stabilization_factor=100),
        t_end=0.1,
        fixed_target=0.8,
        target_function=TargetFunctions.TARGET_UP_TO_END,
        n_points=1000,
        fatigue_models=(FatigueModels.XIA, FatigueModels.XIA_STABILIZED),
        x0=((0, 0.6, 0), (0, 0.6, 0)),
        plot_options=({"linestyle": "-"}, {"linestyle": "--"}),
    )


def main():

    # Define the study to perform
    study = Study.STUDY1_XIA_LONG

    # Prepare and run the integrator
    runner = FatigueIntegrator(study.value)
    runner.perform()

    # Print some results
    runner.print_final_sum()
    runner.print_integration_time()
    if len(runner.study.fatigue_models.models) == 2:
        runner.print_rmse()
    runner.plot_results()


if __name__ == "__main__":
    main()
