from feasibility_studies import FatigueIntegrator
from studies import Study


def main():
    # Define the studies to perform
    all_studies = (
        Study.STUDY1_1_XIA_STABILIZED,
        Study.STUDY1_2_XIA_VS_STABILIZED_GOOD_X0,
        Study.STUDY1_3_XIA_VS_STABILIZED_BAD_X0,
        Study.STUDY1_4_XIA_STABILIZED_FATIGUE_NEGATIVE,
        Study.STUDY2_1_MICHAUD_LONG,
        Study.STUDY2_2_MICHAUD_VELOCITY_COMPARISON,
        Study.STUDY2_3_MICHAUD_THRESHOLD_COMPARISON,
        Study.STUDY2_4_MICHAUD_VS_XIA,
        Study.STUDY3_0_EFFORT_VS_MICHAUD_LONG,
    )

    # Prepare and run the studies
    for i, study in enumerate(all_studies):
        print(f"Study #{i}")
        runner = FatigueIntegrator(study.value)
        runner.perform()

        # Print some results
        runner.print_integration_time()
        if len(runner.study.fatigue_models) == 2:
            runner.print_rmse()
        runner.print_custom_analyses()
        runner.plot_results()
        print("----------------")


if __name__ == "__main__":
    main()
