from feasibility_studies import FatigueIntegrator
from studies import Study


def main():
    # Define the study to perform
    all_studies = [Study.STUDY3_0_EFFORT_VS_MICHAUD_LONG]

    # Prepare and run the integrator
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
