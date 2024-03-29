from studies import StudyConfig


def main():

    reload_data = True
    all_studies = (
        StudyConfig.STUDY1_OCP,
        StudyConfig.STUDY2_TAU_10_CYCLES,
        StudyConfig.STUDY3_TAU_10_CYCLES_3_AT_A_TIME,
        StudyConfig.STUDY4A_VIOLIN,
        StudyConfig.STUDY4B_VIOLIN,
    )

    for study in all_studies:
        # Perform the study (or reload)
        study.perform(
            reload_if_exists=reload_data,
            limit_memory_max_iter=100,
            exact_max_iter=1000,
            show_graphs=False,
            save_solutions=True,
        )

        # Print the results
        study.generate_figures()
        study.perform_custom_analyses()
        study.generate_latex_table()
        study.generate_videos()
        study.generate_snapshots()
        study.generate_extra_figures()
        print("----------")


if __name__ == "__main__":
    main()
