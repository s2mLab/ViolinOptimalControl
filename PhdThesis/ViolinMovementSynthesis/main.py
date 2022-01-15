from studies import StudyConfig


def main():

    all_studies = (
        StudyConfig.STUDY1_OCP,
        StudyConfig.STUDY2_TAU_10_CYCLES,
        StudyConfig.STUDY3_TAU_10_CYCLES_3_AT_A_TIME,
        StudyConfig.STUDY4_VIOLIN,
    )

    for study in all_studies:
        # Perform the study
        study.perform(limit_memory_max_iter=100, exact_max_iter=1000, show_graphs=False)

        # Print the results
        study.save_solutions()
        study.generate_latex_table()
        print("----------")


if __name__ == "__main__":
    main()
