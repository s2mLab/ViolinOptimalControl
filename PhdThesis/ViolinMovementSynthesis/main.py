from studies import StudyConfig


def main():
    study = StudyConfig.DEBUG_TAU_NO_FATIGUE

    # Perform the study
    study.perform(limit_memory_max_iter=100, exact_max_iter=1000)

    # Print the results
    study.generate_latex_table()


if __name__ == "__main__":
    main()
