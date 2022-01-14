from studies import StudyConfig


def main():
    study = StudyConfig.STUDY1_OCP

    # Perform the study
    study.perform(limit_memory_max_iter=100, exact_max_iter=1000)

    # Print the results
    study.save_solutions()
    study.generate_latex_table()


if __name__ == "__main__":
    main()
