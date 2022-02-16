from studies import Study, Conditions, DataType


def main():
    all_studies = (Study(Conditions.STUDY1),)

    # --- Solve the program --- #
    for study in all_studies:
        study.run()

        study.generate_latex_table()
        study.save_solutions()
        study.prepare_plot_data(DataType.STATES, "q")
        # study.plot()
        print("----------------")


if __name__ == "__main__":
    main()
