from studies import Study, Conditions, DataType


def main():
    """
    Prepare and solve and animate a reaching task ocp
    """

    study = Study(Conditions.STUDY1)

    # --- Solve the program --- #
    study.run()

    study.generate_latex_table()
    study.prepare_plot_data(DataType.STATES, "q")
    study.plot()


if __name__ == "__main__":
    main()
