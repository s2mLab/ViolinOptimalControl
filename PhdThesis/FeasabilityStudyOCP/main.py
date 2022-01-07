from studies import Study


def main():
    """
    Prepare and solve and animate a reaching task ocp
    """

    ocp = Study.MUSCLE_DRIVEN_MICHAUD

    # --- Solve the program --- #
    sol = ocp.perform()
    sol.print()

    # --- Show results --- #
    sol.animate(show_meshes=True, show_floor=False)


if __name__ == "__main__":
    main()
