from bioptim import Solution


class Video:
    def __init__(self, cycle_from: int, cycle_to: int):
        self.cycle_from = cycle_from
        self.cycle_to = cycle_to

    @staticmethod
    def generate_video(all_solutions: list[tuple[Solution, list[Solution, ...]], ...], save_path: str):
        for solution, all_iterations in all_solutions:
            solution.animate(
                show_meshes=True,
                show_global_center_of_mass=False,
                show_gravity_vector=False,
                show_floor=False,
                show_segments_center_of_mass=False,
                show_global_ref_frame=False,
                show_local_ref_frame=False,
                show_markers=True,
                show_muscles=False,
                show_wrappings=False,
                background_color=(0, 1, 0),
            )
