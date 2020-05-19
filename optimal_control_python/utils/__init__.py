import numpy as np


class Bow:
    """
    Contains references from useful markers.
    """

    segment_idx = 8
    contact_marker = 19
    frog_marker = 16
    tip_marker = 18

    def __init__(self, bow_side):
        """
        Contains the side of the bow.
        """
        if bow_side not in ["frog", "tip"]:
            raise RuntimeError(bow_side + " is not a valid side of bow, it must be frog or tip.")
        self.bow_side = bow_side

    @property
    def side(self):
        return self.bow_side


class Violin:
    """
    Contains initial values and references from useful markers and segments.
    """

    segment_idx = 17

    def __init__(self, string):
        """
        Contains some references and values specific to the string.
        :param string: violin string letter
        :param bow_side: side of the bow, "frog" or "tip".
        """
        if string not in ["E", "A", "D", "G"]:
            raise RuntimeError(string + " is not a valid string, it must be E, A, D or G. Do you know violin ?")
        self.string = string

    def initial_position(self):
        """
        :return: List of initial positions according to the string and the side of the bow.
        """
        return {
            "E": {
                "frog": [-0.2908, -0.4622, 0.6952, 1.1347, 1.4096, -0.1030, 0.1516, -0.2379, -0.2633, 0],
                "tip": [0.0876, -0.5649, 0.6498, 1.0598, -0.1866, 0.2434, 0.1582, 0.2087, 0.7162, 0],
            },
            "A": {
                "frog": [-0.1569, -0.5216, 0.5900, 1.1063, 1.4728, 0.0393, 0.3143, -0.3959, -0.4446, 0],
                "tip": [0.0305, -0.6904, 0.3695, 0.8809, 0.1557, 0.2997, 0.2071, 0.1471, 0.5546, 0],
            },
            "D": {
                "frog": [-0.1259, -0.4520, 0.5822, 1.1106, 1.4595, 0.1194, 0.5033, -0.4040, -0.4567, 0],
                "tip": [0.0378, -0.7034, 0.2345, 0.947, 0.1111, 0.4134, 0.2470, 0.2606, 0.4842, 0],
            },
            "G": {
                "frog": [-0.2697, -0.3733, 0.5529, 1.1676, 1.5453, 0.0877, 0.6603, -0.5842, -0.6424, 0],
                "tip": [-0.0182, -1.3112, 0.1928, 0.6092, 0.7065, -0.0755, 0.1720, 0.1136, 0.2626, 0],
            },
        }[self.string]

    @property
    def bridge_marker(self):
        """
        :return: Marker number on the bridge, associate to the string.
        """
        return {"E": 35, "A": 37, "D": 39, "G": 41,}[self.string]

    @property
    def neck_marker(self):
        """
        :return: Marker number on the neck of the violin, associate to the string.
        """
        return {"E": 36, "A": 38, "D": 40, "G": 42, }[self.string]

    @property
    def rt_on_string(self):
        """
        :return: RT number according to the string.
        """
        return {"E": 3, "A": 2, "D": 1, "G": 0, }[self.string]

    @property
    def external_force(self):
        return {"E": np.array([0., 0., 0., 0.40989355, 1.84413989, 0.65660896]),
                "A": np.array([0., 0., 0., 0.30881124, 1.65124622, 1.08536701]),
                "D": np.array([0., 0., 0., 0.16081784, 1.30189937, 1.50970052]),
                "G": np.array([0., 0., 0., 0.05865013, 1.05013794, 1.7011086])}[self.string]
