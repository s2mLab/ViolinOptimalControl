from enum import Enum, auto


class PlotOptions:
    def __init__(
        self,
        title: str = "",
        legend: tuple[str, ...] = None,
        supplementary_legend: tuple[str, ...] = None,
        options: tuple[dict, ...] = None,
        save_path: str = None,
    ):
        self.title = title
        self.legend = legend
        self.options = options
        self.save_path = save_path
        self.supplementary_legend = supplementary_legend


class CustomAnalysis:
    def __init__(self, name, fun):
        self.name = name
        self.fun = fun


class Integrator(Enum):
    RK4 = auto()
    RK45 = auto()
