from typing import Union
from enum import Enum, auto

from bioptim import DynamicsFcn as BioptimDynamicsFcn


class PlotOptions:
    def __init__(
        self,
        title: str,
        legend_indices: Union[tuple[bool, ...], None],
        options: tuple[dict, ...],
        maximize: bool = False,
        save_path: Union[tuple[str, ...], None] = None,
    ):
        self.title = title
        self.legend_indices = legend_indices
        self.options = options
        self.maximize = maximize
        self.save_path = save_path


class DataType(Enum):
    STATES = "states"
    CONTROLS = "controls"


class DynamicsFcn(Enum):
    TORQUE_DRIVEN = BioptimDynamicsFcn.TORQUE_DRIVEN
    MUSCLE_DRIVEN = BioptimDynamicsFcn.MUSCLE_DRIVEN


class FatigableStructure(Enum):
    MUSCLES = auto()
    JOINTS = auto()
