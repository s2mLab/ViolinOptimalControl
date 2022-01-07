from enum import Enum, auto

from bioptim import DynamicsFcn as BioptimDynamicsFcn


class DynamicsFcn(Enum):
    TORQUE_DRIVEN = BioptimDynamicsFcn.TORQUE_DRIVEN
    MUSCLE_DRIVEN = BioptimDynamicsFcn.MUSCLE_DRIVEN


class FatigableStructure(Enum):
    MUSCLES = auto()
    JOINTS = auto()
