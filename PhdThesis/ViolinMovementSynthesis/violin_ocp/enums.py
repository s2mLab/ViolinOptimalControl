from enum import Enum, auto

from bioptim import EffortPerception, MichaudFatigue


class StructureType(Enum):
    TAU = auto
    MUSCLE = auto()


class FatigueType(Enum):
    NO_FATIGUE = None
    EFFORT_PERCEPTION = EffortPerception
    QCC = MichaudFatigue


class DataType(Enum):
    STATES = "states"
    CONTROLS = "controls"
