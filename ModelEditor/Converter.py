import numpy as np
import math
from Modelizer import *


class ConvertModel:
    def __init__(self, model_to_convert, default_model=BiorbdModel("../models/Bras.bioMod")):
        if type(model_to_convert) == str:
            self.converted_model = BiorbdModel(model_to_convert)
            self.converted_model.read()
        else:
            self.converted_model = model_to_convert
        if type(default_model) == str:
            self.default_model = BiorbdModel(default_model)
            self.default_model.read()
        else:
            self.default_model = default_model

    def get_converted_model(self):
        return self.converted_model

    def remodel(self, default_segment_name, segment_to_convert_name):
        segment_to_convert_index = self.converted_model.get_segment_index(segment_to_convert_name)
        segment = self.converted_model.get_segments()[segment_to_convert_index]
        segment_default_index = self.default_model.get_segment_index(default_segment_name)
        new_length = self.default_model.get_segments()[segment_default_index].length()
        segment.set_length(new_length)
        self.converted_model.set_segment(segment_to_convert_index, segment)
        return self.converted_model


def main():
    model_to_convert = BiorbdModel()
    model_to_convert.read("../models/model_Clara/AdaJef_1g_Model.s2mMod")
    model_default = BiorbdModel()
    model_default.read("../models/Bras.bioMod")

    for segment in model_default.get_segments():
        segment_name = segment.get_name()
        print(segment_name, "-", segment.length())

    conversion = ConvertModel(model_to_convert, model_default)
    print("***")

    conversion.remodel("Pelvis", "Pelvis")
    conversion.remodel("Thorax", "Thorax")
    conversion.remodel("Clavicle", "ClaviculeRight")
    conversion.remodel("Scapula", "ScapulaRight")
    conversion.remodel("Arm", "ArmRight")
    conversion.remodel("LowerArm1", "LowerArm1Right")
    conversion.remodel("LowerArm2", "LowerArm2Right")
    conversion.remodel("hand", "HandRight")
    conversion.remodel("Clavicle", "ClaviculeLeft")
    conversion.remodel("Scapula", "ScapulaLeft")
    conversion.remodel("Arm", "ArmLeft")
    conversion.remodel("LowerArm1", "LowerArm1Left")
    conversion.remodel("LowerArm2", "LowerArm2Left")
    conversion.remodel("hand", "HandLeft")

    converted_model = conversion.get_converted_model()
    converted_model.write("../models/model_Clara/converted-AdaJef_1g_Model.bioMod")

    for segment in model_to_convert.get_segments():
        segment_name = segment.get_name()
        print(segment_name, "-", segment.length())

    return 0


if __name__ == "__main__":
    main()
