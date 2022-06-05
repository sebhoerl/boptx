from boptx.problem import Problem, ContinuousParameter
from boptx.matsim import MATSimProblem

import numpy as np
import pandas as pd

def scale_bounds(bounds, scaling):
    return (bounds[0] * scaling, bounds[1] * scaling)

def apply_bounds(bounds, value):
    return min(max(value, bounds[0]), bounds[1])

class ModeParameter(ContinuousParameter):
    def __init__(self, parameter, bounds = (-5.0, 5.0), initial_value = 0.0, scaling = 1.0):
        super().__init__("Modal({})".format(parameter), scale_bounds(bounds, scaling), initial_value * scaling)
        self.parameter = parameter
        self.scaling = scaling

    def implement(self, settings, value):
        settings["arguments"] += ["--mode-choice-parameter:%s" % self.parameter, str(apply_bounds(self.bounds, value / self.scaling))]

class CostParameter:
    def __init__(self, parameter, bounds = (0.0, 5.0), initial_value = 1.0, scaling = 1.0):
        super().__init__("Cost({})".format(parameter), scale_bounds(bounds, scaling), initial_value * scaling)
        self.parameter = parameter
        self.scaling = scaling

    def implement(self, settings, value):
        settings["arguments"] += ["--cost-parameter:%s" % self.parameter, str(apply_bounds(self.bounds, value / self.scaling))]

class LineSwitchParameter:
    def __init__(self, bounds = (-5.0, 0.0), initial_value = -0.1, scaling = 1.0):
        super().__init__("LineSwitch", scale_bounds(bounds, scaling), initial_value * scaling)
        self.scaling = scaling

    def implement(self, settings, value):
        settings["arguments"] += ["--line-switch-utility", str(apply_bounds(self.bounds, value / self.scaling))]

class OsmSpeedParameter(ContinuousParameter):
    def __init__(self, slot, bounds = (0.1, 2.0), initial_value = 1.0, scaling = 1.0):
        super().__init__("OsmSpeed({})".format(slot), scale_bounds(bounds, scaling), initial_value * scaling)
        self.slot = slot
        self.scaling = scaling

    def implement(self, settings, value):
        settings["arguments"] += ["--osm-speed:%s" % self.slot, str(apply_bounds(self.bounds, value / self.scaling))]

class OsmCapacityParameter(ContinuousParameter):
    def __init__(self, slot, bounds = (0.1, 2.0), initial_value = 1.0, scaling = 1.0):
        super().__init__("OsmCapacity({})".format(slot), scale_bounds(bounds, scaling), initial_value * scaling)
        self.slot = slot
        self.scaling = scaling

    def implement(self, settings, value):
        settings["arguments"] += ["--osm-capacity:%s" % self.slot, str(apply_bounds(self.bounds, value / self.scaling))]

class CapacityParameter(ContinuousParameter):
    def __init__(self, bounds = (0.1, 2.0), initial_value = 1.0, scaling = 1.0):
        super().__init__("Capacity", scale_bounds(bounds, scaling), initial_value * scaling)
        self.scaling = scaling

    def implement(self, settings, value):
        for slot in ("major", "intermediate", "minor"):
            settings["arguments"] += ["--osm-capacity:%s" % slot, str(apply_bounds(self.bounds, value / self.scaling))]

class CalibrationProblem(MATSimProblem):
    def __init__(self, objective, parameters):
        self.objective = objective
        self.parameters = parameters

    def get_parameters(self):
        return self.parameters

    def get_state_count(self):
        return self.objective.get_state_count()

    def parameterize(self, settings, values, information):
        for parameter, value in zip(self.parameters, values):
            parameter.implement(settings, value)

    def process(self, output_path, values, information):
        information = self.objective.calculate(output_path)

        return dict(
            objective = information["objective"],
            states = information["states"],
            information = information,
        )
