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

    def implement(self, response, value):
        response["arguments"] += ["--mode-choice-parameter:%s" % self.parameter, str(apply_bounds(self.bounds, value / self.scaling))]

class CostParameter:
    def __init__(self, parameter, bounds = (0.0, 5.0), initial_value = 1.0, scaling = 1.0):
        super().__init__("Cost({})".format(parameter), scale_bounds(bounds, scaling), initial_value * scaling)
        self.parameter = parameter
        self.scaling = scaling

    def implement(self, response, value):
        response["arguments"] += ["--cost-parameter:%s" % self.parameter, str(apply_bounds(self.bounds, value / self.scaling))]

class LineSwitchParameter:
    def __init__(self, bounds = (-5.0, 0.0), initial_value = -0.1, scaling = 1.0):
        super().__init__("LineSwitch", scale_bounds(bounds, scaling), initial_value * scaling)
        self.scaling = scaling

    def implement(self, response, value):
        response["arguments"] += ["--line-switch-utility", str(apply_bounds(self.bounds, value / self.scaling))]

class OsmSpeedParameter(ContinuousParameter):
    def __init__(self, slot, bounds = (0.1, 2.0), initial_value = 1.0, scaling = 1.0):
        super().__init__("OsmSpeed({})".format(slot), scale_bounds(bounds, scaling), initial_value * scaling)
        self.slot = slot
        self.scaling = scaling

    def implement(self, response, value):
        response["arguments"] += ["--osm-speed:%s" % self.slot, str(apply_bounds(self.bounds, value / self.scaling))]

class OsmCapacityParameter(ContinuousParameter):
    def __init__(self, slot, bounds = (0.1, 2.0), initial_value = 1.0, scaling = 1.0):
        super().__init__("OsmCapacity({})".format(slot), scale_bounds(bounds, scaling), initial_value * scaling)
        self.slot = slot
        self.scaling = scaling

    def implement(self, response, value):
        response["arguments"] += ["--osm-capacity:%s" % self.slot, str(apply_bounds(self.bounds, value / self.scaling))]

class CapacityParameter(ContinuousParameter):
    def __init__(self, bounds = (0.1, 2.0), initial_value = 1.0, scaling = 1.0):
        super().__init__("Capacity", scale_bounds(bounds, scaling), initial_value * scaling)
        self.scaling = scaling

    def implement(self, response, value):
        for slot in ("major", "intermediate", "minor"):
            response["arguments"] += ["--osm-capacity:%s" % slot, str(apply_bounds(self.bounds, value / self.scaling))]

class PenaltyCalculator:
    def calculate(self, parameters, values):
        raise NotImplementedError()

class NoopPenaltyCalculator:
    def calculate(self, parameters, values):
        return [0.0] * len(parameters)

class LinearPenaltyCalculator:
    def __init__(self, offset, factor):
        self.offset = offset
        self.factor = factor

    def calculate(self, parameters, values):
        penalties = []

        for p, v in zip(parameters, values):
            lower_violation = np.maximum(p.get_bounds()[0] - v, 0.0)
            uppper_violation = np.maximum(v - p.get_bounds()[1], 0.0)

            penalty = 0.0

            if lower_violation > 0.0:
                penalty += self.offset + self.factor * lower_violation

            if uppper_violation > 0.0:
                penalty += self.offset + self.factor * uppper_violation

            penalties.append(penalty)

        return penalties

class CalibrationProblem(MATSimProblem):
    def __init__(self, objective, parameters, penalty : PenaltyCalculator = NoopPenaltyCalculator()):
        self.parameters = parameters
        self.objective = objective
        self.penalty = penalty

    def get_parameters(self):
        return self.parameters

    def get_settings(self):
        return dict()

    def get_state_count(self):
        return self.objective.get_state_count()

    def parameterize(self, values):
        if np.sum(self.penalty.calculate(self.parameters, values)) > 0.0:
            return { "skip": True }

        response = { "arguments": [], "config": {} }

        for parameter, value in zip(self.parameters, values):
            parameter.implement(response, value)

        return response

    def evaluate(self, values, path):
        penalties = self.penalty.calculate(self.parameters, values)

        if np.sum(penalties) > 0.0:
            return np.sum(penalties), {
                "penalties": penalties
            }

        information = self.objective.calculate(path)
        return information["objective"], information

if __name__ == "__main__":
    from objectives import FlowObjective

    flow_objective = FlowObjective(
        "data/hourly_flow.csv",
        objective = "l1"
    )

    problem = CalibrationProblem(
        flow_objective, [
            ModeParameter("car.alpha_u", (-5.0, 5.0)),
            OsmCapacityParameter("primary")
        ], LinearPenaltyCalculator(1000.0, 1.0)
    )

    # print("INFORMATION", problem.get_information())
    print("PARAMETERIZE", problem.parameterize([2.0, 2.0]))
    print("EVALUATE", problem.evaluate([2.0, 2.0], "data/test_output"))
