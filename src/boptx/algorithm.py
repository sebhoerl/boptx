import numpy as np

from .evaluator import Evaluator

class Algorithm:
    def __init__(self):
        raise NotImplementedError()

    def advance(self, evaluator: Evaluator):
        raise NotImplementedError()

    def get_state(self):
        raise NotImplementedError()

    def set_state(self):
        raise NotImplementedError()

    def get_settings(self):
        raise NotImplementedError()

    def _require_bounds(self, parameters):
        for parameter in parameters:
            if parameter.get_bounds() is None:
                raise RuntimeError("Expected bounds for parameter {}".format(parameter.get_name()))

    def _warn_ignore_bounds(self, parameters, logger):
        for parameter in parameters:
            if not parameter.get_bounds() is None:
                logger.warn("Not taking into account bounds for parameter {}".format(parameter.get_name()))

    def _require_initial_values(self, parameters):
        values = []

        for parameter in parameters:
            if parameter.get_initial_value() is None:
                raise RuntimeError("Expected initial value for parameter {}".format(parameter.get_name()))
            else:
                values.append(parameter.get_initial_value())

        return values

    def _get_dimensions(self, problem):
        return len(problem.get_parameters())

    def _truncate_bounds(self, problem, values):
        values = np.copy(values)
        assert values.shape[1] == len(problem.get_parameters())

        for k in range(values.shape[1]):
            bounds = problem.get_parameters()[k].get_bounds()

            if not bounds is None:
                if not bounds[0] is None:
                    values[:,k] = np.maximum(values[:,k], bounds[0])

                if not bounds[1] is None:
                    values[:,k] = np.minimum(values[:,k], bounds[1])

        return values

    def _check_bounds(self, problem, values):
        assert values.shape[1] == len(problem.get_parameters())
        valid = np.ones(values.shape, dtype = bool)

        for k in range(values.shape[1]):
            bounds = problem.get_parameters()[k].get_bounds()

            if not bounds is None:
                if not bounds[0] is None:
                    valid[:,k] &= values[:,k] >= bounds[0]

                if not bounds[1] is None:
                    valid[:,k] &= values[:,k] <= bounds[1]

        return np.all(valid, axis = 1)

class SampleProcessAlgorithm(Algorithm):
    def advance(self, evaluator: Evaluator):
        identifiers = []

        for sample in self.sample():
            if isinstance(sample, tuple):
                identifiers.append(evaluator.submit([sample[0]], sample[1])[0])
            else:
                identifiers.append(evaluator.submit([sample])[0])

        evaluations = evaluator.get(identifiers)
        self.process(evaluations)
        evaluator.clean(identifiers)

    def sample(self):
        raise NotImplementedError()

    def process(self, evaluations):
        raise NotImplementedError()
