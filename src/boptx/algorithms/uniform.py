import numpy as np

from ..algorithm import SampleProcessAlgorithm
from ..evaluator import Evaluator
from ..problem import Problem

class UniformAlgorithm(SampleProcessAlgorithm):
    def __init__(self, problem: Problem, seed = 0, candidates = 1):
        # Problem
        self.problem = problem
        self._require_bounds(self.problem.get_parameters())

        # Algorithm settings
        self.candidates = candidates
        self.seed = seed

        # Algorithm state
        self.random = np.random.RandomState(self.seed)

    def set_state(self, state):
        self.random.set_state(state["random"])

    def get_state(self):
        return {
            "random": self.random.get_state()
        }

    def get_settings(self):
        return {
            "seed": self.seed,
            "candidates": self.candidates
        }

    def sample(self):
        parameters = self.problem.get_parameters()
        values = self.random.random_sample(size = (self.candidates, len(parameters)))

        for k, parameter in enumerate(parameters):
            bounds = parameter.get_bounds()

            values[:,k] *= (bounds[1] - bounds[0])
            values[:,k] += bounds[0]

        return values

    def process(self, evaluations):
        pass
