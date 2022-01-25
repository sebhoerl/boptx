import numpy as np
import scipy.linalg as la

from ..algorithm import SampleProcessAlgorithm
from ..evaluator import Evaluator
from ..problem import Problem

import logging
logger = logging.getLogger(__name__)

class SensitivityAlgorithm(SampleProcessAlgorithm):
    def __init__(self, problem: Problem, step_lengths, steps):
        # Problem
        self.problem = problem
        self.initial = self._require_initial_values(self.problem.get_parameters())

        dimensions = self._get_dimensions(problem)

        assert len(steps) == dimensions
        assert len(step_lengths) == dimensions

        # Algorithm settings
        self.steps = steps
        self.step_lengths = step_lengths
        self.dimensions = dimensions

        # Algorithm state
        self.done = False

    def set_state(self, state):
        pass

    def get_state(self):
        return {
            "done": done
        }

    def get_settings(self):
        return {
            "step_lengths": self.step_lengths,
            "steps": self.steps,
        }

    def sample(self):
        if self.done: return []

        values = np.repeat([self.initial], 1 + np.sum(self.steps) * 2, axis = 0)

        # First is base configuration
        k = 1

        for d in range(self.dimensions):
            sigma = self.step_lengths[d]
            steps = self.steps[d]

            offsets = np.linspace(0.0, sigma * steps, steps + 1)

            for offset in offsets:
                values[k + 0, d] += offset
                values[k + 1, d] -= offset
                k += 2

        return values

    def process(self, evaluations):
        self.done = True
