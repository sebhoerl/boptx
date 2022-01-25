import numpy as np

from ..algorithm import SampleProcessAlgorithm
from ..evaluator import Evaluator
from ..problem import Problem

import logging
logger = logging.getLogger(__name__)

class EvolutionarySearchAlgorithm(SampleProcessAlgorithm):
    def __init__(self, problem: Problem, sigma = 0.1, candidates = 1, always_replace = False, seed = 0):
        # Problem
        self.problem = problem
        self._require_initial_values(self.problem.get_parameters())

        # Algorithm settings
        self.sigma = sigma
        self.always_replace = always_replace
        self.candidates = candidates
        self.seed = seed
        self.dimensions = self._get_dimensions(problem)

        # Algorithm state
        self.random = np.random.RandomState(self.seed)

        self.mean = np.array(self._require_initial_values(self.problem.get_parameters()))
        self.objective = np.inf

    def set_state(self, state):
        assert self.dimensions == len(state["mean"])

        self.random.set_state(state["random"])
        self.mean = state["mean"]
        self.objective = state["objective"]

    def get_state(self):
        return {
            "random": self.random.get_state(),
            "mean": self.mean,
            "objective": self.objective,
        }

    def get_settings(self):
        return {
            "sigma": self.sigma,
            "always_replace": self.always_replace,
            "candidates": self.candidates,
            "seed": self.seed
        }

    def sample(self):
        values = self.random.normal(size = (self.candidates, self.dimensions)) * self.sigma
        values += self.mean.reshape((-1, self.dimensions))

        for k in range(self.dimensions):
            bounds = self.problem.get_parameters()[k].get_bounds()

            if not bounds is None:
                values[:,k] = np.maximum(values[:,k], bounds[0])
                values[:,k] = np.minimum(values[:,k], bounds[1])

        return values

    def process(self, evaluations):
        objectives = [e.get_objective() for e in evaluations]

        minimum_index = np.argmin(objectives)
        minimum_objective = objectives[minimum_index]

        if self.always_replace or minimum_objective < self.objective:
            self.objective = minimum_objective
            self.mean = evaluations[minimum_index].get_values()
