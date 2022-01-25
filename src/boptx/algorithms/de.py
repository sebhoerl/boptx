import numpy as np

from ..algorithm import SampleProcessAlgorithm
from ..evaluator import Evaluator
from ..problem import Problem

import logging
logger = logging.getLogger(__name__)

class DifferentialEvolutionAlgorithm(SampleProcessAlgorithm):
    def __init__(self, problem: Problem, candidates = 4, crossover = 0.9, differential_weight = 0.8, seed = 0):
        if candidates < 4:
            raise RuntimeError("Number of candidates must be at least 4")

        # Problem
        self.problem = problem

        # Algorithm settings
        self.crossover = crossover
        self.differential_weight = differential_weight
        self.candidates = candidates
        self.seed = seed
        self.dimensions = self._get_dimensions(problem)

        # Algorithm state
        self.random = np.random.RandomState(self.seed)

        # Initial state
        parameters = self.problem.get_parameters()
        self.values = self.random.random_sample(size = (self.candidates, len(parameters)))

        for k, parameter in enumerate(parameters):
            bounds = parameter.get_bounds()

            self.values[:,k] *= (bounds[1] - bounds[0])
            self.values[:,k] += bounds[0]

        self.objectives = np.ones((candidates,))

    def set_state(self, state):
        assert self.dimension == len(state["values"].shape[1])

        self.random.set_state(state["random"])
        self.values = state["values"]
        self.objectives = state["objectives"]

    def get_state(self):
        return {
            "random": self.random.get_state(),
            "values": self.values,
            "objectives": self.objectives,
        }

    def get_settings(self):
        return {
            "crossover": self.crossover,
            "differential_weight": self.differential_weight,
            "candidates": self.candidates,
            "seed": self.seed
        }

    def sample(self):
        parameters = self.problem.get_parameters()
        values = np.copy(self.values)

        for k in range(self.candidates):
            choice_set = list(np.arange(self.candidates))
            choice_set.remove(k)

            a = choice_set[self.random.randint(len(choice_set))]
            choice_set.remove(a)

            b = choice_set[self.random.randint(len(choice_set))]
            choice_set.remove(b)

            c = choice_set[self.random.randint(len(choice_set))]

            selected_dimension = self.random.randint(self.dimensions)

            for i in range(self.dimensions):
                r = self.random.random_sample()

                if r < self.crossover or i == selected_dimension:
                    values[k,i] = self.values[a,i] + self.differential_weight * (self.values[b,i] - self.values[c,i])

        for k in range(self.dimensions):
            bounds = parameters[k].get_bounds()

            if not bounds is None:
                values[:,k] = np.maximum(values[:,k], bounds[0])
                values[:,k] = np.minimum(values[:,k], bounds[1])

        return values

    def process(self, evaluations):
        for k, evaluation in enumerate(evaluations):
            if evaluation.get_objective() < self.objectives[k]:
                self.objectives[k] = evaluation.get_objective()
                self.values[k] = evaluation.get_values()
