import numpy as np

from ..algorithm import Algorithm
from ..evaluator import Evaluator
from ..problem import Problem

import logging
logger = logging.getLogger(__name__)

class NelderMeadAlgorithm(Algorithm):
    def __init__(self, problem: Problem, reflection = 1.0, expansion = 2.0, contraction = 0.5, shrink = 0.5, seed = 0):
        # Problem
        self.problem = problem
        self._require_initial_values(self.problem.get_parameters())

        # Algorithm settings
        self.reflection = reflection
        self.expansion = expansion
        self.contraction = contraction
        self.shrink = shrink
        self.dimensions = self._get_dimensions(problem)
        self.seed = seed

        random = np.random.RandomState(self.seed)

        # Algorithm state
        self.x = random.random_sample(size = (self.dimensions + 1, self.dimensions))

        for k, parameter in enumerate(self.problem.get_parameters()):
            bounds = parameter.get_bounds()

            self.x[:,k] *= (bounds[1] - bounds[0])
            self.x[:,k] += bounds[0]

        self.objectives = None

    def set_state(self, state):
        assert self.dimensions == len(state["x"].shape[1])
        self.x = state["x"]

    def get_state(self):
        return {
            "x": self.x,
        }

    def get_settings(self):
        return {
            "seed": self.seed,
            "reflection": self.reflection,
            "expansion": self.expansion,
            "contraction": self.contraction,
            "shrink": self.shrink
        }

    def _evaluate(self, evaluator, x, purpose):
        identifiers = evaluator.submit([x], { "type": purpose })
        evaluation = evaluator.get(identifiers)[0]
        evaluator.clean(identifiers)
        return evaluation.get_objective()

    def advance(self, evaluator: Evaluator):
        logger.debug("Next Nelder-Mead iteration")

        if self.objectives is None:
            identifiers = evaluator.submit(self.x, { "type": "initial" })

            self.objectives = np.array([
                e.get_objective() for e in evaluator.get(identifiers)
            ])

            evaluator.clean(identifiers)

        # Sort all points
        sorter = np.argsort(self.objectives)
        self.objectives = self.objectives[sorter]
        self.x = self.x[sorter]

        # Centroid
        x_0 = np.mean(self.x[:-1], axis = 1)
        objective_0 = self._evaluate(evaluator, x_0, "centroid")

        logger.debug("Centroid at {} with objective {}".format(x_0, objective_0))

        # Reflection
        x_r = x_0 + self.reflection * (x_0 - self.x[-1])
        objective_r = self._evaluate(evaluator, x_r, "reflection")

        logger.debug("Reflection point at {} with objective {}".format(x_r, objective_r))

        if self.objectives[0] <= objective_r and objective_r < self.objectives[-2]:
            self.x[-1] = x_r
            self.objectives[-1] = objective_r

            logger.debug("  Replaced worst with reflection point")
            return

        # Expansion
        if objective_r < self.objectives[0]:
            x_e = x_0 + self.expansion * (x_r - x_0)
            objective_e = self._evaluate(evaluator, x_e, "expansion")

            logger.debug("Expansion point at {} with objective {}".format(x_e, objective_e))

            if objective_e < objective_r:
                self.x[-1] = x_e
                self.objectives[-1] = objective_e

                logger.debug("  Replaced worst with expansion point")
                return
            else:
                self.x[-1] = x_r
                self.objectives[-1] = objective_r

                logger.debug("  Replaced worst with reflection point")
                return

        # Contraction
        x_c = x_0 + self.contraction * (self.x[-1] - x_0)
        objective_c = self._evaluate(evaluator, x_c, "contraction")

        logger.debug("Contraction point at {} with objective {}".format(x_c, objective_c))

        if objective_c < self.objectives[-1]:
            self.x[-1] = x_c
            self.objectives[-1] = objective_c

            logger.debug("  Replaced worst with contraction point")
            return

        # Shrink
        logger.debug("Shrinking simplex")

        for i in range(1, self.dimensions + 1):
            self.x[i] = self.shrink * (self.x[i] - self.x[0])

        identifiers = evaluator.submit(self.x[1:], { "type": "shrink" })
        evaluations = evaluator.get(identifiers)
        evaluator.clean(identifiers)

        self.objectives[1:] = [
            e.get_objective() for e in evaluations
        ]
