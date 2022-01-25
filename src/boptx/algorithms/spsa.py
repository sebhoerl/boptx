import numpy as np

from ..algorithm import Algorithm
from ..evaluator import Evaluator
from ..problem import Problem

import logging
logger = logging.getLogger(__name__)

# https://www.jhuapl.edu/spsa/PDF-SPSA/Spall_Implementation_of_the_Simultaneous.PDF

class SPSAAlgorithm(Algorithm):
    def __init__(self, problem: Problem, perturbation_factor, gradient_factor, perturbation_exponent = 0.101, gradient_exponent = 0.602, gradient_offset = 0, compute_objective = True, seed = 0):
        # Problem
        self.problem = problem
        self._require_initial_values(self.problem.get_parameters())

        # Algorithm settings
        self.compute_objective = compute_objective

        self.perturbation_factor = perturbation_factor
        self.perturbation_exponent = perturbation_exponent

        self.gradient_factor = gradient_factor
        self.gradient_exponent = gradient_exponent
        self.gradient_offset = gradient_offset

        self.seed = seed
        self.dimensions = self._get_dimensions(problem)

        # Algorithm state
        self.random = np.random.RandomState(self.seed)
        self.iteration = 0
        self.values = self._require_initial_values(self.problem.get_parameters())

        # Ephemeral state
        self.gradient_length = np.nan
        self.perturbation_length = np.nan

        self._warn_ignore_bounds(self.problem.get_parameters(), logger)

    def set_state(self, state):
        assert self.dimensions == len(state["values"])

        self.iteration = state["iteration"]
        self.random.set_state(state["random"])
        self.values = state["values"]

    def get_state(self):
        return {
            "iteration": self.iteration,
            "random": self.random.get_state(),
            "values": self.values,

            # Just for information
            "gradient_length": self.gradient_length,
            "perturbation_length": self.perturbation_length
        }

    def get_settings(self):
        return {
            "compute_objective": self.compute_objective,
            "perturbation_factor": self.perturbation_factor,
            "perturbation_exponent": self.perturbation_exponent,
            "gradient_factor": self.gradient_factor,
            "gradient_exponent": self.gradient_exponent,
            "gradient_offset": self.gradient_offset,
            "seed": self.seed
        }

    def advance(self, evaluator: Evaluator):
        self.iteration += 1

        # Update lengths
        self.gradient_length = self.gradient_factor / (self.iteration + self.gradient_offset)**self.gradient_exponent
        self.perturbation_length = self.perturbation_factor / self.iteration**self.perturbation_exponent

        logger.debug("SPSA Iteration {} (Gradient {}, Perturbation {})".format(
            self.iteration, self.gradient_length, self.perturbation_length
        ))

        # Calculate objective
        if self.compute_objective:
            objective_identifier = evaluator.submit_one(self.values, { "type": "objective" })

        # Sample direction from Rademacher distribution
        direction = self.random.randint(0, 2, self.dimensions) - 0.5

        # Schedule samples
        positive_values = np.copy(self.values)
        positive_values += direction * self.perturbation_length
        positive_identifier = evaluator.submit_one(positive_values, { "type": "positive_gradient" })

        negative_values = np.copy(self.values)
        negative_values -= direction * self.perturbation_length
        negative_identifier = evaluator.submit_one(negative_values, { "type": "negative_gradient" })

        if self.compute_objective:
            evaluator.clean_one(objective_identifier)

        positive_objective = evaluator.get_one(positive_identifier).get_objective()
        evaluator.clean_one(positive_identifier)

        negative_objective = evaluator.get_one(negative_identifier).get_objective()
        evaluator.clean_one(negative_identifier)

        gradient = (positive_objective - negative_objective) / (2.0 * self.perturbation_length)
        gradient *= direction**-1

        # Update state
        self.values -= self.gradient_length * gradient
