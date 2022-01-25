import numpy as np
import scipy.linalg as la

from ..algorithm import SampleProcessAlgorithm
from ..evaluator import Evaluator
from ..problem import Problem

import logging
logger = logging.getLogger(__name__)

class CMAES1P1Algorithm(SampleProcessAlgorithm):
    def __init__(self, problem: Problem, sigma = 0.1, candidates = 1, seed = 0):
        # Problem
        self.problem = problem
        self._require_initial_values(self.problem.get_parameters())

        # Algorithm settings
        self.seed = seed
        self.dimensions = self._get_dimensions(problem)
        self.candidates = candidates

        self.d = 1.0 + self.dimensions / 2.0
        self.p_target = 2.0 / 11.0
        self.cp = 1.0 / 12.0
        self.cc = 2.0 / (self.dimensions + 2.0)
        self.ccov = 2.0 / (self.dimensions**2 + 6)
        self.p_threshold = 0.44

        # Algorithm state
        self.solution = np.array(self._require_initial_values(self.problem.get_parameters())).reshape((self.dimensions, 1))
        self.solution_objective = np.inf

        self.sigma = sigma
        self.A = np.eye(self.dimensions)
        self.Ainv = np.eye(self.dimensions)
        self.p_success = self.p_target
        self.pc = np.zeros((self.dimensions, 1))

        self.random = np.random.RandomState(self.seed)

    def set_state(self, state):
        self.random.set_state(state["random"])

        self.solution = state["solution"]
        self.solution_objective = state["solution_objective"]

        self.sigma = state["sigma"]
        self.A = state["A"]
        self.Ainv = state["Ainv"]
        self.p_success = state["p_success"]
        self.pc = state["pc"]

    def get_state(self):
        return {
            "random": self.random.get_state(),
            "solution": self.solution,
            "solution_objective": self.solution_objective,
            "sigma": self.sigma,
            "A": self.A,
            "Ainv": self.Ainv,
            "p_success": self.p_success,
            "pc": self.pc
        }

    def get_settings(self):
        return {
            "seed": self.seed,
            "candidates": self.candidates,
            "dimensions": self.dimensions,
            "d": self.d,
            "cp": self.cp,
            "cc": self.cc,
            "p_target": self.p_target,
            "ccovp": self.ccov,
            "p_threshold": self.p_threshold
        }

    def sample(self):
        candidates_z = np.zeros(shape = (self.dimensions, 0))

        while candidates_z.shape[1] < self.candidates:
            z = self.random.normal(size = (self.dimensions, self.candidates))
            x = self.solution + self.sigma * np.dot(self.A, z)

            # Apply bounds (resample if we exceed them)
            z = z[:,self._check_bounds(self.problem, x.T)]
            candidates_z = np.append(candidates_z, z, axis = 1)
            candidates_z = candidates_z[:,:self.candidates]

        candidates_y = np.dot(self.A, candidates_z)
        candidates_x = self.solution + self.sigma * candidates_y

        for xi, zi in zip(candidates_x.T, candidates_z.T):
            yield xi, { "z": zi }

    def process(self, evaluations):
        if len(evaluations) > 1:
            objectives = np.array([e.get_objective() for e in evaluations])
            minimum_index = np.argmin(objectives)
            evaluations = [evaluations[minimum_index]]

        candidate_objective = evaluations[0].get_objective()
        candidate_x = evaluations[0].get_values().reshape((self.dimensions, 1))
        candidate_z = evaluations[0].get_information()["z"].reshape((self.dimensions, 1))

        # Update step size
        self.p_success *= (1 - self.cp)
        self.p_success += self.cp if candidate_objective < self.solution_objective else 0.0

        self.sigma *= np.exp(self.d**-1 * (self.p_success - self.p_target) / (1 - self.p_target))

        if candidate_objective < self.solution_objective:
            self.solution = candidate_x
            self.solution_objective = candidate_objective

            # Update Cholesky
            alpha = None

            if self.p_success < self.p_threshold:
                self.pc *= 1 - self.cc
                self.pc += np.sqrt(self.cc * (2 - self.cc)) * candidate_z

                alpha = 1 - self.ccov
            else:
                self.pc *= 1 - self.cc
                alpha = (1 - self.ccov) + self.ccov * self.cc * (2 - self.cc)

            beta = self.ccov

            w = np.dot(self.Ainv, self.pc)
            norm2 = la.norm(w)**2

            self.A = np.sqrt(alpha) * self.A + np.sqrt(alpha) / norm2 * (np.sqrt(1 + beta / alpha * norm2) - 1) * np.dot(self.pc, w.T)
            self.Ainv = np.sqrt(alpha)**-1 * self.Ainv - np.sqrt(alpha)**-1 * norm2**-1 * (1 - (1 + beta / alpha * norm2)**-1) * np.dot(w, np.dot(w.T, self.Ainv))
