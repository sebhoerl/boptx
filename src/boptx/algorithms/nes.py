import numpy as np
import scipy.linalg as la

from ..algorithm import SampleProcessAlgorithm
from ..evaluator import Evaluator
from ..problem import Problem

import logging
logger = logging.getLogger(__name__)

class XNESAlgorithm(SampleProcessAlgorithm):
    def __init__(self, problem: Problem, sigma = 0.1, candidates = None, seed = 0):
        # Problem
        self.problem = problem
        self._require_initial_values(self.problem.get_parameters())

        dimensions = self._get_dimensions(problem)
        default_candidates = int(4 + np.floor(3 * np.log(dimensions)))

        if not candidates is None:
            if candidates < default_candidates:
                logger.warn("Using {} candidates although xNES proposes at least {}".format(candidates, default_candidates))
        else:
            candidates = default_candidates

        # Algorithm settings
        self.seed = seed
        self.candidates = candidates
        self.dimensions = dimensions
        self.eta_sigma = 3.0 * (3.0 + np.log(self.dimensions)) / (5.0 * self.dimensions * np.sqrt(self.dimensions))
        self.eta_B = self.eta_sigma

        # Algorithm state
        self.mean = np.array(self._require_initial_values(self.problem.get_parameters())).reshape((self.dimensions, 1))
        self.sigma = sigma
        self.B = np.eye(len(self.mean)) * self.sigma

        self.random = np.random.RandomState(self.seed)

    def set_state(self, state):
        assert state["mean"].shape == (self.dimensions,)
        assert state["B"].shape == (self.dimensions, self.dimensions)

        self.random.set_state(state["random"])
        self.mean = state["mean"]
        self.sigma = state["sigma"]
        self.B = state["B"]

    def get_state(self):
        return {
            "random": self.random.get_state(),
            "mean": self.mean,
            "sigma": self.sigma,
            "B": self.B,
        }

    def get_settings(self):
        return {
            "seed": self.seed,
            "candidates": self.candidates,
        }

    def sample(self):
        s = self.random.normal(size = (self.candidates, self.dimensions)).T
        z = self.sigma * np.dot(self.B, s) + self.mean
        #z = self._truncate_bounds(self.problem, z.T).T

        # Back transformation in case bounds have been truncated
        #s = np.dot(np.linalg.pinv(self.B), (z - self.mean) / self.sigma)

        for k in range(self.candidates):
            yield z[:,k], { "s": s[:,k] }

    def process(self, evaluations):
        objectives = np.array([-e.get_objective() for e in evaluations])

        sorter = np.argsort(objectives)
        available = len(sorter)

        rank = np.zeros((available,), dtype = int)
        rank[sorter] = np.arange(available)

        utilities = np.maximum(0.0, np.log(0.5 * available + 1) - np.log(available - rank))
        utilities /= np.sum(utilities)
        utilities -= 1.0 / available

        s = np.array([e.get_information()["s"] for e in evaluations]).T

        Gd = sum([
            utilities[k] * s[:,k]
            for k in range(available)
        ]).reshape((self.dimensions, 1))

        GM = sum([
            utilities[k] * (np.outer(s[:,k], s[:,k]) - np.eye(self.dimensions))
            for k in range(available)
        ])

        Gs = np.trace(GM) / self.dimensions

        GB = GM - Gs * np.eye(self.dimensions)

        self.mean += self.sigma * np.dot(self.B, Gd)
        self.sigma *= np.exp(0.5 * self.eta_sigma * Gs)

        self.B = np.dot(self.B, la.expm(0.5 * self.eta_B * GB))

class ElitistXNESAlgorithm(SampleProcessAlgorithm):
    def __init__(self, problem: Problem, sigma = 0.1, candidates = 1, seed = 0):
        # Problem
        self.problem = problem
        self._require_initial_values(self.problem.get_parameters())

        # Algorithm settings
        self.seed = seed
        self.candidates = candidates
        self.dimensions = self._get_dimensions(problem)
        self.eta_A = 0.25 * self.dimensions**-1.5
        self.eta_sigman = (1.0 / 5.0) * self.dimensions**-1.5
        self.eta_sigmap = self.dimensions**-1.5

        # Algorithm state
        self.solution = np.array(self._require_initial_values(self.problem.get_parameters())).reshape((self.dimensions, 1))
        self.solution_objective = np.inf
        self.sigma = sigma
        self.A = np.eye(self.dimensions) * sigma

        self.random = np.random.RandomState(self.seed)

    def set_state(self, state):
        self.random.set_state(state["random"])
        self.solution = state["solution"]
        self.solution_objective = state["solution_objective"]
        self.sigma = state["sigma"]
        self.A = state["A"]

    def get_state(self):
        return {
            "random": self.random.get_state(),
            "solution": self.solution,
            "solution_objective": self.solution_objective,
            "sigma": self.sigma,
            "A": self.A,
        }

    def get_settings(self):
        return {
            "seed": self.seed,
            "candidates": self.candidates,
        }

    def sample(self):
        candidates_z = self.random.normal(size = (self.dimensions, self.candidates))
        candidates_x = self.solution + self.sigma * np.dot(self.A, candidates_z)

        for xi, zi in zip(candidates_x.T, candidates_z.T):
            yield xi, { "z": zi }

    def process(self, evaluations):
        objectives = np.array([e.get_objective() for e in evaluations])
        index = np.argmin(objectives)
        evaluation = evaluations[index]

        x = evaluation.get_values().reshape((self.dimensions, 1))
        z = evaluation.get_information()["z"].reshape((self.dimensions, 1))
        objective = objectives[index]

        if objective < self.solution_objective:
            self.solution_objective = objective
            self.solution = x

            GA = np.dot(z, z.T) - np.eye(self.dimensions)
            self.A = np.dot(self.A, la.expm(self.eta_A * GA))
            self.sigma *= np.exp(self.eta_sigmap)

        else:
            self.sigma /= np.exp(self.eta_sigman)
