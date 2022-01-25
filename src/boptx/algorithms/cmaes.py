import numpy as np
import scipy.linalg as la

from ..algorithm import SampleProcessAlgorithm
from ..evaluator import Evaluator
from ..problem import Problem

import logging
logger = logging.getLogger(__name__)

class CMAESAlgorithm(SampleProcessAlgorithm):
    def __init__(self, problem: Problem, sigma = 0.1, candidates = None, seed = 0, evaluate_mean = False):
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

        if evaluate_mean:
            self.candidates += 1

        # Algorithm state
        self.pc = np.zeros((self.dimensions, 1))
        self.ps = np.zeros((self.dimensions, 1))
        self.B = np.eye(self.dimensions)
        self.D = np.ones((self.dimensions,))
        self.C = np.dot(self.B, np.dot(np.diag(self.D**2), self.B.T))
        self.invsqrtC = np.dot(self.B, np.dot(np.diag(self.D**-1), self.B.T))

        self.eigeneval = 0
        self.counteval = 0

        self.mean = np.array(self._require_initial_values(self.problem.get_parameters())).reshape((self.dimensions, 1))
        self.sigma = sigma

        self.random = np.random.RandomState(self.seed)
        self.evaluate_mean = evaluate_mean

    def set_state(self, state):
        self.random.set_state(state["random"])

        self.mean = state["mean"]
        self.sigma = state["sigma"]
        self.C = state["C"]
        self.B = state["B"]
        self.D = state["D"]
        self.invsqrtC = state["invsqrtC"]
        self.eigeneval = state["eigeneval"]
        self.counteval = state["counteval"]

    def get_state(self):
        return {
            "random": self.random.get_state(),
            "mean": self.mean,
            "sigma": self.sigma,
            "C": self.C,
            "B": self.B,
            "D": self.D,
            "invsqrtC": self.invsqrtC,
            "eigeneval": self.eigeneval,
            "counteval": self.counteval
        }

    def get_settings(self):
        return {
            "seed": self.seed,
            "candidates": self.candidates,
            "dimensions": self.dimensions,
            "evaluate_mean": self.evaluate_mean
        }

    def sample(self):
        selected_values = np.zeros(shape = (0, self.dimensions))

        if self.evaluate_mean:
            selected_values = np.copy(self.mean).T

        while selected_values.shape[0] < self.candidates:
            z = self.random.normal(size = (self.dimensions, self.candidates))
            values = (self.sigma * np.dot(self.B, z * self.D[:, np.newaxis]) + self.mean).T

            # Apply bounds (resample if we exceed them)
            values = values[self._check_bounds(self.problem, values)]
            selected_values = np.append(selected_values, values, axis = 0)
            selected_values = selected_values[:self.candidates]

        return selected_values

    def process(self, evaluations):
        candidates = len(evaluations)
        self.counteval += candidates
        assert candidates == self.candidates

        # Obtain fitness
        candidate_parameters = np.array([e.get_values() for e in evaluations])
        candidate_objectives = np.array([e.get_objective() for e in evaluations])

        sorter = np.argsort(candidate_objectives)

        candidate_objectives = candidate_objectives[sorter]
        candidate_parameters = candidate_parameters[sorter, :]

        # Parameters
        mu = candidates / 2.0
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        mu = int(np.floor(mu))
        weights = weights[:mu] / np.sum(weights[:mu])
        mueff = np.sum(weights)**2 / np.sum(weights ** 2)

        N = self.dimensions
        cc = (4 + mueff / N) / (N + 4 + 2.0 * mueff / N)
        cs = (mueff + 2.0) / (N + mueff + 5.0)
        c1 = 2.0 / ((N + 1.3)**2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((N + 2.0)**2 + mueff))
        damps = 1.0 + 2.0 * max(0, np.sqrt((mueff - 1.0) / (N + 1.0)) - 1.0) + cs
        chiN = N**0.5 * (1.0 - 1.0 / (4.0 * N) + 1.0 / (21.0 * N**2))

        # Update mean
        previous_mean = np.copy(self.mean)
        self.mean = np.sum(candidate_parameters[:mu] * weights[:, np.newaxis], axis = 0).reshape((self.dimensions, 1))

        # Update evolution paths
        psa = (1.0 - cs) * self.ps
        psb = np.sqrt(cs * (2.0 - cs) * mueff) * np.dot(self.invsqrtC, self.mean - previous_mean) / self.sigma
        self.ps = psa + psb

        hsig = la.norm(self.ps) / np.sqrt(1.0 - (1.0 - cs)**(2.0 * self.counteval / candidates)) / chiN < 1.4 + 2.0 / (self.dimensions + 1.0)
        pca = (1.0 - cc) * self.pc
        pcb = hsig * np.sqrt(cc * (2.0 - cc) * mueff) * (self.mean - previous_mean) / self.sigma
        self.pc = pca + pcb

        # Adapt covariance matrix
        artmp = (1.0 / self.sigma) * (candidate_parameters[:mu].T - previous_mean)

        Ca = (1.0 - c1 - cmu) * self.C
        Cb = c1 * (np.dot(self.pc, self.pc.T) + (not hsig) * cc * (2.0 - cc) * self.C)
        Cc = cmu * np.dot(artmp, np.dot(np.diag(weights), artmp.T))
        C = Ca + Cb + Cc

        # Adapt step size
        self.sigma = self.sigma * np.exp((cs / damps) * (la.norm(self.ps) / chiN - 1.0))

        if self.counteval - self.eigeneval > candidates / (c1 + cmu) / self.dimensions / 10.0:
            self.eigeneval = self.counteval

            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            d, self.B = la.eigh(self.C)

            self.D = np.sqrt(d)
            Dm = np.diag(1.0 / np.sqrt(d))

            self.invsqrtC = np.dot(self.B.T, np.dot(Dm, self.B))

        if np.max(self.D) > 1e7 * np.min(self.D):
            logger.warning("Condition exceeds 1e14")
