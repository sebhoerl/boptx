import numpy as np
import scipy.linalg as la
import scipy.optimize as opt

from ..algorithm import SampleProcessAlgorithm, Algorithm
from ..evaluator import Evaluator, Evaluation
from ..problem import Problem

import logging
logger = logging.getLogger(__name__)

import types

class OpdytsProblem(Problem):
    def get_state_count(self):
        raise NotImplementedError()

class OpdytsEvaluation(Evaluation):
    INITIAL = "initial"
    CANDIDATE = "candidate"
    TRANSITION = "transition"

    def get_state(self):
        raise NotImplementedError()

class OpdytsAlgorithm(Algorithm):
    def __init__(self, problem: OpdytsProblem, delegate: SampleProcessAlgorithm, adaptation_weight = 0.3, seed = 0):
        # Problem
        self.problem = problem
        self._require_initial_values(self.problem.get_parameters())

        self.delegate = delegate

        # Settings
        self.seed = seed
        self.states = problem.get_state_count()
        self.adaptation_weight = adaptation_weight

        # State variables
        self.random = np.random.RandomState(self.seed)
        self.base_identifier = None
        self.v, self.w = 0.0, 0.0
        self.iteration = 1

        self.global_selection_performance = []
        self.global_transient_performance = []
        self.global_equilibrium_gap = []
        self.global_uniformity_gap = []

    def set_state(self, state):
        self.delegate.set_state(state["delegate"])
        self.random.set_state(state["random"])

        self.iteration = state["iteration"]
        self.base_identifier = state["base_identifier"]

        self.global_selection_performance = state["adaptation_selection_performance"]
        self.global_transient_performance = state["global_transient_performance"]
        self.global_equilibrium_gap = state["global_equilibrium_gap"]
        self.global_uniformity_gap = state["global_uniformity_gap"]

    def get_state(self):
        return {
            "random": self.random.get_state(),
            "iteration": self.iteration,
            "base_identifier": self.base_identifier,
            "delegate": self.delegate.get_state(),
            "adaptation_selection_performance": self.global_selection_performance,
            "adaptation_transient_performance": self.global_transient_performance,
            "adaptation_equilibrium_gap": self.global_equilibrium_gap,
            "adaptation_uniformity_gap": self.global_uniformity_gap,
        }

    def get_settings(self):
        return {
            "seed": self.seed,
            "delegate": self.delegate.get_settings(),
            "adaptation_weight": self.adaptation_weight
        }

    def _combine(self, first, second):
        information = {}
        information.update(first)
        information.update(second)
        return information

    def advance(self, evaluator: Evaluator):
        if self.base_identifier is None:
            logger.debug("Initilizing Opdyts with one simulation run")
            initial_values = self._require_initial_values(self.problem.get_parameters())
            self.base_identifier = evaluator.submit_one(initial_values, dict(opdyts = dict(
                type = OpdytsEvaluation.INITIAL
            )))

        base_evaluation = evaluator.get_one(self.base_identifier)

        # Sample new candidate values (see SampleProcessAlgorithm.advance)
        values = []
        information = []

        for sample in self.delegate.sample():
            if isinstance(sample, tuple):
                values.append(sample[0])
                information.append(sample[1])
            else:
                values.append(sample)
                information.append({})

        candidates = len(values)

        logger.debug("Running {} candidates from delegate sampler".format(candidates))

        # Initialize iteration state and perform first transition
        states = np.zeros((candidates, self.states))
        deltas = np.zeros((candidates, self.states))
        objectives = np.zeros((candidates,))
        transitions = np.ones((candidates,))

        identifiers = [
            evaluator.submit_one(values[k], self._combine(dict(opdyts = dict(
                type = OpdytsEvaluation.CANDIDATE,
                candidate = k, iteration = self.iteration,
                restart = self.base_identifier,
                restart_convergence = False
            )), information[k])) for k in range(candidates)
        ]

        for k in range(candidates):
            evaluation = evaluator.get_one(identifiers[k])
            assert isinstance(evaluation, OpdytsEvaluation)

            objectives[k] = evaluation.get_objective()
            states[k] = evaluation.get_state()
            deltas[k] = states[k] - base_evaluation.get_state()

        # Start to advance single candidates
        local_transient_performance = []
        local_equilibrium_gap = []
        local_uniformity_gap = []

        converged_k = None

        while converged_k is None:
            # Approximate selection problem
            selection_problem = ApproximateSelectionProblem(
                self.v, self.w, deltas, objectives)
            alpha = selection_problem.solve()

            transient_performance = selection_problem.get_transient_performance(alpha)
            equilibrium_gap = selection_problem.get_equilibrium_gap(alpha)
            uniformity_gap = selection_problem.get_uniformity_gap(alpha)

            logger.debug(
                "Transient performance: {}, Equilibirum gap: {}, Uniformity_gap: {}".format(
                transient_performance, equilibrium_gap, uniformity_gap))

            # Save the local trace in this iteration
            local_transient_performance.append(transient_performance)
            local_equilibrium_gap.append(equilibrium_gap)
            local_uniformity_gap.append(uniformity_gap)

            # Select one candidate to advance
            cumulative_alpha = np.cumsum(alpha)
            k = np.sum(self.random.random_sample() > cumulative_alpha)

            transitions[k] += 1
            logger.debug("Transitioning candidate {} (transition {})".format(k, transitions[k]))

            identifier = evaluator.submit_one(values[k], self._combine(dict(opdyts = dict(
                type = OpdytsEvaluation.TRANSITION,
                candidate = k, iteration = self.iteration,
                transient_performance = transient_performance,
                equilibrium_gap = equilibrium_gap,
                uniformity_gap = uniformity_gap,
                restart = identifiers[k],
                restart_convergence = True,
                transition = transitions[k]
            )), information[k]))

            # Get information and clean up
            evaluation = evaluator.get_one(identifier)
            evaluator.clean_one(identifiers[k])

            objectives[k] = evaluation.get_objective()
            deltas[k] = evaluation.get_state() - states[k]
            states[k] = evaluation.get_state()
            identifiers[k] = identifier

            if not evaluation.is_transitional():
                converged_k = k

        # One candidate has converged
        logger.debug("Solved selection problem with candidate {}".format(converged_k))

        # Clean up lose ends
        for k in range(candidates):
            if k != converged_k:
                evaluator.clean_one(identifiers[k])

        # Reset base simulation
        evaluator.clean_one(self.base_identifier)
        self.base_identifier = identifiers[converged_k]

        # Update states
        self.global_selection_performance.append(objectives[converged_k])
        self.global_transient_performance.append(np.array(local_transient_performance))
        self.global_equilibrium_gap.append(np.array(local_equilibrium_gap))
        self.global_uniformity_gap.append(np.array(local_uniformity_gap))

        adaptation_problem = AdaptationProblem(
            self.adaptation_weight,
            self.global_selection_performance,
            self.global_transient_performance,
            self.global_equilibrium_gap,
            self.global_uniformity_gap)

        self.v, self.w = adaptation_problem.solve()
        logger.debug("Solved Adaptation Problem. v = {}, w = {}".format(self.v, self.w))

        self.iteration += 1

        # Pass candidate to delegate algorithm
        self.delegate.process([evaluator.get_one(identifiers[converged_k])])

class ApproximateSelectionProblem:
    def __init__(self, v, w, deltas, objectives):
        self.deltas = deltas
        self.objectives = objectives
        self.w = w
        self.v = v

    def get_uniformity_gap(self, alpha):
        return np.sum(alpha**2)

    def get_equilibrium_gap(self, alpha):
        return np.sqrt(np.sum((alpha[:, np.newaxis] * self.deltas)**2))

    def get_transient_performance(self, alpha):
        return np.sum(alpha * self.objectives)

    def get_objective(self, alpha):
        objective = self.get_transient_performance(alpha)
        objective += self.v * self.get_equilibrium_gap(alpha)
        objective += self.w * self.get_uniformity_gap(alpha)
        return objective

    def solve(self):
        initial = np.ones((len(self.objectives),)) / len(self.objectives)
        result = opt.minimize(self.get_objective, initial, constraints = [
            { "type": "eq", "fun": lambda alpha: np.sum(alpha) - 1.0 },
        ], bounds = [(0.0, 1.0)] * len(self.objectives), options = { "disp": False })

        if not result.success:
            logger.warn("Could not solve Approximate Selection Problem")
            logger.debug("Deltas: {}".format(self.deltas))
            logger.debug("Objectives {}:".format(self.objectives))
            logger.debug("v, w: {}, {}".format(self.v, self.w))
            return initial

        return result.x

class AdaptationProblem:
    def __init__(self, weight, selection_performance, transient_performance, equilibrium_gap, uniformity_gap):
        self.weight = weight
        self.selection_performance = selection_performance
        self.transient_performance = transient_performance
        self.uniformity_gap = uniformity_gap
        self.equilibrium_gap = equilibrium_gap

    def get_objective(self, vw):
        R = len(self.selection_performance)
        v, w = vw

        objective = 0.0

        for r in range(R):
            local_objective = np.abs(self.transient_performance[r] - self.selection_performance[r])
            local_objective -= (v * self.equilibrium_gap[r] + w * self.uniformity_gap[r])
            local_objective = np.sum(local_objective**2)
            objective += self.weight**(R - r) * local_objective

        return objective

    def solve(self):
        initial = np.array([0.0, 0.0])

        result = opt.minimize(self.get_objective, initial, bounds = [
            (0.0, 1.0), (0.0, 1.0)
        ], options = { "disp": False })

        if not result.success:
            logger.warn("Could not solve Adaptation Problem")
            return initial

        return result.x
