from .evaluator import Evaluator
from .algorithm import Algorithm

import numpy as np

import logging
logger = logging.getLogger(__name__)

class Loop:
    def __init__(self, evaluator: Evaluator, algorithm: Algorithm, absolute_tolerance = None, relative_tolerance = None, maximum_iterations = None, maximum_evaluations = None):
        self.evaluator = LoopEvaluator(evaluator)
        self.algorithm = algorithm

        # Settings
        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance
        self.maximum_iterations = maximum_iterations
        self.maximum_evaluations = maximum_evaluations

        # Loop state
        self.iteration = 0

    def get_state(self):
        return {
            "iteration": self.iteration,
            "evaluator": self.evaluator.get_state(),
            "algorithm": self.algorithm.get_state(),
        }

    def set_state(self, state):
        self.iteration = state["iteration"]
        self.evaluator.set_state(state["evaluator"])
        self.algorithm.set_state(state["algorithm"])

    def get_settings(self):
        return {
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "maximum_iterations": self.maximum_iterations,
            "maximum_evaluations": self.maximum_evaluations,
            "algorithm": self.algorithm.get_settings(),
            "evaluator": self.evaluator.get_settings(),
        }

    def advance(self, maximum_iterations = None, callback = None):
        do_continue = True

        while do_continue:
            do_continue = True

            if self.absolute_tolerance is not None or self.relative_tolerance is not None:
                do_continue = False
                do_continue |= self.absolute_tolerance is not None and self.evaluator.absolute_improvement > self.absolute_tolerance
                do_continue |= self.relative_tolerance is not None and self.evaluator.relative_improvement > self.relative_tolerance

            if not self.maximum_iterations is None and self.iteration >= self.maximum_iterations:
                do_continue = False

                logger.info("Maximum number of iterations has been reached.")

            if not self.maximum_evaluations is None and self.evaluator.evaluations >= self.maximum_evaluations:
                do_continue = False

                logger.info("Maximum number of evaluations has been reached.")

            if not maximum_iterations is None and self.iteration >= maximum_iterations:
                do_continue = False

            if do_continue:
                self.algorithm.advance(self.evaluator)
                self.iteration += 1

                if not callback is None:
                    callback(self.get_state(), self.evaluator.finished)
                    self.evaluator.finished = []

class LoopEvaluator(Evaluator):
    def __init__(self, delegate: Evaluator):
        self.delegate = delegate
        self.waiting = []
        self.finished = []

        # State
        self.best_values = None
        self.best_objective = None

        self.absolute_improvement = np.inf
        self.relative_improvement = np.inf

        self.evaluations = 0

    def set_state(self, state):
        self.best_values = state["values"]
        self.best_objective = state["objective"]
        self.absolute_improvement = state["absolute_improvement"]
        self.relative_improvement = state["relative_improvement"]
        self.evaluations = state["evaluations"]

    def get_state(self):
        return {
            "values": self.best_values,
            "objective": self.best_objective,
            "absolute_improvement": self.absolute_improvement,
            "relative_improvement": self.relative_improvement,
            "evaluations": self.evaluations,
        }

    def submit(self, values, information = None):
        identifiers = self.delegate.submit(values, information)
        self.waiting += identifiers
        return identifiers

    def _process(self, identifiers):
        for identifier in identifiers:
            if identifier in self.waiting:
                self.waiting.remove(identifier)

                evaluation = self.delegate.get([identifier])[0]
                self.evaluations += 1
                self.finished.append(evaluation)

                if not evaluation.is_transitional():
                    if self.best_objective is None or evaluation.get_objective() < self.best_objective:
                        if not self.best_objective is None:
                            self.absolute_improvement = np.abs(self.best_objective - evaluation.get_objective())
                            self.relative_improvement = self.absolute_improvement / np.abs(self.best_objective)

                        self.best_objective = evaluation.get_objective()
                        self.best_values = evaluation.get_values()

                        logger.info("New best objective {} at {} in evaluation #{}".format(
                            self.best_objective, self.best_values, self.evaluations
                        ))

    def clean(self, identifiers):
        self._process(identifiers)
        return self.delegate.clean(identifiers)

    def get(self, identifiers):
        self._process(identifiers)
        return self.delegate.get(identifiers)

    def get_settings(self):
        return self.delegate.get_settings()
