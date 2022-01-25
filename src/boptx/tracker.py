import logging
logger = logging.getLogger(__name__)

import copy
import pickle

class Tracker:
    def __call__(self, state, finished):
        self.track(state, finished)

    def track(self, state, finished):
        raise NotImplementedError()

class PickleTracker(Tracker):
    def __init__(self, output_path):
        self.output_path = output_path
        self.round = 0

        self.states = []
        self.evaluations = []

    def track(self, state, finished):
        state = copy.deepcopy(state)
        state.update({ "round": self.round })
        self.states.append(state)

        for evaluation in finished:
            information = copy.deepcopy(evaluation.get_information())

            self.evaluations.append({
                "transitional": evaluation.is_transitional(),
                "objective": evaluation.get_objective(),
                "values": evaluation.get_values(),
                "information": information,
                "round": self.round
             })

        self.round += 1

        with open(self.output_path, "wb+") as f:
            pickle.dump(dict(
                states = self.states, evaluations = self.evaluations
            ), f)
