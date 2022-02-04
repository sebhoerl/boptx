from boptx.evaluator import BaseEvaluator, DefaultEvaluation
from boptx.problem import ContinuousProblem
from boptx.algorithms.opdyts import OpdytsProblem, OpdytsEvaluation
from model import ModeShareModel

class ModeShareProblem(ContinuousProblem, OpdytsProblem):
    def get_state_count(self): # Implemented for opdyts
        return 3

class ModeShareEvaluation(DefaultEvaluation, OpdytsEvaluation):
    def __init__(self, values, objective, information, state, transitional):
        DefaultEvaluation.__init__(self, values, objective, information)
        self.state = state
        self.transitional = transitional

    def get_state(self):
        return self.state

    def is_transitional(self):
        return self.transitional

import numpy as np

class ModeShareEvaluator(BaseEvaluator):
    def __init__(self, transition_size):
        super().__init__()

        self.transition_size = transition_size

        self.results = {}

        # To save for later
        self.information = {}
        self.values = {}
        self.converged = {}

    def start_evaluation(self, identifier, values, information):
        """
        Here, the framework asks us to start a simulation run. In a heavy-weight
        simulator, we could start the simulation in a parallel process. Since
        our calculation is light-weight, we perform it directly and save the
        result based on the unique identifier.

        We use the values in the "values" vector to propagate the model. And we
        need to back up the value and information as we want to return them in
        a complete Evaluation object later on.
        """

        # Run the model
        model = ModeShareModel(
            constants = [values[0], values[1], -0.3],
            beta = values[2], capacity = 0.5, travel_time = 1.0,
            blending = 1e-2
        )

        initial_state = None

        if "opdyts" in information:
            opdyts = information["opdyts"]

            if "restart" in opdyts:
                # Get the last state of the simulation we should restart
                initial_state = self.results[opdyts["restart"]][-1]

        self.results[identifier] = model.run(
            iterations = self.transition_size,
            initial_state = initial_state)

        self.converged[identifier] = len(self.results[identifier]) < self.transition_size

        self.values[identifier] = values
        self.information[identifier] = information

    def check_evaluation(self, identifier):
        """
        Checks whether the simulation has finished. As we calculate the results
        directly, we can just return True.
        """
        return True

    def get_evaluation(self, identifier):
        """
        Here we need to return an Evaluation object with the objective value
        and potential supplementary information that can also be extended at
        this stage.
        """

        # Calculate the objective value

        reference = np.array([0.4 , 0.44, 0.16])
        state = self.results[identifier][-1]
        delta = reference - state
        objective = np.sum(np.abs(delta))

        return ModeShareEvaluation(
            self.values[identifier], objective, self.information[identifier],
            state, not self.converged[identifier]
        )

    def clean_evaluation(self, identifier):
        """
        Delete the information we have on an evaluation.
        """
        del self.results[identifier]
        del self.values[identifier]
        del self.information[identifier]
        del self.converged[identifier]


if __name__ == "__main__":
    wrapper = ModeShareEvaluator()

    wrapper.start_evaluation("uuid", [0.0, 0.0, 0.0], {})
    wrapper.check_evaluation("uuid")
    evaluation = wrapper.get_evaluation("uuid")
    wrapper.clean_evaluation("uuid")

    print(evaluation.get_objective())
