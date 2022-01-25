from boptx.evaluator import BaseEvaluator, DefaultEvaluation
from model import ModeShareModel

import numpy as np

class ModeShareEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()

        self.results = {}

        # To save for later
        self.information = {}
        self.values = {}

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

        self.results[identifier] = model.run()
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

        return DefaultEvaluation(
            self.values[identifier], objective, self.information[identifier]
        )

    def clean_evaluation(self, identifier):
        """
        Delete the information we have on an evaluation.
        """
        del self.results[identifier]
        del self.values[identifier]
        del self.information[identifier]


if __name__ == "__main__":
    wrapper = ModeShareEvaluator()

    wrapper.start_evaluation("uuid", [0.0, 0.0, 0.0], {})
    wrapper.check_evaluation("uuid")
    evaluation = wrapper.get_evaluation("uuid")
    wrapper.clean_evaluation("uuid")

    print(evaluation.get_objective())
