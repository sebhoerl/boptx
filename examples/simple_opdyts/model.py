import numpy as np

class ModeShareModel:
    def __init__(self, constants, beta, capacity, travel_time, blending):
        self.constants = constants
        self.beta = beta
        self.capacity = capacity
        self.travel_time = travel_time
        self.blending = blending
        assert len(self.constants) == 3

    def run(self, tolerance = 1e-6, iterations = None, initial_state = None):
        if initial_state is None:
            initial_state = np.array([1.0, 1.0, 1.0])
        else:
            initial_state = np.array(initial_state)

        # Make sure that the shares sum to one
        state = initial_state / np.sum(initial_state)

        # Only if zero iterations are requested, we abort right away
        converged = iterations == 0

        iteration = 0
        states = []

        while not converged:
            iteration += 1
            states.append(state)

            previous_state = np.copy(state)

            travel_time = self.travel_time * (state[0] / self.capacity)**4

            utilities = np.array([
                self.constants[0] + self.beta * travel_time,
                self.constants[1],
                self.constants[2]
            ])

            exponentials = np.exp(utilities)
            shares = exponentials / np.sum(exponentials)

            state = self.blending * shares + (1.0 - self.blending) * state

            if iterations is not None:
                converged = iteration >= iterations

            delta = np.abs(previous_state - state)
            converged |= np.all(delta < tolerance)

        return np.array(states)

if __name__ == "__main__":
    model = ModeShareModel(
        constants = [-0.1, -0.2, -0.3],
        beta = -0.1, capacity = 0.1, travel_time = 1.0,
        blending = 1e-2
    )

    print(model.run())
