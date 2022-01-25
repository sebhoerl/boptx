class Problem:
    def __init__(self):
        raise NotImplementedError()

    def get_parameters(self):
        raise NotImplementedError()

    def get_settings(self):
        raise NotImplementedError()

class Parameter:
    def __init__(self, name = None):
        self.name = name

    def get_bounds(self):
        raise NotImplementedError()

    def get_initial_value(self):
        raise NotImplementedError()

    def get_name(self):
        return "Unnamed" if self.name is None else self.name

    def __str__(self):
        return "Parameter[{}]".format(self.get_name())

class ContinuousParameter(Parameter):
    def __init__(self, name = None, bounds = None, initial_value = None):
        super().__init__(name)

        self.bounds = bounds
        self.initial_value = initial_value

    def get_bounds(self):
        return self.bounds

    def get_initial_value(self):
        return self.initial_value

class ContinuousProblem:
    def __init__(self, parameters, bounds = None, initial_values = None):
        assert bounds is None or len(bounds) == parameters
        assert initial_values is None or len(initial_values) == parameters

        self.parameters = [
            ContinuousParameter(
                name = "x{}".format(k),
                bounds = bounds[k] if not bounds is None else None,
                initial_value = initial_values[k] if not initial_values is None else None
            ) for k in range(parameters)
        ]

    def get_parameters(self):
        return self.parameters
