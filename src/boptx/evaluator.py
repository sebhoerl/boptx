import hashlib
import time

class Evaluator:
    def __init__(self):
        raise NotImplementedError()

    def submit(self, values, information):
        raise NotImplementedError()

    def get(self, identifiers):
        raise NotImplementedError()

    def clean(self, identifiers):
        raise NotImplementedError()

    def submit_one(self, values, information):
        return self.submit([values], information)[0]

    def clean_one(self, identifier):
        return self.clean([identifier])

    def get_one(self, identifier):
        return self.get([identifier])[0]

    def get_settings(self):
        raise NotImplementedError()

class Evaluation:
    def get_values(self):
        raise NotImplementedError()

    def get_objective(self):
        raise NotImplementedError()

    def is_transitional(self):
        raise NotImplementedError()

    def get_information(self):
        raise NotImplementedError()

class DefaultEvaluation(Evaluation):
    def __init__(self, values, objective, information):
        self.values = values
        self.objective = objective
        self.information = information

    def get_values(self):
        return self.values

    def get_objective(self):
        return self.objective

    def get_information(self):
        return self.information

    def is_transitional(self):
        return False

def create_identifier(*elements):
    hash = hashlib.md5()

    for element in elements:
        hash.update(str(element).encode("utf-8"))

    return str(hash.hexdigest())

class DefaultEvaluator(Evaluator):
    def __init__(self):
        self.evaluations = {}

    def submit(self, values, information):
        identifiers = []

        for v in values:
            identifier = create_identifier(v)

            while identifier in self.evaluations.keys():
                identifier += "/2"

            identifiers.append(identifier)
            self.evaluations[identifier] = self.evaluate(identifier, v, information)

        return identifiers

    def evaluate(self, identifier, values, information):
        raise NotImplementedError()

    def get(self, identifiers):
        return [
            self.evaluations[identifier]
            for identifier in identifiers]

    def clean(self, identifiers):
        for identifier in identifiers:
            del self.evaluations[identifier]

    def get_settings(self):
        return {}

class FunctionEvaluator(DefaultEvaluator):
    def __init__(self, callable):
        super().__init__()
        self.callable = callable

    def evaluate(self, identifier, values, information):
        return DefaultEvaluation(
            values, self.callable(values, information),
            information
        )

class BaseEvaluator(Evaluator):
    def __init__(self, interval = 1e-2, parallelism = 1):
        self.queue = []
        self.active = set()
        self.evaluations = {}

        self.parallelism = parallelism
        self.check_interval = interval

    def start_evaluation(self, identifier, values, information):
        raise NotImplementedError()

    def check_evaluation(self, identifier):
        raise NotImplementedError()

    def get_evaluation(self, identifier, information):
        raise NotImplementedError()

    def clean_evaluation(self, identifier):
        raise NotImplementedError()

    def submit(self, values, information = None):
        identifiers = []

        for v in values:
            identifier = create_identifier(v, information)

            while identifier in self.active:
                identifier += "/2"

            identifiers.append(identifier)

            self.evaluations[identifier] = {
                "information": information,
                "values": v,
            }

            self.queue.append(identifier)

        self.ping()
        return identifiers

    def ping(self):
        for identifier in set(self.active):
            if self.check_evaluation(identifier):
                self.active.remove(identifier)

        while len(self.queue) > 0 and len(self.active) < self.parallelism:
            identifier = self.queue.pop(0)
            self.active.add(identifier)

            self.start_evaluation(identifier, self.evaluations[identifier]["values"], self.evaluations[identifier]["information"])

    def get(self, identifiers):
        remaining_indices = list(range(len(identifiers)))
        response = [None] * len(identifiers)

        while len(remaining_indices) > 0:
            self.ping()

            for index in remaining_indices:
                identifier = identifiers[index]

                if self.check_evaluation(identifier):
                    response[index] = self.get_evaluation(identifier)
                    remaining_indices.remove(index)

            if len(remaining_indices) > 0:
                time.sleep(self.check_interval)

        return response

    def clean(self, identifiers):
        for identifier in identifiers:
            if self.check_evaluation(identifier):
                self.clean_evaluation(identifier)
            else:
                raise RuntimeError("Evaluation {} is not ready".format(identifier))

        self.ping()
