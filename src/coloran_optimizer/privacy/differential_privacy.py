from diffprivlib.mechanisms import Laplace

class DifferentialPrivacyManager:
    def __init__(self, epsilon: float = 1.0, delta: float = 0.0):
        self.epsilon = epsilon
        self.delta = delta

    def privatize_data(self, data, sensitivity: float):
        mechanism = Laplace(epsilon=self.epsilon, delta=self.delta, sensitivity=sensitivity)
        return mechanism.randomise(data)
