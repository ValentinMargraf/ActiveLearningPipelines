from abc import ABC, abstractmethod


class Initializer(ABC):

    def __init__(self):
        return

    @abstractmethod
    def sample(self, X_u, num_samples):
        pass
