from abc import ABC, abstractmethod

import numpy as np
from skactiveml.pool import RandomSampling


class Initializer(ABC):

    def __init__(self):
        return

    @abstractmethod
    def sample(self, X_u, num_samples):
        pass


class RandomInitializer(Initializer):
    """RandomInitializer

    Randomly selects a subset of the unlabeled data to be queried.

    Args:
        seed (int): The seed for the random number generator.
    Attributes:
        qs (class): RandomSampling class from skactiveml.pool.
    """

    def __init__(self, seed):
        self.qs = RandomSampling(random_state=seed)

    def sample(self, X_u, num_samples):
        nan_labels = np.full(len(X_u), np.nan, dtype=float)
        queried_ids = self.qs.query(X=X_u, y=nan_labels, batch_size=num_samples)
        return queried_ids
