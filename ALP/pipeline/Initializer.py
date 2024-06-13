from abc import ABC, abstractmethod

import numpy as np
from skactiveml.pool import RandomSampling


class Initializer(ABC):
    """Initializer

    This abstract class initially queries instances of the data to costruct the set of labeled data.
    """

    def __init__(self):
        return

    @abstractmethod
    def sample(self, X_u, num_samples):
        """
        Abstract method, that queries the initial instances of the data.
        """
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
        super().__init__()
        self.qs = RandomSampling(random_state=seed)

    def sample(self, X_u, num_samples):
        """Queries the initial instances of the data, sampling uniformly at random

        Parameters:
            X_u (np.ndarray): The unlabeled data.
            num_samples (int): The number of instances to query.
        Returns:
            np.ndarray: The indices of the queried instances.
        """
        nan_labels = np.full(len(X_u), np.nan, dtype=float)
        queried_ids = self.qs.query(X=X_u, y=nan_labels, batch_size=num_samples)
        return queried_ids
