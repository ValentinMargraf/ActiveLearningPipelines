from abc import ABC, abstractmethod

import numpy as np
from skactiveml.pool import RandomSampling


class Initializer(ABC):

    def __init__(self):
        return

    @abstractmethod
    def sample(self, X_u, num_queries):
        pass


class RandomInitializer(Initializer):
    def __init__(self, seed):
        self.qs = RandomSampling(random_state=seed)

    def sample(self, X_u, num_queries):
        nan_labels = np.full(len(X_u), np.nan, dtype=float)
        queried_ids = self.qs.query(X=X_u, y=nan_labels, batch_size=num_queries)
        return queried_ids
