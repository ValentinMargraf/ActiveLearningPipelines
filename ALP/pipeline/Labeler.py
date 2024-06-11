from abc import ABC, abstractmethod

import numpy as np


class Labeler(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def label(self, learner, X_l, y_l, X_u, num_samples):
        pass


class HighestConfidenceLabeler(Labeler):
    def __init__(self):
        super().__init__()

    def label(self, learner, X_l, y_l, X_u, num_samples):
        y_hat = np.array(learner.predict(X_u))
        y_hat_proba = np.array(learner.predict_proba(X_u))
        x = np.amax(y_hat_proba, axis=1)
        top = np.argpartition(x, -num_samples)[-num_samples:]
        return top, y_hat[top]


class CoLearningLabeler(Labeler):
    def __init__(self, colearner):
        super().__init__()
        self.colearner = colearner

    def label(self, learner, X_l, y_l, X_u, num_samples):
        self.colearner.fit(X_l, y_l)
        return HighestConfidenceLabeler().label(self.colearner, X_l, y_l, X_u, num_samples)
