from abc import ABC, abstractmethod


class SamplingStrategy(ABC):
    def __init__(self):
        return

    @abstractmethod
    def sample(self, learner, X_l, y_l, X_u, num_samples):
        pass


class WrappedSamplingStrategy(SamplingStrategy):
    def __init__(self, wrapped_strategy: SamplingStrategy, learner):
        self.wrapped_strategy = wrapped_strategy
        self.learner = learner

    def sample(self, learner, X_l, y_l, X_u, num_samples):
        self.learner.fit(X_l, y_l)
        self.wrapped_strategy.sample(self.learner, X_l, y_l, X_u, num_samples)


class MarginSampling(SamplingStrategy):
    def sample(self, learner, X_l, y_l, X_u, num_samples):
        pass
