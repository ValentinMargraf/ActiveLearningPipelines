from ALP.benchmark.ActiveLearningScenario import ActiveLearningScenario
from ALP.pipeline import ActiveLearningPipeline
from ALP.pipeline.Oracle import Oracle


class Evaluator:

    def __init__(self):
        self.active_learning_scenario = None
        self.oracle = None
        self.active_learning_pipeline = None

    def with_active_learning_scenario(self, active_learning_scenario: ActiveLearningScenario):
        self.active_learning_scenario = active_learning_scenario
        return self

    def with_oracle(self, oracle):
        self.oracle = oracle
        return self

    def with_active_learning_pipeline(self, active_learning_pipeline: ActiveLearningPipeline):
        self.active_learning_pipeline = active_learning_pipeline
        return self

    def run(self):
        X_l, y_l, X_u, y_u, X_test, y_test = self.active_learning_scenario.get_data_split()
        self.oracle.set_data(X_u, y_u)

        self.active_learning_pipeline.active_fit(X_l, y_l, X_u, self.oracle)


eval = Evaluator()
alp = ActiveLearningPipeline()
eval.with_oracle(Oracle()).with_active_learning_scenario()
