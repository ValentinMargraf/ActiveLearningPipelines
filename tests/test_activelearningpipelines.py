import pytest

from sklearn.ensemble import RandomForestClassifier as RF
from ALP.benchmark.Observer import Observer
from ALP.pipeline.ActiveLearningPipeline import ActiveLearningPipeline
from ALP.pipeline.Oracle import Oracle
from ALP.pipeline.QueryStrategy import RandomQueryStrategy



class DummyObserver(Observer):
    def __init__(self):
        self.data_logs = False
        self.model_logs = False

    def observe_data(self, iteration, X_u_selected, y_u_selected, X_l_aug, y_l_aug, X_u_red):
        self.data_logs = True

    def observe_model(self, iteration, model):
        self.model_logs = True


@pytest.mark.usefixtures("scenario")
def test_pipeline_executions(scenario):
    print("get data split")
    X_l, y_l, X_u, y_u, X_test, y_test = scenario.get_data_split()

    print("setup learner")
    learner = RF(n_estimators=10)
    query_strategy = RandomQueryStrategy(42)

    ALP = ActiveLearningPipeline(
        learner=learner,
        query_strategy=query_strategy,
        init_budget=10,
        num_iterations=1,
        num_queries_per_iteration=10,
    )

    oracle = Oracle(X_u, y_u)
    ALP.active_fit(X_l, y_l, X_u, oracle)
    y_test_hat = ALP.predict(X_test)
    assert len(y_test_hat) == len(y_test)


