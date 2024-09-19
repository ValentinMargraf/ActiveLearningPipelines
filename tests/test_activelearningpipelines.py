import pytest
from sklearn.ensemble import RandomForestClassifier as RF

from alpbench.benchmark.Observer import Observer
from alpbench.evaluation.experimenter.LogTableObserver import LogTableObserver, SparseLogTableObserver
from alpbench.pipeline.ActiveLearningPipeline import ActiveLearningPipeline
from alpbench.pipeline.Initializer import RandomInitializer
from alpbench.pipeline.Oracle import Oracle
from alpbench.pipeline.QueryStrategy import RandomQueryStrategy


class MockupLogTableObserver(LogTableObserver):
    def __init__(self):
        pass

    def observe_data(self, iteration, X_u_selected, y_u_selected, X_l_aug, y_l_aug, X_u_red, D_l_ind):
        pass

    def observe_model(self, iteration, model):
        pass


class MockupSparseLogTableObserver(SparseLogTableObserver):
    def __init__(self):
        pass

    def observe_data(self, iteration, X_u_selected, y_u_selected, X_l_aug, y_l_aug, X_u_red, D_l_ind):
        pass

    def observe_model(self, iteration, model):
        pass

    def log_data(self, dict):
        pass

    def log_model(self, dict):
        pass


class DummyObserver(Observer):
    def __init__(self):
        self.data_logs = False
        self.model_logs = False

    def observe_data(self, iteration, X_u_selected, y_u_selected, X_l_aug, y_l_aug, X_u_red, D_l_ind):
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
        initially_labeled_indices=scenario.labeled_indices,
    )

    oracle = Oracle(X_u, y_u)
    ALP.active_fit(X_l, y_l, X_u, oracle)
    y_test_hat = ALP.predict(X_test)
    assert len(y_test_hat) == len(y_test)


@pytest.mark.usefixtures("scenario")
def test_pipeline_execution_with_initializer(scenario):
    print("get data split")
    X_l, y_l, X_u, y_u, X_test, y_test = scenario.get_data_split()

    print("setup learner")
    initializer = RandomInitializer(42)
    learner = RF(n_estimators=10)
    query_strategy = RandomQueryStrategy(42)

    ALP = ActiveLearningPipeline(
        initializer=initializer,
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


@pytest.mark.usefixtures("scenario")
def test_pipeline_execution_with_observers(scenario):
    X_l, y_l, X_u, y_u, X_test, y_test = scenario.get_data_split()
    initializers = [None, RandomInitializer(42)]

    for initializer in initializers:
        learner = RF(n_estimators=10)
        query_strategy = RandomQueryStrategy(42)

        ALP = ActiveLearningPipeline(
            initializer=initializer,
            learner=learner,
            query_strategy=query_strategy,
            observer_list=[MockupLogTableObserver(), MockupSparseLogTableObserver()],
            init_budget=10,
            num_iterations=1,
            num_queries_per_iteration=10,
        )

        oracle = Oracle(X_u, y_u)
        ALP.active_fit(X_l, y_l, X_u, oracle)
        y_test_hat = ALP.predict(X_test)
        assert len(y_test_hat) == len(y_test)


@pytest.mark.usefixtures("scenario")
def test_fewer_instances_than_budget(scenario):
    X_l, y_l, X_u, y_u, X_test, y_test = scenario.get_data_split()
    learner = RF(n_estimators=10)
    query_strategy = RandomQueryStrategy(42)

    ALP = ActiveLearningPipeline(
        learner=learner,
        query_strategy=query_strategy,
        init_budget=10,
        num_iterations=2,
        num_queries_per_iteration=10,
        initially_labeled_indices=scenario.labeled_indices,
    )

    oracle = Oracle(X_u, y_u)
    ALP.active_fit(X_l, y_l, X_u[0:12], oracle)
    y_test_hat = ALP.predict(X_test)
    assert len(y_test_hat) == len(y_test)
