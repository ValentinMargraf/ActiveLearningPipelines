import pytest

from alpbench.pipeline.ActiveLearningPipeline import ActiveLearningPipeline
from alpbench.pipeline.Oracle import Oracle
from alpbench.pipeline.QueryStrategy import RandomQueryStrategy
from alpbench.util.transformer_prediction_interface import TabPFNClassifier


@pytest.mark.usefixtures("scenario")
def test_tabpfn_executions(scenario):
    print("get data split")
    X_l, y_l, X_u, y_u, X_test, y_test = scenario.get_data_split()

    print("setup learner")
    learner = TabPFNClassifier(device="cpu", N_ensemble_configurations=32)
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
