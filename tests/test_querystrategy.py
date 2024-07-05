import pytest
from sklearn.ensemble import RandomForestClassifier as RF

from alpbench.pipeline.ActiveLearningPipeline import ActiveLearningPipeline
from alpbench.pipeline.Oracle import Oracle
from alpbench.pipeline.QueryStrategy import (
    BALDQueryStrategy,
    ClusterMarginQueryStrategy,
    CoreSetQueryStrategy,
    DiscriminativeQueryStrategy,
    EntropyQueryStrategy,
    FalcunQueryStrategy,
    KMeansQueryStrategy,
    LeastConfidentQueryStrategy,
    MarginQueryStrategy,
    MaxEntropyQueryStrategy,
    MinMarginQueryStrategy,
    PowerBALDQueryStrategy,
    PowerMarginQueryStrategy,
    QBCVarianceRatioQueryStrategy,
    QueryByCommitteeEntropyQueryStrategy,
    QueryByCommitteeKLQueryStrategy,
    RandomMarginQueryStrategy,
    RandomQueryStrategy,
    TypicalClusterQueryStrategy,
    WeightedClusterQueryStrategy,
)

default_query_strategies = {
    "random": RandomQueryStrategy(42),
    "random_margin": RandomMarginQueryStrategy(42),
    "cluster_margin": ClusterMarginQueryStrategy(42),
    "core_set": CoreSetQueryStrategy(42),
    "entropy": EntropyQueryStrategy(42),
    "falcun": FalcunQueryStrategy(42),
    "margin": MarginQueryStrategy(42),
    "max_entropy": MaxEntropyQueryStrategy(42, 10),
    "least_confident": LeastConfidentQueryStrategy(42),
    "kmeans": KMeansQueryStrategy(42),
    "discrim": DiscriminativeQueryStrategy(42),
    "qbc_entropy": QueryByCommitteeEntropyQueryStrategy(42, 10),
    "qbc_kl": QueryByCommitteeKLQueryStrategy(42, 10),
    "bald": BALDQueryStrategy(42, 10),
    "power_margin": PowerMarginQueryStrategy(42),
    "min_margin": MinMarginQueryStrategy(42, 10),
    "typ_cluster": TypicalClusterQueryStrategy(42),
    "weighted_cluster": WeightedClusterQueryStrategy(42),
    "qbc_variance_ratio": QBCVarianceRatioQueryStrategy(42, 10),
    "power_bald": PowerBALDQueryStrategy(42, 10),
}


@pytest.mark.usefixtures("scenario")
def test_pipeline_executions(scenario):
    print("get data split")
    X_l, y_l, X_u, y_u, X_test, y_test = scenario.get_data_split()
    print("defined sampling strategies")
    query_strategies = [value for key, value in default_query_strategies.items()]
    print("setup learner")
    learner = RF(n_estimators=10)

    print("test sampling strategies")
    for query_strategy in query_strategies:
        print("current sampling strategy: ", query_strategy)
        ALP = ActiveLearningPipeline(
            learner=learner,
            query_strategy=query_strategy,
            init_budget=10,
            num_iterations=1,
            num_queries_per_iteration=10,
        )

        oracle = Oracle(X_u, y_u)
        ALP.active_fit(X_l, y_l, X_u, oracle)
