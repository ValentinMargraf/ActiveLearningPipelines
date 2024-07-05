import catboost as cb
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


from alpbench.benchmark.BenchmarkConnector import BenchmarkConnector

from alpbench.pipeline.QueryStrategy import (
    BALDQueryStrategy,
    ClusterMarginQueryStrategy,
    CoreSetQueryStrategy,
    DiscriminativeQueryStrategy,
    EntropyQueryStrategy,
    EpistemicUncertaintyQueryStrategy,
    FalcunQueryStrategy,
    KMeansQueryStrategy,
    LeastConfidentQueryStrategy,
    MarginQueryStrategy,
    MaxEntropyQueryStrategy,
    MinMarginQueryStrategy,
    MonteCarloEERLogLoss,
    MonteCarloEERMisclassification,
    PowerBALDQueryStrategy,
    PowerMarginQueryStrategy,
    QBCVarianceRatioQueryStrategy,
    QueryByCommitteeEntropyQueryStrategy,
    QueryByCommitteeKLQueryStrategy,
    RandomMarginQueryStrategy,
    RandomQueryStrategy,
    TypicalClusterQueryStrategy,
    WeightedClusterQueryStrategy
)

def ensure_default_setup(dbbc: BenchmarkConnector):
    """
    Ensures that the default settings, sampling strategies and learners are loaded into the database.
    This allows to later on load the default settings and strategies.
    """
    # init settings
    settings = [
        {
            "name": "small",
            "labeled_train_size": 30,
            "train_type": "absolute",
            "test_size": 0.33,
            "number_of_queries": 10,
            "number_of_iterations": 20,
            "factor": -1,
        },
        {
            "name": "medium",
            "labeled_train_size": 100,
            "train_type": "absolute",
            "test_size": 0.33,
            "number_of_queries": 50,
            "number_of_iterations": 20,
            "factor": -1,
        },
        {
            "name": "large",
            "labeled_train_size": 300,
            "train_type": "absolute",
            "test_size": 0.33,
            "number_of_queries": 200,
            "number_of_iterations": 20,
            "factor": -1,
        },
        {
            "name": "small_dynamic",
            "labeled_train_size": 10,
            "train_type": "absolute",
            "test_size": 0.33,
            "number_of_queries": 10,
            "number_of_iterations": 20,
            "factor": 5,
        },
        {
            "name": "large_dynamic",
            "labeled_train_size": 10,
            "train_type": "absolute",
            "test_size": 0.33,
            "number_of_queries": 10,
            "number_of_iterations": 20,
            "factor": 20,
        },
    ]
    for setting in settings:
        dbbc.load_or_create_setting(**setting)

    default_query_strategies = {
        "random": RandomQueryStrategy(42),
        "random_margin": RandomMarginQueryStrategy(42),
        "cluster_margin": ClusterMarginQueryStrategy(42),
        "core_set": CoreSetQueryStrategy(42),
        "epistemic": EpistemicUncertaintyQueryStrategy(42),
        "entropy": EntropyQueryStrategy(42),
        "falcun": FalcunQueryStrategy(42),
        "margin": MarginQueryStrategy(42),
        "max_entropy": MaxEntropyQueryStrategy(42, 10),
        "least_confident": LeastConfidentQueryStrategy(42),
        "mc_logloss": MonteCarloEERLogLoss(42),
        "mc_misclass": MonteCarloEERMisclassification(42),
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

    for k, obj in default_query_strategies.items():
        dbbc.load_or_create_query_strategy(k, obj)

    default_learners = {
        "svm_lin": SVC(kernel="linear", probability=True),
        "svm_rbf": SVC(kernel="rbf", probability=True),
        "rf_entropy": RandomForestClassifier(n_estimators=100, max_depth=10, criterion="entropy"),
        "rf_gini": RandomForestClassifier(n_estimators=100, max_depth=10, criterion="gini"),
        "rf_entropy_large": RandomForestClassifier(n_estimators=250, max_depth=10, criterion="entropy"),
        "rf_gini_large": RandomForestClassifier(n_estimators=250, max_depth=10, criterion="gini"),
        "knn_3": KNeighborsClassifier(n_neighbors=3),
        "knn_10": KNeighborsClassifier(n_neighbors=10),
        "log_reg": LogisticRegression(),
        "multinomial_bayes": MultinomialNB(),
        "etc_entropy": ExtraTreesClassifier(n_estimators=100, max_depth=10, criterion="entropy"),
        "etc_gini": ExtraTreesClassifier(n_estimators=100, max_depth=10, criterion="gini"),
        "etc_entropy_large": ExtraTreesClassifier(n_estimators=250, max_depth=10, criterion="entropy"),
        "etc_gini_large": ExtraTreesClassifier(n_estimators=250, max_depth=10, criterion="gini"),
        "naive_bayes": GaussianNB(),
        "mlp": MLPClassifier(),
        "GBT_logloss": GradientBoostingClassifier(n_estimators=100),
        "GBT_exp": GradientBoostingClassifier(n_estimators=100, loss="exponential"),
        "GBT_logloss_large": GradientBoostingClassifier(n_estimators=250),
        "xgb": xgb.XGBClassifier(tree_method="hist", max_depth=6, n_estimators=100),
        "catboost": cb.CatBoostClassifier(rsm=0.1, iterations=100, depth=10, verbose=0, boosting_type='Plain',
                                          bootstrap_type='Bernoulli', subsample=.75),
    }

    for k, obj in default_learners.items():
        dbbc.load_or_create_learner(k, obj)
