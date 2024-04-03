from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from ALP.benchmark.BenchmarkConnector import MySQLBenchmarkConnector
from ALP.pipeline.SamplingStrategy import (
    BatchBaldSampling,
    DiscriminativeSampling,
    EntropySampling,
    ExpectedAveragePrecision,
    LeastConfidentSampling,
    MarginSampling,
    MinMarginSampling,
    MonteCarloEERLogLoss,
    MonteCarloEERMisclassification,
    PowerMarginSampling,
    QueryByCommitteeEntropySampling,
    QueryByCommitteeKLSampling,
    RandomMarginSampling,
    RandomSamplingStrategy,
    TypicalClusterSampling,
    WeightedClusterSampling,
)


def ensure_default_setup(dbbc: MySQLBenchmarkConnector):
    # init settings
    settings = [{
        'name': 'small',
        'labeled_train_size': 30,
        'train_type': 'absolute',
        'test_size': 0.33,
        'number_of_samples': 1,
        'number_of_iterations': 20
    }, {
        'name': 'medium',
        'labeled_train_size': 100,
        'train_type': 'absolute',
        'test_size': 0.33,
        'number_of_samples': 5,
        'number_of_iterations': 20
    }, {
        'name': 'large-10',
        'labeled_train_size': 300,
        'train_type': 'absolute',
        'test_size': 0.33,
        'number_of_samples': 5,
        'number_of_iterations': 20
    }, {
        'name': 'large-20',
        'labeled_train_size': 100,
        'train_type': 'absolute',
        'test_size': 0.33,
        'number_of_samples': 5,
        'number_of_iterations': 20
    }]
    for setting in settings:
        dbbc.load_or_create_setting(**setting)

    default_sampling_strategies = {
        "random": RandomSamplingStrategy(42),
        "random_margin": RandomMarginSampling(42),
        "entropy": EntropySampling(42),
        "margin": MarginSampling(42),
        "least_confident": LeastConfidentSampling(42),
        "mc_logloss": MonteCarloEERLogLoss(42),
        "mc_misclass": MonteCarloEERMisclassification(42),
        "discrim": DiscriminativeSampling(42),
        "qbc_entropy": QueryByCommitteeEntropySampling(42, 10),
        "qbc_kl": QueryByCommitteeKLSampling(42, 10),
        "bald": BatchBaldSampling(42, 10),
        "power_margin": PowerMarginSampling(42),
        "min_margin": MinMarginSampling(42),
        "expected_avg": ExpectedAveragePrecision(42),
        "typ_cluster": TypicalClusterSampling(42),
        "weighted_cluster": WeightedClusterSampling(42)
    }
    for k, obj in default_sampling_strategies.items():
        dbbc.load_or_create_sampling_strategy(k, obj)

    default_learners = {
        "svm_lin": SVC(kernel='linear', probability=True),
        "svm_rbf": SVC(kernel='rbf', probability=True),
        "rf_entropy": RandomForestClassifier(n_estimators=100, max_depth=10, criterion='entropy'),
        "rf_gini": RandomForestClassifier(n_estimators=100, max_depth=10, criterion='gini'),
        "rf_entropy_large": RandomForestClassifier(n_estimators=250, max_depth=10, criterion='entropy'),
        "rf_gini_large": RandomForestClassifier(n_estimators=250, max_depth=10, criterion='gini'),
        "knn_3": KNeighborsClassifier(n_neighbors=3),
        "knn_10": KNeighborsClassifier(n_neighbors=10),
        "log_reg": LogisticRegression(),
        "multinomial_bayes": MultinomialNB(),
        "etc_entropy": ExtraTreesClassifier(n_estimators=100, max_depth=10, criterion='entropy'),
        "etc_gini": ExtraTreesClassifier(n_estimators=100, max_depth=10, criterion='gini'),
        "etc_entropy_large": ExtraTreesClassifier(n_estimators=250, max_depth=10, criterion='entropy'),
        "etc_gini_large": ExtraTreesClassifier(n_estimators=250, max_depth=10, criterion='gini'),
        "naive_bayes": GaussianNB(),
        "mlp": MLPClassifier(),
        "GBT_logloss": GradientBoostingClassifier(n_estimators=100),
        "GBT_exp": GradientBoostingClassifier(n_estimators=100, loss='exponential'),
        "GBT_logloss_large": GradientBoostingClassifier(n_estimators=250),
        "GBT_exp_large": GradientBoostingClassifier(n_estimators=250, loss='exponential')
    }
    for k, obj in default_learners.items():
        dbbc.load_or_create_learner(k, obj)
