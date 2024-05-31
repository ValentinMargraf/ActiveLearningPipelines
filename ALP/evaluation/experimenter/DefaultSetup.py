import catboost as cb
import xgboost as xgb
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier

from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


from ALP.benchmark.BenchmarkConnector import MySQLBenchmarkConnector
from ALP.pipeline.SamplingStrategy import (
    BALDSampling,
    ClusterMargin,
    CoreSetSampling,
    DiscriminativeSampling,
    EntropySampling,
    EpistemicUncertaintySamplingStrategy,
    FalcunSampling,
    KMeansSampling,
    LeastConfidentSampling,
    MarginSampling,
    MaxEntropySampling,
    MinMarginSampling,
    MonteCarloEERLogLoss,
    MonteCarloEERMisclassification,
    PowerBALDSampling,
    PowerMarginSampling,
    QBCVarianceRatioSampling,
    QueryByCommitteeEntropySampling,
    QueryByCommitteeKLSampling,
    RandomMarginSampling,
    RandomSamplingStrategy,
    TypicalClusterSampling,
    WeightedClusterSampling,
)


def ensure_default_setup(dbbc: MySQLBenchmarkConnector):
    # init settings
    settings = [
        {
            "name": "small",
            "labeled_train_size": 30,
            "train_type": "absolute",
            "test_size": 0.33,
            "number_of_samples": 10,
            "number_of_iterations": 20,
            "factor": None,
        },
        {
            "name": "medium",
            "labeled_train_size": 100,
            "train_type": "absolute",
            "test_size": 0.33,
            "number_of_samples": 50,
            "number_of_iterations": 20,
            "factor": None,
        },
        {
            "name": "large",
            "labeled_train_size": 300,
            "train_type": "absolute",
            "test_size": 0.33,
            "number_of_samples": 200,
            "number_of_iterations": 20,
            "factor": None,
        },
        {
            "name": "small_dynamic",
            "labeled_train_size": 10,
            "train_type": "absolute",
            "test_size": 0.33,
            "number_of_samples": 10,
            "number_of_iterations": 20,
            "factor": 5,
        },
        {
            "name": "large_dynamic",
            "labeled_train_size": 10,
            "train_type": "absolute",
            "test_size": 0.33,
            "number_of_samples": 10,
            "number_of_iterations": 20,
            "factor": 20,
        },
    ]
    for setting in settings:
        dbbc.load_or_create_setting(**setting)

    default_sampling_strategies = {
        "random": RandomSamplingStrategy(42),
        "random_margin": RandomMarginSampling(42),
        "cluster_margin": ClusterMargin(42),
        "core_set": CoreSetSampling(42),
        "epistemic": EpistemicUncertaintySamplingStrategy(42),
        "entropy": EntropySampling(42),
        "falcun": FalcunSampling(42),
        "margin": MarginSampling(42),
        "max_entropy": MaxEntropySampling(42, 10),
        "least_confident": LeastConfidentSampling(42),
        "mc_logloss": MonteCarloEERLogLoss(42),
        "mc_misclass": MonteCarloEERMisclassification(42),
        "kmeans": KMeansSampling(42),
        "discrim": DiscriminativeSampling(42),
        "qbc_entropy": QueryByCommitteeEntropySampling(42, 10),
        "qbc_kl": QueryByCommitteeKLSampling(42, 10),
        "bald": BALDSampling(42, 10),
        "power_margin": PowerMarginSampling(42),
        "min_margin": MinMarginSampling(42, 10),
        "typ_cluster": TypicalClusterSampling(42),
        "weighted_cluster": WeightedClusterSampling(42),
        "qbc_variance_ratio": QBCVarianceRatioSampling(42, 10),
        "power_bald": PowerBALDSampling(42, 10),
    }

    for k, obj in default_sampling_strategies.items():
        dbbc.load_or_create_sampling_strategy(k, obj)

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
        "catboost": cb.CatBoostClassifier(iterations=500, depth=6, verbose=0, rsm=0.1),
        "tabnet": TabNetClassifier(verbose=0),
        "tabpfn": TabPFNClassifier(device="cpu", N_ensemble_configurations=32)
    }

    for k, obj in default_learners.items():
        dbbc.load_or_create_learner(k, obj)
