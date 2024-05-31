from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score

from ALP.benchmark.BenchmarkConnector import DataFileBenchmarkConnector
from ALP.evaluation.experimenter.DefaultSetup import ensure_default_setup
from ALP.pipeline.ALPEvaluator import ALPEvaluator

# create benchmark connector and establish database connection
benchmark_connector = DataFileBenchmarkConnector()

# load some default settings and algorithm choices
ensure_default_setup(benchmark_connector)

alpeval = ALPEvaluator(
    benchmark_connector=benchmark_connector,
    setting_name="small",
    openml_id=31,
    sampling_strategy_name="cluster_margin",
    learner_name="rf_gini",
)
alpeval.with_learner_obj(TabNetClassifier(verbose=0))
alp = alpeval.fit()

# fit / predict and evaluate predictions
X_test, y_test = alpeval.get_test_data()
y_hat = alp.predict(X=X_test)
print("test acc", accuracy_score(y_test, y_hat))
