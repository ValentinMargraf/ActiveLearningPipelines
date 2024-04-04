from sklearn.metrics import accuracy_score

from ALP.benchmark.BenchmarkConnector import DataFileBenchmarkConnector
from ALP.evaluation.experimenter.DefaultSetup import ensure_default_setup
from ALP.pipeline.SALTEvaluator import SALTEvaluator

# create benchmark connector and establish database connection
benchmark_connector = DataFileBenchmarkConnector()

# load some default settings and algorithm choices
ensure_default_setup(benchmark_connector)
benchmark_connector.dump()

salt = SALTEvaluator(benchmark_connector=benchmark_connector,
                     setting_name="small", openml_id=31, sampling_strategy_name="margin", learner_name="rf_gini")
alp = salt.fit()

# fit / predict and evaluate predictions
X_test, y_test = salt.get_test_data()
y_hat = alp.predict(X=X_test)
print("test acc", accuracy_score(y_test, y_hat))
