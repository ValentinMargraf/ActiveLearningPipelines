import sklearn.metrics

from ALP.pipeline.ActiveLearningPipeline import ActiveLearningPipeline

from ALP.benchmark.BenchmarkConnector import MySQLBenchmarkConnector
from ALP.evaluation.experimenter.DefaultSetup import ensure_default_setup

# create benchmark connector and establish database connection
from ALP.pipeline.Oracle import Oracle
from ALP.pipeline.SALTEvaluator import SALTEvaluator

benchmark_connector = MySQLBenchmarkConnector("host", "user", "password", "database")
# load some default settings and algorithm choices
ensure_default_setup(benchmark_connector)

salt = SALTEvaluator(setting_name="small", openml_id=31, test_split_seed=42, sampling_strategy_name="margin",
                     learner_name="rf-gini")
alp = salt.fit()

# fit / predict and evaluate predictions
X_test, y_test = salt.get_test_data()
y_hat = alp.predict(X=X_test)
print("test acc", sklearn.metrics.accuracy_score(y_test, y_hat))
