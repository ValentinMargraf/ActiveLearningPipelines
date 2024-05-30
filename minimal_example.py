from sklearn.metrics import accuracy_score

from ALP.benchmark.BenchmarkConnector import DataFileBenchmarkConnector
from ALP.evaluation.experimenter.DefaultSetup import ensure_default_setup
from ALP.pipeline.ALTEvaluator import ALTEvaluator

# create benchmark connector and establish database connection
benchmark_connector = DataFileBenchmarkConnector()

# load some default settings and algorithm choices
ensure_default_setup(benchmark_connector)

evaluator = ALTEvaluator(
    benchmark_connector=benchmark_connector,
    setting_name="small_dynamic",
    openml_id=3,
    sampling_strategy_name="entropy",
    learner_name="catboost",
)

alp = evaluator.fit()
print("Active learning pipeline fitted")

# fit / predict and evaluate predictions
X_test, y_test = evaluator.get_test_data()
y_hat = alp.predict(X=X_test)
print("Final test acc", accuracy_score(y_test, y_hat))
