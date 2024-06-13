from sklearn.metrics import accuracy_score

from alpbench.benchmark.BenchmarkConnector import DataFileBenchmarkConnector
from alpbench.evaluation.experimenter.DefaultSetup import ensure_default_setup
from alpbench.pipeline.ALPEvaluator import ALPEvaluator

# create benchmark connector and establish database connection
benchmark_connector = DataFileBenchmarkConnector()

# load some default settings and algorithm choices
ensure_default_setup(benchmark_connector)
# save the parameters of the used algorithms into .json files
benchmark_connector.cleanup()

eval = ALPEvaluator(
    benchmark_connector=benchmark_connector,
    setting_name="small",
    openml_id=31,
    query_strategy_name="margin",
    learner_name="rf_gini",
)
alp = eval.fit()

# fit / predict and evaluate predictions
X_test, y_test = eval.get_test_data()
y_hat = alp.predict(X=X_test)
print("test acc", accuracy_score(y_test, y_hat))
