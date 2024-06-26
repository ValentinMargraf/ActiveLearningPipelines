import sklearn.metrics

from alpbench.benchmark.BenchmarkConnector import MySQLBenchmarkConnector
from alpbench.evaluation.experimenter.DefaultSetup import ensure_default_setup
from alpbench.pipeline.ActiveLearningPipeline import ActiveLearningPipeline
from alpbench.pipeline.Oracle import Oracle

# create benchmark connector and establish database connection
benchmark_connector = MySQLBenchmarkConnector("host", "user", "password", "database", use_ssl=False)

# load some default settings and algorithm choices
ensure_default_setup(benchmark_connector)

# fetch setting and scenario
setting = benchmark_connector.load_setting_by_name("small")
scenario = benchmark_connector.load_or_create_scenario(
    openml_id=31, train_split_seed=1337, test_split_seed=42, seed=0, setting_id=setting.get_setting_id()
)

# fetch labeled, unlabeled, test split
X_l, y_l, X_u, y_u, X_test, y_test = scenario.get_data_split()

# instantiate active learning pipeline
query_strategy = benchmark_connector.load_query_strategy_by_name("margin")
learner = benchmark_connector.load_learner_by_name("rf100")
oracle = Oracle(X_u=X_u, y_u=y_u)
alp = ActiveLearningPipeline(learner=learner, query_strategy=query_strategy, oracle=oracle)

# fit / predict and evaluate predictions
alp.active_fit(X_l=X_l, y_l=y_l, X_u=X_u, oracle=oracle)
y_hat = alp.predict(X=X_test)
print("test acc", sklearn.metrics.accuracy_score(y_test, y_hat))
