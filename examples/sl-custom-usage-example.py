from sklearn.metrics import accuracy_score

from ALP.benchmark.BenchmarkConnector import DataFileBenchmarkConnector
from ALP.evaluation.experimenter.DefaultSetup import ensure_default_setup
from ALP.pipeline.Labeler import CoLearningLabeler, HighestConfidenceLabeler
from ALP.pipeline.SemiSupervisedLearningPipeline import SemiSupervisedLearningPipeline

benchmark_connector = DataFileBenchmarkConnector()
ensure_default_setup(benchmark_connector)

# fetch setting and scenario
setting = benchmark_connector.load_setting_by_name("small")
scenario = benchmark_connector.load_or_create_scenario(
    openml_id=61, train_split_seed=1337, test_split_seed=42, seed=0, setting_id=setting.get_setting_id()
)

X_l, y_l, X_u, y_u, X_test, y_test = scenario.get_data_split()

learner = benchmark_connector.load_learner_by_name("rf_gini")

pipe = SemiSupervisedLearningPipeline(learner=learner, num_iterations=5, num_samples_per_iteration=10)
pipe.semi_supervised_fit(X_l, y_l, X_u, CoLearningLabeler(benchmark_connector.load_learner_by_name("svm_lin")))
y_hat = pipe.predict(X_test)

print(accuracy_score(y_test, y_hat))

pipe.semi_supervised_fit(X_l, y_l, X_u, HighestConfidenceLabeler())
y_hat = pipe.predict(X_test)

print(accuracy_score(y_test, y_hat))
