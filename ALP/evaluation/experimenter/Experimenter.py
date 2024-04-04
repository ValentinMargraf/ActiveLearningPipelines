import numpy as np
from py_experimenter.experimenter import PyExperimenter, ResultProcessor

from ALP.benchmark.BenchmarkConnector import MySQLBenchmarkConnector
from ALP.benchmark.BenchmarkSuite import (
    SALTBenchmarkSuiteLarge,
    SALTBenchmarkSuiteMedium,
    SALTBenchmarkSuiteSmall,
)
from ALP.evaluation.experimenter.LogTableObserver import LogTableObserver
from ALP.pipeline.ActiveLearningPipeline import ActiveLearningPipeline
from ALP.pipeline.Oracle import Oracle

exp_learner_sampler_file = "config/exp_learner_sampler.yml"
db_config_file = "config/db_conf.yml"

run_setup = True

small_openml_ids = SALTBenchmarkSuiteSmall().get_openml_dataset_ids()
medium_openml_ids = SALTBenchmarkSuiteMedium().get_openml_dataset_ids()
large_openml_ids = SALTBenchmarkSuiteLarge().get_openml_dataset_ids()
test_openml_ids = [3, 6, 12, 14, 16, 18, 20]


class ExperimentRunner:

    def __init__(self, dbbc):
        self.dbbc = dbbc

    def run_experiment(self, parameters: dict, result_processor: ResultProcessor, custom_config: dict):
        connector: MySQLBenchmarkConnector = self.dbbc

        OPENML_ID = int(parameters["openml_id"])
        SETTING_NAME = parameters["setting_name"]
        TEST_SPLIT_SEED = int(parameters["test_split_seed"])
        TRAIN_SPLIT_SEED = int(parameters["train_split_seed"])
        SEED = int(parameters["seed"])

        setting = connector.load_setting_by_name(SETTING_NAME)
        scenario = connector.load_or_create_scenario(openml_id=OPENML_ID, test_split_seed=TEST_SPLIT_SEED,
                                                     train_split_seed=TRAIN_SPLIT_SEED, seed=SEED,
                                                     setting_id=setting.get_setting_id())

        X_l, y_l, X_u, y_u, X_test, y_test = scenario.get_data_split()

        SAMPLING_STRATEGY = connector.load_sampling_strategy_by_name(parameters["sampling_strategy_name"])
        LEARNER = connector.load_learner_by_name(parameters["learner_name"])

        OBSERVER = [LogTableObserver(result_processor, X_test, y_test)]
        ALP = ActiveLearningPipeline(learner=LEARNER, sampling_strategy=SAMPLING_STRATEGY, observer_list=OBSERVER,
                                     # init_budget=INIT_BUDGET,
                                     num_iterations=setting.get_number_of_iterations(),
                                     num_samples_per_iteration=setting.get_number_of_samples())

        oracle = Oracle(X_u, y_u)
        ALP.active_fit(X_l, y_l, X_u, oracle)


def main():
    experimenter = PyExperimenter(experiment_configuration_file_path=exp_learner_sampler_file,
                                  database_credential_file_path=db_config_file)

    db_name = experimenter.config.database_configuration.database_name
    db_credentials = experimenter.db_connector._get_database_credentials()
    dbbc = MySQLBenchmarkConnector(host=db_credentials["host"], user=db_credentials["user"],
                                   password=db_credentials["password"], database=db_name)

    if run_setup:
        from DefaultSetup import ensure_default_setup
        ensure_default_setup(dbbc=dbbc)

        setting_combinations = []
        # setting_combinations += [{'setting_name': 'small', 'openml_id': oid} for oid in small_openml_ids]
        # setting_combinations += [{'setting_name': 'medium', 'openml_id': oid} for oid in medium_openml_ids]
        # setting_combinations += [{'setting_name': 'large-10', 'openml_id': oid} for oid in large_openml_ids]
        # setting_combinations += [{'setting_name': 'large-20', 'openml_id': oid} for oid in large_openml_ids]
        setting_combinations += [{'setting_name': 'small', 'openml_id': oid} for oid in test_openml_ids]

        experimenter.fill_table_from_combination(
            parameters={
                "learner_name": ["svm_lin", "svm_rbf", "rf_entropy", "rf_gini", "rf_entropy_large"],  # ,
                # "rf_gini_large", "knn_3", "knn_10", "log_reg", "multinomial_bayes",
                # "etc_entropy", "etc_gini", "etc_entropy_large", "etc_gini_large",
                # "naive_bayes", "mlp", "GBT_logloss", "GBT_exp", "GBT_logloss_large",
                # "GBT_exp_large"],
                "sampling_strategy_name": ["random", "entropy", "margin", "least_confident", "mc_logloss"],
                # "mc_misclass", "discrim", "qbc_entropy", "qbc_kl", "bald", "power_margin",
                # "random_margin", "min_margin", "expected_avg", "typ_cluster",
                # "weighted_cluster", "random_margin"],
                "test_split_seed": np.arange(1),
                "train_split_seed": np.arange(5),
                "seed": np.arange(5)
            },
            fixed_parameter_combinations=setting_combinations)

    er = ExperimentRunner(dbbc=dbbc)
    experimenter.execute(er.run_experiment, 1)


if __name__ == "__main__":
    main()
