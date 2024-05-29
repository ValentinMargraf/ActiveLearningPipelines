import types

import numpy as np
from py_experimenter.experimenter import PyExperimenter, ResultProcessor

from ALP.benchmark.BenchmarkConnector import MySQLBenchmarkConnector
from ALP.benchmark.BenchmarkSuite import (
    OpenMLBenchmarkSuite,
    SALTBenchmarkSuiteLarge,
    SALTBenchmarkSuiteMedium,
    SALTBenchmarkSuiteSmall,
)
from ALP.evaluation.experimenter.LogTableObserver import SparseLogTableObserver
from ALP.pipeline.ActiveLearningPipeline import ActiveLearningPipeline
from ALP.pipeline.Oracle import Oracle

# exp_learner_sampler_file = "config/exp_learner_sampler.yml"
exp_learner_sampler_file = "config/sparse_learner_sampler.yml"
db_config_file = "config/db_conf.yml"


small_openml_ids = SALTBenchmarkSuiteSmall().get_openml_dataset_ids()
medium_openml_ids = SALTBenchmarkSuiteMedium().get_openml_dataset_ids()
large_openml_ids = SALTBenchmarkSuiteLarge().get_openml_dataset_ids()
cc18 = OpenMLBenchmarkSuite(99)
cc18_ids = cc18.get_openml_dataset_ids()
test_openml_ids = [3, 6, 12]  # , 14, 16, 18, 20]
kickout_small = []
kickout_medium = []
kickout_large = []


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
        scenario = connector.load_or_create_scenario(
            openml_id=OPENML_ID,
            test_split_seed=TEST_SPLIT_SEED,
            train_split_seed=TRAIN_SPLIT_SEED,
            seed=SEED,
            setting_id=setting.get_setting_id(),
        )

        X_l, y_l, X_u, y_u, X_test, y_test = scenario.get_data_split()

        SAMPLING_STRATEGY = connector.load_sampling_strategy_by_name(parameters["sampling_strategy_name"])
        LEARNER = connector.load_learner_by_name(parameters["learner_name"])

        # OBSERVER = [LogTableObserver(result_processor, X_test, y_test)]
        OBSERVER = [SparseLogTableObserver(result_processor, X_test, y_test)]

        ALP = ActiveLearningPipeline(
            learner=LEARNER,
            sampling_strategy=SAMPLING_STRATEGY,
            observer_list=OBSERVER,
            # init_budget=INIT_BUDGET,
            num_iterations=setting.get_number_of_iterations(),
            num_samples_per_iteration=setting.get_number_of_samples(),
        )

        oracle = Oracle(X_u, y_u)
        ALP.active_fit(X_l, y_l, X_u, oracle)


def run_parallel(variable, run_setup=False):

    def connect(self):
        # db = MySQLBenchmarkConnector("db01-kiml.kiml.ifi.lmu.de", "valentin_m",
        #                               "thiswillnotbevalidforverylongreallysodontcountonit", "vale_test",
        #                               True)
        db = MySQLBenchmarkConnector("isys-otfml.cs.upb.de", "results", "Hallo333!", "ALP_evaluation", False)
        return db.con

    def _start_transaction(self, connection, readonly=False):
        if not readonly:
            connection.start_transaction()

    from py_experimenter.database_connector_mysql import DatabaseConnectorMYSQL

    DatabaseConnectorMYSQL.connect = connect
    DatabaseConnectorMYSQL._start_transaction = _start_transaction

    experimenter = PyExperimenter(
        experiment_configuration_file_path=exp_learner_sampler_file,
        database_credential_file_path=db_config_file,
        name=variable,
    )

    db_name = experimenter.config.database_configuration.database_name
    db_credentials = experimenter.db_connector._get_database_credentials()
    dbbc = MySQLBenchmarkConnector(
        host=db_credentials["host"],
        user=db_credentials["user"],
        password=db_credentials["password"],
        database=db_name,
        use_ssl=False,
    )

    experimenter.db_connector.connect = types.MethodType(connect, experimenter.db_connector)

    if run_setup:
        from DefaultSetup import ensure_default_setup

        ensure_default_setup(dbbc=dbbc)
        run_setup = False

        setting_combinations = []
        # without 40923
        # small_cc18 = list(set(cc18_ids) - set(40923))
        setting_combinations += [{"setting_name": "small", "openml_id": oid} for oid in cc18_ids[:] if oid != 40923]
        setting_combinations += [{"setting_name": "medium", "openml_id": oid} for oid in cc18_ids[:]]
        setting_combinations += [{"setting_name": "large", "openml_id": oid} for oid in cc18_ids[:]]

        experimenter.fill_table_from_combination(
            parameters={
                "learner_name": [
                    "rf_entropy",
                    "svm_rbf",
                    "svm_lin",
                    "rf_gini",
                    "knn_3",
                    "knn_10",
                    "log_reg",
                    "multinomial_bayes",
                    "etc_entropy",
                    "etc_gini",
                    "naive_bayes",
                    "mlp",
                    "GBT_logloss",
                ],
                "sampling_strategy_name": [
                    "power_margin",
                    "typ_cluster",
                    "weighted_cluster",
                    "random",
                    "min_margin",
                    "entropy",
                    "margin",
                    "least_confident",
                    "discrim",
                    "qbc_entropy",
                    "qbc_kl",
                    "bald",
                ],
                "test_split_seed": np.arange(30),
                "train_split_seed": np.arange(1),
                "seed": np.arange(1),
            },
            fixed_parameter_combinations=setting_combinations,
        )

    else:
        er = ExperimentRunner(dbbc=dbbc)
        experimenter.execute(er.run_experiment, 1)


def main():
    import sys

    variable_job_id = str(sys.argv[1])
    variable_task_id = str(sys.argv[2])

    # concatenate both variables with _
    variable = variable_job_id + "_" + variable_task_id

    run_parallel(variable, run_setup=True)
    # run_parallel(variable, run_setup=False)


if __name__ == "__main__":
    main()
