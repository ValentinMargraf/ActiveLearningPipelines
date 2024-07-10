import numpy as np
from py_experimenter.exceptions import DatabaseConnectionError
from py_experimenter.experimenter import PyExperimenter, ResultProcessor
import types
from alpbench.benchmark.BenchmarkConnector import DataFileBenchmarkConnector
from alpbench.evaluation.experimenter.DefaultSetup import ensure_default_setup
from alpbench.evaluation.experimenter.LogTableObserver import LogTableObserver, SparseLogTableObserver
from alpbench.pipeline.ActiveLearningPipeline import ActiveLearningPipeline
from alpbench.pipeline.Oracle import Oracle
import sqlite3
import pandas as pd
import os


# loads parameters from the grid and runs each active learning pipeline on every active learning problem
class ExperimentRunner:

    def __init__(self):
        pass

    def run_experiment(self, parameters: dict, result_processor: ResultProcessor, custom_config: dict):

        dbbc = DataFileBenchmarkConnector()

        connector: DataFileBenchmarkConnector = dbbc

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

        QUERY_STRATEGY = connector.load_query_strategy_by_name(parameters["query_strategy_name"])

        LEARNER = connector.load_learner_by_name(parameters["learner_name"])

        OBSERVER = [SparseLogTableObserver(result_processor, X_test, y_test)]

        ALP = ActiveLearningPipeline(
            learner=LEARNER,
            query_strategy=QUERY_STRATEGY,
            observer_list=OBSERVER,
            num_iterations=setting.get_number_of_iterations(),
            num_queries_per_iteration=setting.get_number_of_queries(),
        )

        oracle = Oracle(X_u, y_u)
        ALP.active_fit(X_l, y_l, X_u, oracle)



def run(db_spec, run_setup=False, reset_experiments=False, setting_combinations=None, parameters=None):


    experimenter = PyExperimenter(experiment_configuration_file_path=db_spec)

    if run_setup:
        benchmark_connector = DataFileBenchmarkConnector()
        ensure_default_setup(dbbc=benchmark_connector)

        benchmark_connector.cleanup()

        setting_combinations = []
        setting_combinations += [{"setting_name": "small"}]

        if reset_experiments:
            experimenter.reset_experiments("running", "failed")

        else:
            experimenter.fill_table_from_combination(
                parameters=parameters,
                fixed_parameter_combinations=setting_combinations,
            )

    else:
        er = ExperimentRunner()
        experimenter.execute(er.run_experiment, -1)