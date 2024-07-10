import os
import sqlite3

import numpy as np
import pandas as pd

from alpbench.benchmark.BenchmarkSuite import OpenMLBenchmarkSuite
from alpbench.evaluation.analysis.get_results import StudyDataFromFile
from alpbench.evaluation.analysis.plot_functions import BudgetPerformancePlot
from alpbench.evaluation.experimenter.Experimenter import run

# create config file that specifies the databasename, tables name, and keyfields
content = """
PY_EXPERIMENTER:
  n_jobs: 1

  Database:
    provider: sqlite
    database: ALPBenchmark
    table:
      name: results
      keyfields:
        "setting_name":
          type: VARCHAR(50)
        "openml_id":
          type: INT
        "learner_name":
          type: VARCHAR(50)
        "query_strategy_name":
          type: VARCHAR(50)
        "test_split_seed":
          type: INT
        "train_split_seed":
          type: INT
        "seed":
          type: INT

    logtables:
      accuracy_log:
        model_dict: TEXT

      labeling_log:
        data_dict: TEXT
"""

# Create config directory if it does not exist
if not os.path.exists("config"):
    os.mkdir("config")

# Create config file
experiment_configuration_file_path = os.path.join("config", "exp_config.yml")
with open(experiment_configuration_file_path, "w") as f:
    f.write(content)
db_spec = "config/exp_config.yml"
# load dataset ids from the benchmark suite
cc18 = OpenMLBenchmarkSuite(99)
# we only take the first 3 for demonstration purposes
cc18_ids = cc18.get_openml_dataset_ids()[:3]
setting_combinations = [{"setting_name": "small"}]
parameters = {
    "learner_name": ["rf_entropy", "svm_rbf"],
    "query_strategy_name": ["random", "margin"],
    "test_split_seed": np.arange(1),
    "train_split_seed": np.arange(1),
    "seed": np.arange(1),
    "openml_id": cc18_ids,
}


# check whether the database has been created
created = False
done = False


def initialization():

    global created
    if not created:

        run(
            db_spec,
            run_setup=True,
            reset_experiments=False,
            setting_combinations=setting_combinations,
            parameters=parameters,
        )
        # Specify the path to your .db file
        db_path = "ALPBenchmark.db"

        # Connect to the database
        conn = sqlite3.connect(db_path)

        # Create a cursor object to interact with the database
        cursor = conn.cursor()

        # Get the list of tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        # Display the contents of each table
        for table_name in tables[:1]:
            table_name = table_name[0]
            print(f"Contents of table {table_name}:")
            query = f"SELECT * FROM {table_name}"
            results_df = pd.read_sql_query(query, conn)

        assert len(results_df) == 12
        created = True

        for index, row in results_df.iterrows():
            if row["status"] != "created":
                created = False

        assert created is True


def experiments():

    global created
    global done

    if not created:
        initialization()

    if not done:
        run(db_spec, run_setup=False, reset_experiments=False)

        # Specify the path to your .db file
        db_name = "ALPBenchmark"
        db_path = db_name + ".db"
        # db_path = "ALPBenchmark.db"
        # Connect to the database
        conn = sqlite3.connect(db_path)

        # Create a cursor object to interact with the database
        cursor = conn.cursor()

        # Get the list of tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        # Display the contents of each table
        for table_name in tables[:1]:
            table_name = table_name[0]
            print(f"Contents of table {table_name}:")
            query = f"SELECT * FROM {table_name}"
            results_df = pd.read_sql_query(query, conn)

        done = True

        for index, row in results_df.iterrows():
            if row["status"] != "done":
                done = False

        assert done is True


def test_plotting_utilities():

    global done
    if not done:
        experiments()

    # Specify the path to your .db file
    db_name = "ALPBenchmark"
    db_path = db_name + ".db"

    study = StudyDataFromFile("DATAFRAMES", db_path)

    study.get_dataframes()

    dataframe = study.generate_summary_df()
    study.generate_aubc_df(dataframe)

    openmlids = dataframe["openml_id"].unique()
    learners = dataframe["learner_name"].unique()

    for oid in openmlids:
        for learner in learners:
            perfPlot = BudgetPerformancePlot(dataframe, oid, learner, "test_accuracy", "FIGURES/")
            perfPlot.generate_plot_data()
            perfPlot.show(show_fig=False)
