import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
import pandas as pd
from py_experimenter.experimenter import PyExperimenter
import json
import os
import sqlite3
from scipy.stats import ttest_ind_from_stats
from scipy.stats import ttest_ind as tt


class StudyData():

    def __init__(self, dir_to_save):
        self.dir_to_save = dir_to_save
        if not os.path.exists(dir_to_save + "/"):
            os.makedirs(dir_to_save + "/")
        self.base = None

    def generate_summary_df(self):
        results = pd.read_csv(self.dir_to_save+"/results.csv")
        if self.base == "file":
            results = results.rename(columns={'ID': 'experiment_id'})
        accuracies = pd.read_csv(self.dir_to_save+"/accuracies.csv")
        labeling = pd.read_csv(self.dir_to_save+"/labeling.csv")

        experiment_ids = results['experiment_id'].values

        data = accuracies['model_dict'].values
        max_iter = 20

        exp_ids = []
        for i in experiment_ids[:]:
            if i in labeling['experiment_id'].values and i in accuracies['experiment_id'].values:
                exp_ids.append(i)

        dict_to_fill = pd.DataFrame(columns=['experiment_id', 'iteration', 'test_accuracy', 'test_f1',
                                                'test_precision', 'test_recall', 'test_auc', 'test_log_loss',
                                                'len_X_l'])

        for enum,i in enumerate(exp_ids[:]):
            if enum%100 == 0:
                print("Experiment", enum, "of", len(exp_ids))
            for j in range(int(max_iter) + 1):
                lab = labeling[labeling['experiment_id'] == i]
                label = lab['data_dict'].values
                actual_max_iter = max(map(int, json.loads(data[0]).keys()))
                if j > actual_max_iter:
                    continue
                label_dict = json.loads(label[0])[str(j)]
                acc = accuracies[accuracies['experiment_id'] == i]
                data = acc['model_dict'].values
                dict = json.loads(data[0])[str(j)]
                dict_to_fill.loc[len(dict_to_fill)] = [int(i), int(dict['iteration']), dict['test_accuracy'],
                                                        dict['test_f1'],
                                                        dict['test_precision'], dict['test_recall'],
                                                        dict['test_auc'], dict['test_log_loss'],
                                                        int(label_dict['len_X_l'])]

        merged_df = pd.merge(results, dict_to_fill, on='experiment_id', how='inner')
        merged_df = merged_df[['setting_name', 'openml_id', 'learner_name', 'query_strategy_name',
                                'test_split_seed', 'train_split_seed', 'seed', 'iteration', 'test_accuracy',
                                'test_f1',
                                'test_precision', 'test_recall', 'test_auc', 'test_log_loss', 'len_X_l']]
        end_df = merged_df
        end_df.to_csv(self.dir_to_save + "/summarized_df.csv")
        return end_df


    def generate_aubc_df(self, df):
        aubc_df = pd.DataFrame(columns=['setting_name', 'openml_id', 'learner_name', 'query_strategy_name',
                                'test_split_seed', 'train_split_seed', 'seed', "aubc"])
        for openmlid in df['openml_id'].unique():
            for learner_name in df['learner_name'].unique():
                for query_strategy_name in df['query_strategy_name'].unique():
                    for test_split_seed in df['test_split_seed'].unique():
                        for train_split_seed in df['train_split_seed'].unique():
                            for seed in df['seed'].unique():
                                sub_df = df[(df['openml_id'] == openmlid) & (df['learner_name'] == learner_name) &
                                            (df['query_strategy_name'] == query_strategy_name) &
                                            (df['test_split_seed'] == test_split_seed) &
                                            (df['train_split_seed'] == train_split_seed) &
                                            (df['seed'] == seed)]
                                if len(sub_df) == 0:
                                    continue
                                aubc = np.trapz(sub_df['test_accuracy'].values, sub_df['len_X_l'].values)
                                aubc_df.loc[len(aubc_df)] = [sub_df['setting_name'].values[0], openmlid, learner_name,
                                                            query_strategy_name, test_split_seed, train_split_seed, seed, aubc]

        aubc_df.to_csv(self.dir_to_save + "/aubc_df.csv")

class StudyDataFromFile(StudyData):

    def __init__(self, dir_to_save, source):
            super().__init__(dir_to_save=dir_to_save)
            self.source = source
            self.base = "file"


    def get_dataframes(self):
        # Connect to the database
        conn = sqlite3.connect(self.source)
        # Create a cursor object to interact with the database
        cursor = conn.cursor()

        # Get the list of tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        table_name_dict = {"results": "results", "results__accuracy_log": "accuracies",
                           "results__labeling_log": "labeling"}

        # Display the contents of each table
        for table_name in tables:
            table_name = table_name[0]
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn)
            if table_name in table_name_dict.keys():
                df.to_csv(self.dir_to_save + "/" + table_name_dict[table_name] + ".csv")



class StudyDataFromDatabase(StudyData):
    def __init__(self, dir_to_save, db_config, scenario_config):
        super().__init__(dir_to_save=dir_to_save)
        self.db_config, self.scenario_config = db_config, scenario_config
        self.base = "database"

    def get_dataframes(self):
        experimenter_scenarios = PyExperimenter(experiment_configuration_file_path=self.scenario_config,
                                                database_credential_file_path=self.db_config)
        results = experimenter_scenarios.get_table()
        results = results.rename(columns={'ID': 'experiment_id'})
        accuracies = experimenter_scenarios.get_logtable('accuracy_log')
        labeling = experimenter_scenarios.get_logtable('labeling_log')

        results.to_csv(self.dir_to_save+"/results.csv")
        accuracies.to_csv(self.dir_to_save+"/accuracies.csv")
        labeling.to_csv(self.dir_to_save+"/labeling.csv")