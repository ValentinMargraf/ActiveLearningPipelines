from abc import ABC, abstractmethod

import json

import mysql.connector

from ALP.benchmark.ActiveLearningScenario import ActiveLearningScenario
from ALP.benchmark.ActiveLearningSetting import ActiveLearningSetting
from ALP.common import format_insert_query, format_select_query


class BenchmarkConnector(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def load_scenario(self, scenario_id):
        pass

    @abstractmethod
    def load_or_create_scenario(self, openml_id, seed, setting_id):
        pass

    @abstractmethod
    def load_setting(self, setting_id):
        pass

    @abstractmethod
    def load_or_create_setting(self, name, labeled_train_size, train_type, test_size, number_of_iterations,
                               number_of_samples):
        pass


class MySQLBenchmarkConnector(BenchmarkConnector):
    scenario_table = "salt_scenario"
    setting_table = "salt_setting"

    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database

        self.con = mysql.connector.connect(user=user, password=password, database=database)

        setting_table_query = f"CREATE TABLE IF NOT EXISTS {MySQLBenchmarkConnector.setting_table} (" \
                              f"setting_id INT(10) NOT NULL AUTO_INCREMENT PRIMARY KEY, setting_name VARCHAR(250), " \
                              f"setting_labeled_train_size VARCHAR(50), setting_train_type VARCHAR(250), " \
                              f"setting_test_size VARCHAR(50), number_of_iterations INT(10), number_of_samples INT(10))"
        scenario_table_query = f"CREATE TABLE IF NOT EXISTS {MySQLBenchmarkConnector.scenario_table} (" \
                               f"scenario_id INT(10) NOT NULL AUTO_INCREMENT PRIMARY KEY, openml_id INT(10), " \
                               f"test_split_seed INT(10), train_split_seed INT(10), seed INT(10), " \
                               f"labeled_indices LONGTEXT, test_indices LONGTEXT, setting_id INT(10))"
        cursor = self.con.cursor()
        cursor.execute(setting_table_query)
        cursor.execute(scenario_table_query)

    def load_scenario(self, scenario_id):
        query = format_select_query(MySQLBenchmarkConnector.scenario_table, {"scenario_id": scenario_id})
        cursor = self.con.cursor(buffered=True, dictionary=True)
        cursor.execute(query)
        res = cursor.fetchall()
        if len(res) > 0:
            scenario_data = res[0]
            setting = self.load_setting(res[0]["setting_id"])
            scenario_data.pop("setting_id")
            scenario_data["setting"] = setting
            return ActiveLearningScenario(**scenario_data)
        else:
            raise Exception("Scenario with ID " + str(scenario_id) + " unknown")

    def load_or_create_scenario(self, openml_id, test_split_seed, train_split_seed, seed, setting_id):
        setting = self.load_setting(setting_id)
        scenario_data = {
            "openml_id": openml_id,
            "test_split_seed": test_split_seed,
            "train_split_seed": train_split_seed,
            "seed": seed,
            "setting_id": setting_id
        }
        check_query = format_select_query(MySQLBenchmarkConnector.scenario_table, scenario_data)
        cursor = self.con.cursor(buffered=True, dictionary=True)
        cursor.execute(check_query)
        res_check = cursor.fetchall()
        cursor.close()

        scenario_data["setting"] = setting
        scenario_data.pop("setting_id")

        if len(res_check) > 0:
            scenario_data["scenario_id"] = int(res_check[0]["scenario_id"])
            scenario_data["labeled_indices"] = json.loads(res_check[0]["labeled_indices"])
            scenario_data["test_indices"] = json.loads(res_check[0]["test_indices"])
            return ActiveLearningScenario(**scenario_data)

        scenario_data["scenario_id"] = -1
        new_scenario = ActiveLearningScenario(**scenario_data)
        scenario_data.pop("setting")
        scenario_data.pop("scenario_id")
        scenario_data["setting_id"] = setting_id
        scenario_data["labeled_indices"] = str(new_scenario.get_labeled_instances())
        scenario_data["test_indices"] = str(new_scenario.get_test_indices())

        insert_query = format_insert_query(MySQLBenchmarkConnector.scenario_table, scenario_data)
        print(insert_query)
        cursor = self.con.cursor()
        cursor.execute(insert_query)
        scenario_id = cursor.lastrowid
        self.con.commit()

        new_scenario.scenario_id = scenario_id
        return new_scenario

    def load_setting(self, setting_id):
        query = format_select_query(MySQLBenchmarkConnector.setting_table, {"setting_id": setting_id})
        cursor = self.con.cursor(buffered=True, dictionary=True)
        cursor.execute(query)
        res = cursor.fetchall()
        if len(res) > 0:
            return ActiveLearningSetting.from_dict(res[0])
        else:
            raise Exception("Setting with ID " + str(setting_id) + " could not be found.")

    def load_or_create_setting(self, name, labeled_train_size, train_type, test_size, number_of_iterations,
                               number_of_samples):
        """
        This method checks whether the specified setting already exists. If so, it just fetches the data from the
        database and returns an instance. If not, the specified setting is added to the database and then also returned
        to the invoker.
        """
        setting_descriptor = {
            "setting_name": name,
            "setting_labeled_train_size": labeled_train_size,
            "setting_train_type": train_type,
            "setting_test_size": test_size,
            "number_of_iterations": number_of_iterations,
            "number_of_samples": number_of_samples
        }

        # check whether the specified setting already exists. if so, fetch its id from the database and return an
        # instance of that setting
        query_check = format_select_query(MySQLBenchmarkConnector.setting_table, setting_descriptor)
        print(query_check)
        cursor = self.con.cursor(buffered=True, dictionary=True)
        cursor.execute(query_check)
        res_check = cursor.fetchall()
        cursor.close()

        print(res_check)
        if len(res_check) > 0:
            print("fetched setting from db")
            setting_descriptor["setting_id"] = res_check[0]["setting_id"]
            return ActiveLearningSetting.from_dict(setting_descriptor)

        # The specified setting does not yet exist so create it in the database and then return it to the invoker.
        query = format_insert_query(MySQLBenchmarkConnector.setting_table, setting_descriptor)
        cursor = self.con.cursor()
        cursor.execute(query)
        inserted_id = cursor.lastrowid
        self.con.commit()

        return ActiveLearningSetting(setting_id=inserted_id, setting_name=name, setting_test_size=test_size,
                                     setting_train_type=train_type, setting_labeled_train_size=labeled_train_size,
                                     number_of_iterations=number_of_iterations, number_of_samples=number_of_samples)
