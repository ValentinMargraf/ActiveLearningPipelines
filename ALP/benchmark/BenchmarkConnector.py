from abc import ABC, abstractmethod

import json

import mysql.connector

from ALP.benchmark.ActiveLearningScenario import ActiveLearningScenario
from ALP.benchmark.ActiveLearningSetting import ActiveLearningSetting
from ALP.util.common import (
    format_insert_query,
    format_select_query,
    fullname,
    instantiate_class_by_fqn,
)


class BenchmarkConnector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def load_scenario(self, scenario_id: int):
        pass

    @abstractmethod
    def load_or_create_scenario(self, openml_id, test_split_seed, train_split_seed, seed, setting_id):
        pass

    @abstractmethod
    def load_setting(self, setting_id: int):
        pass

    @abstractmethod
    def load_setting_by_name(self, setting_name):
        pass

    @abstractmethod
    def load_or_create_setting(self, name, labeled_train_size, train_type, test_size, number_of_iterations,
                               number_of_samples):
        pass

    @abstractmethod
    def load_learner_by_name(self, learner_name):
        pass

    @abstractmethod
    def load_learner(self, learner_id: int):
        pass

    @abstractmethod
    def load_or_create_learner(self, learner_name, obj):
        pass

    @abstractmethod
    def load_sampling_strategy_by_name(self, sampling_strategy_name):
        pass

    @abstractmethod
    def load_sampling_strategy(self, sampling_strategy_id):
        pass

    @abstractmethod
    def load_or_create_sampling_strategy(self, sampling_strategy_name, obj):
        pass


def _fetch_data_of_descriptor(data, descriptor: dict):
    fetched_entry = None
    for entry in data:
        does_match = True
        for k, v in descriptor.items():
            if entry[k] != descriptor[k]:
                does_match = False
                break
        if does_match:
            fetched_entry = entry
            break
    return fetched_entry


class DataFileBenchmarkConnector(BenchmarkConnector):
    base_folder = "saltbench/"
    learner_file = base_folder + "learner.json"
    sampling_strategy_file = base_folder + "sampling_strategy.json"
    scenario_file = base_folder + "scenario.json"
    setting_file = base_folder + "setting.json"

    def __init__(self, learner_file=None, sampling_strategy_file=None, scenario_file=None, setting_file=None):
        self.learner_file = learner_file if learner_file is not None else DataFileBenchmarkConnector.learner_file
        if sampling_strategy_file is not None:
            self.sampling_strategy_file = sampling_strategy_file
        else:
            self.sampling_strategy_file = DataFileBenchmarkConnector.sampling_strategy_file
        self.scenario_file = scenario_file if scenario_file is not None else DataFileBenchmarkConnector.scenario_file
        self.setting_file = setting_file if setting_file is not None else DataFileBenchmarkConnector.setting_file

        # ensure files exist and are at least empty json arrays
        import os
        os.makedirs(DataFileBenchmarkConnector.base_folder, exist_ok=True)

        for file in [self.scenario_file, self.setting_file, self.learner_file, self.sampling_strategy_file]:
            if not os.path.isfile(file):
                with open(file, 'w') as f:
                    f.write("[]")

        with open(self.scenario_file, 'r') as f:
            self.scenarios = json.load(f)
        with open(self.setting_file, 'r') as f:
            self.settings = json.load(f)
        with open(self.learner_file, 'r') as f:
            self.learners = json.load(f)
        with open(self.sampling_strategy_file, 'r') as f:
            self.sampling_strategies = json.load(f)

    def __del__(self):
        self.dump()

    def dump(self):
        dump_data = [
            (self.setting_file, self.settings),
            (self.scenario_file, self.scenarios),
            (self.learner_file, self.learners),
            (self.sampling_strategy_file, self.sampling_strategies)
        ]
        for dd in dump_data:
            with open(dd[0], 'w') as f:
                json.dump(obj=dd[1], fp=f, indent=4)

    def load_scenario(self, scenario_id: int):
        stored_scenario = _fetch_data_of_descriptor(self.scenarios, {"scenario_id": scenario_id})
        if stored_scenario is None:
            raise BaseException("No scenario could be found with ID " + str(scenario_id))

        setting = self.load_setting(stored_scenario.pop("setting_id"))
        stored_scenario["setting"] = setting

        return ActiveLearningScenario(**stored_scenario)

    def load_or_create_scenario(self, openml_id, test_split_seed, train_split_seed, seed, setting_id):
        descriptor = {
            "openml_id": openml_id,
            "test_split_seed": test_split_seed,
            "train_split_seed": train_split_seed,
            "seed": seed,
            "setting_id": setting_id
        }
        stored_scenario = _fetch_data_of_descriptor(self.scenarios, descriptor)

        setting = self.load_setting(descriptor.pop("setting_id"))
        descriptor["setting"] = setting

        if stored_scenario is None:
            max_id = 0
            for s in self.scenarios:
                max_id = max(s["scenario_id"], max_id)
            descriptor["scenario_id"] = max_id + 1

            als = ActiveLearningScenario(**descriptor)

            descriptor["labeled_indices"] = als.get_labeled_instances()
            descriptor["test_indices"] = als.get_test_indices()
            descriptor["setting_id"] = setting.get_setting_id()
            descriptor.pop("setting")
            self.scenarios += [dict(descriptor)]

            return als
        else:
            descriptor["scenario_id"] = stored_scenario["scenario_id"]
            return ActiveLearningScenario(**descriptor)

    def load_setting(self, setting_id):
        stored_setting = _fetch_data_of_descriptor(self.settings, {"setting_id": setting_id})

        if stored_setting is None:
            raise BaseException("No setting could be found with ID " + str(setting_id))

        return ActiveLearningSetting(**stored_setting)

    def load_setting_by_name(self, setting_name):
        stored_setting = _fetch_data_of_descriptor(self.settings, {"setting_name": setting_name})

        if stored_setting is None:
            raise BaseException("No setting could be found with ID " + setting_name)

        return ActiveLearningSetting(**stored_setting)

    def load_or_create_setting(self, name, labeled_train_size, train_type, test_size, number_of_iterations,
                               number_of_samples):
        setting_descriptor = {
            "setting_name": name,
            "setting_labeled_train_size": labeled_train_size,
            "setting_train_type": train_type,
            "setting_test_size": test_size,
            "number_of_iterations": number_of_iterations,
            "number_of_samples": number_of_samples
        }

        stored_setting = _fetch_data_of_descriptor(self.settings, setting_descriptor)

        if stored_setting is None:
            max_id = 0
            for s in self.settings:
                max_id = max(s["setting_id"], max_id)

            setting_descriptor["setting_id"] = max_id + 1
            self.settings += [setting_descriptor]
            return ActiveLearningSetting(**setting_descriptor)
        else:
            return ActiveLearningSetting(**stored_setting)

    def load_learner_by_name(self, learner_name):
        stored_learner = _fetch_data_of_descriptor(self.learners, {"learner_name": learner_name})

        if stored_learner is None:
            raise BaseException("No learner could be found with name " + learner_name)

        return instantiate_class_by_fqn(stored_learner["learner_class"],
                                        json.loads(stored_learner["learner_parameterization"]))

    def load_learner(self, learner_id):
        stored_learner = _fetch_data_of_descriptor(self.learners, {"learner_id": learner_id})

        if stored_learner is None:
            raise BaseException("No learner could be found with ID " + str(learner_id))

        return instantiate_class_by_fqn(stored_learner["learner_class"],
                                        json.loads(stored_learner["learner_parameterization"]))

    def load_or_create_learner(self, learner_name, obj):
        learner_descriptor = {
            "learner_class": fullname(obj),
            "learner_parameterization": json.dumps(obj.get_params()),
        }

        # check whether the specified setting already exists. if so, fetch its id from the database and return an
        # instance of that setting
        stored_learner = _fetch_data_of_descriptor(self.learners, learner_descriptor)

        if stored_learner is None:
            # The specified setting does not yet exist so create it in the database and then return it to the invoker.
            learner_descriptor["learner_name"] = learner_name

            max_id = 0
            for learner in self.learners:
                max_id = max(learner["learner_id"], max_id)

            learner_descriptor["learner_id"] = max_id + 1
            self.learners += [learner_descriptor]
        else:
            learner_descriptor["learner_name"] = stored_learner["learner_name"]

        return learner_descriptor["learner_name"], obj

    def load_sampling_strategy_by_name(self, sampling_strategy_name):
        stored_sampling_strategy = _fetch_data_of_descriptor(self.sampling_strategies,
                                                             {"sampling_strategy_name": sampling_strategy_name})

        if stored_sampling_strategy is None:
            raise BaseException("No sampling strategy could be found with ID " + sampling_strategy_name)

        return instantiate_class_by_fqn(stored_sampling_strategy["sampling_strategy_class"], json.loads(
            stored_sampling_strategy["sampling_strategy_parameterization"]))

    def load_sampling_strategy(self, sampling_strategy_id):
        stored_sampling_strategy = _fetch_data_of_descriptor(self.sampling_strategies,
                                                             {"sampling_strategy_id": sampling_strategy_id})

        if stored_sampling_strategy is None:
            raise BaseException("No sampling strategy could be found with ID " + str(sampling_strategy_id))

        return instantiate_class_by_fqn(stored_sampling_strategy["sampling_strategy_class"],
                                        json.loads(stored_sampling_strategy["sampling_strategy_parameterization"]))

    def load_or_create_sampling_strategy(self, sampling_strategy_name, obj):
        sampling_strategy_descriptor = {
            "sampling_strategy_class": fullname(obj),
            "sampling_strategy_parameterization": json.dumps(obj.get_params()),
        }
        stored_sampling_strategy = _fetch_data_of_descriptor(self.sampling_strategies, sampling_strategy_descriptor)

        # check whether the specified setting already exists. if so, fetch its id from the database and return an
        # instance of that setting

        if stored_sampling_strategy is None:
            # The specified setting does not yet exist so create it in the database and then return it to the invoker.
            sampling_strategy_descriptor["sampling_strategy_name"] = sampling_strategy_name
            self.sampling_strategies += [sampling_strategy_descriptor]
        else:
            sampling_strategy_descriptor["sampling_strategy_name"] = stored_sampling_strategy["sampling_strategy_name"]

        return sampling_strategy_descriptor["sampling_strategy_name"], obj


class MySQLBenchmarkConnector(BenchmarkConnector):
    scenario_table = "salt_scenario"
    setting_table = "salt_setting"
    learner_table = "salt_learner"
    sampling_strategy_table = "salt_sampling_strategy"

    def __init__(self, host, user, password, database, use_ssl):
        super().__init__()
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.use_ssl = use_ssl

        self.con = mysql.connector.connect(host=host, user=user, password=password, database=database,
                                           ssl_disabled=not use_ssl)

        setting_table_query = f"CREATE TABLE IF NOT EXISTS {MySQLBenchmarkConnector.setting_table} (" \
                              f"setting_id INT(10) NOT NULL AUTO_INCREMENT PRIMARY KEY, " \
                              f"setting_name VARCHAR(250) UNIQUE, setting_labeled_train_size VARCHAR(50), " \
                              f"setting_train_type VARCHAR(250), setting_test_size VARCHAR(50), " \
                              f"number_of_iterations INT(10), number_of_samples INT(10))"
        scenario_table_query = f"CREATE TABLE IF NOT EXISTS {MySQLBenchmarkConnector.scenario_table} (" \
                               f"scenario_id INT(10) NOT NULL AUTO_INCREMENT PRIMARY KEY, openml_id INT(10), " \
                               f"test_split_seed INT(10), train_split_seed INT(10), seed INT(10), " \
                               f"labeled_indices LONGTEXT, test_indices LONGTEXT, setting_id INT(10))"
        learner_table_query = f"CREATE TABLE IF NOT EXISTS {MySQLBenchmarkConnector.learner_table} (" \
                              f"learner_id INT(10) NOT NULL AUTO_INCREMENT PRIMARY KEY, " \
                              f"learner_name VARCHAR(100) UNIQUE, learner_class VARCHAR(250), " \
                              f"learner_parameterization TEXT)"
        sampling_strategy_table_q = f"CREATE TABLE IF NOT EXISTS {MySQLBenchmarkConnector.sampling_strategy_table}" \
                                    f" (sampling_strategy_id INT(10) NOT NULL AUTO_INCREMENT PRIMARY KEY, " \
                                    f"sampling_strategy_name VARCHAR(100) UNIQUE," \
                                    f"sampling_strategy_class VARCHAR(250), " \
                                    f"sampling_strategy_parameterization TEXT)"

        cursor = self.con.cursor()
        for q in [setting_table_query, scenario_table_query, learner_table_query, sampling_strategy_table_q]:
            cursor.execute(q)

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
            scenario_data["labeled_indices"] = json.loads(scenario_data["labeled_indices"])
            scenario_data["test_indices"] = json.loads(scenario_data["test_indices"])
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
            return self.load_scenario(scenario_data["scenario_id"])

        scenario_data["scenario_id"] = -1
        new_scenario = ActiveLearningScenario(**scenario_data)
        scenario_data.pop("setting")
        scenario_data.pop("scenario_id")
        scenario_data["setting_id"] = setting_id
        scenario_data["labeled_indices"] = str(new_scenario.get_labeled_instances())
        scenario_data["test_indices"] = str(new_scenario.get_test_indices())

        insert_query = format_insert_query(MySQLBenchmarkConnector.scenario_table, scenario_data)
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

    def load_setting_by_name(self, setting_name):
        query = format_select_query(MySQLBenchmarkConnector.setting_table, {"setting_name": setting_name})
        cursor = self.con.cursor(buffered=True, dictionary=True)
        cursor.execute(query)
        res = cursor.fetchall()
        if len(res) > 0:
            return ActiveLearningSetting.from_dict(res[0])
        else:
            raise Exception("Setting with name " + str(setting_name) + " could not be found.")

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
        cursor = self.con.cursor(buffered=True, dictionary=True)
        cursor.execute(query_check)
        res_check = cursor.fetchall()
        cursor.close()

        if len(res_check) > 0:
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

    def load_learner_by_name(self, learner_name):
        query = format_select_query(MySQLBenchmarkConnector.learner_table, {"learner_name": learner_name})
        cursor = self.con.cursor(buffered=True, dictionary=True)
        cursor.execute(query)
        res = cursor.fetchall()
        if len(res) > 0:
            learner_data = res[0]
            return instantiate_class_by_fqn(learner_data["learner_class"],
                                            json.loads(learner_data["learner_parameterization"]))
        else:
            raise Exception("Learner with name " + str(learner_name) + " unknown")

    def load_learner(self, learner_id):
        query = format_select_query(MySQLBenchmarkConnector.learner_table, {"learner_id": learner_id})
        cursor = self.con.cursor(buffered=True, dictionary=True)
        cursor.execute(query)
        res = cursor.fetchall()
        if len(res) > 0:
            learner_data = res[0]
            return instantiate_class_by_fqn(learner_data["learner_class"],
                                            json.loads(learner_data["learner_parameterization"]))
        else:
            raise Exception("Learner with ID " + str(learner_id) + " unknown")

    def load_or_create_learner(self, learner_name, obj):
        """
        This method checks whether the specified learner already exists in the database. If not, the specified setting
        is added to the database and then also returned to the invoker.
        """
        learner_descriptor = {
            "learner_class": fullname(obj),
            "learner_parameterization": json.dumps(obj.get_params()),
        }

        # check whether the specified setting already exists. if so, fetch its id from the database and return an
        # instance of that setting
        query_check = format_select_query(MySQLBenchmarkConnector.learner_table, learner_descriptor)
        cursor = self.con.cursor(buffered=True, dictionary=True)
        cursor.execute(query_check)
        res_check = cursor.fetchall()
        cursor.close()

        if len(res_check) < 1:
            # The specified setting does not yet exist so create it in the database and then return it to the invoker.
            learner_descriptor["learner_name"] = learner_name
            query = format_insert_query(MySQLBenchmarkConnector.learner_table, learner_descriptor)
            cursor = self.con.cursor()
            cursor.execute(query)
            self.con.commit()
        else:
            learner_descriptor["learner_name"] = res_check[0]["learner_name"]

        return learner_descriptor["learner_name"], obj

    def load_sampling_strategy_by_name(self, sampling_strategy_name):
        query = format_select_query(MySQLBenchmarkConnector.sampling_strategy_table,
                                    {"sampling_strategy_name": sampling_strategy_name})
        cursor = self.con.cursor(buffered=True, dictionary=True)
        cursor.execute(query)
        res = cursor.fetchall()
        if len(res) > 0:
            sampling_strategy_data = res[0]
            return instantiate_class_by_fqn(sampling_strategy_data["sampling_strategy_class"],
                                            json.loads(sampling_strategy_data["sampling_strategy_parameterization"]))
        else:
            raise Exception("Sampling strategy with name " + str(sampling_strategy_name) + " unknown")

    def load_sampling_strategy(self, sampling_strategy_id):
        query = format_select_query(MySQLBenchmarkConnector.sampling_strategy_table,
                                    {"sampling_strategy_id": sampling_strategy_id})
        cursor = self.con.cursor(buffered=True, dictionary=True)
        cursor.execute(query)
        res = cursor.fetchall()
        if len(res) > 0:
            sampling_strategy_data = res[0]
            return instantiate_class_by_fqn(sampling_strategy_data["sampling_strategy_class"],
                                            json.loads(sampling_strategy_data["sampling_strategy_parameterization"]))
        else:
            raise Exception("Sampling strategy with ID " + str(sampling_strategy_id) + " unknown")

    def load_or_create_sampling_strategy(self, sampling_strategy_name, obj):
        """
        This method checks whether the specified sampling strategy already exists in the database. If not, the specified
        sampling strategy including its parameterization is added to the database and then also returned to the invoker.
        """
        sampling_strategy_descriptor = {
            "sampling_strategy_class": fullname(obj),
            "sampling_strategy_parameterization": json.dumps(obj.get_params()),
        }

        # check whether the specified setting already exists. if so, fetch its id from the database and return an
        # instance of that setting
        query_check = format_select_query(MySQLBenchmarkConnector.sampling_strategy_table, sampling_strategy_descriptor)
        cursor = self.con.cursor(buffered=True, dictionary=True)
        cursor.execute(query_check)
        res_check = cursor.fetchall()
        cursor.close()

        if len(res_check) < 1:
            # The specified setting does not yet exist so create it in the database and then return it to the invoker.
            sampling_strategy_descriptor["sampling_strategy_name"] = sampling_strategy_name
            query = format_insert_query(MySQLBenchmarkConnector.sampling_strategy_table, sampling_strategy_descriptor)
            cursor = self.con.cursor()
            cursor.execute(query)
            self.con.commit()
        else:
            sampling_strategy_descriptor["sampling_strategy_name"] = res_check[0]["sampling_strategy_name"]

        return sampling_strategy_descriptor["sampling_strategy_name"], obj
