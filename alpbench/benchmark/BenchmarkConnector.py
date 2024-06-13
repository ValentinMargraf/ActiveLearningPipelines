import json
from abc import ABC, abstractmethod
from pathlib import Path

import mysql.connector

from alpbench.benchmark.ActiveLearningScenario import ActiveLearningScenario
from alpbench.benchmark.ActiveLearningSetting import ActiveLearningSetting
from alpbench.util.common import format_insert_query, format_select_query, fullname, instantiate_class_by_fqn


class BenchmarkConnector(ABC):
    """Benchmark Connector

    This abstract class defines the interface for a benchmark connector. A benchmark connector is responsible for
    storing and loading all data related to the active learning benchmark. This includes the active learning setting and
    scenario as well as the learner and query strategy with all used parameters. The data is stored in a database or
    file and can be accessed by the respective methods of this class.

    """

    def __init__(self):
        pass

    @abstractmethod
    def load_scenario(self, scenario_id: int):
        """
        Abstract method that loads the scenario with the specified ID from the database.
        """
        pass

    @abstractmethod
    def load_or_create_scenario(self, openml_id, test_split_seed, train_split_seed, seed, setting_id):
        """
        Abstract method that loads the scenario with the specified parameters from the database. If the scenario does
        not exist yet, it is created and then returned to the invoker.
        """
        pass

    @abstractmethod
    def load_setting(self, setting_id: int):
        """
        Abstract method that loads the setting with the specified ID from the database.
        """
        pass

    @abstractmethod
    def load_setting_by_name(self, setting_name):
        """
        Abstract method that loads the setting with the specified name from the database.
        """
        pass

    @abstractmethod
    def load_or_create_setting(
        self, name, labeled_train_size, train_type, test_size, number_of_iterations, number_of_queries, factor
    ):
        """
        Abstract method that loads the setting with the specified parameters from the database. If the setting does
        not exist yet, it is created and then returned to the invoker.
        """
        pass

    @abstractmethod
    def load_learner_by_name(self, learner_name):
        """
        Abstract method that loads the learner with the specified name from the database.
        """
        pass

    @abstractmethod
    def load_learner(self, learner_id: int):
        """
        Abstract method that loads the learner with the specified ID from the database.
        """
        pass

    @abstractmethod
    def load_or_create_learner(self, learner_name, obj):
        """
        Abstract method that loads the learner with the specified parameters from the database. If the learner does
        not exist yet, it is created and then returned to the invoker.
        """
        pass

    @abstractmethod
    def load_query_strategy_by_name(self, query_strategy_name):
        """
        Abstract method that loads the query strategy with the specified name from the database.
        """
        pass

    @abstractmethod
    def load_query_strategy(self, query_strategy_id):
        """
        Abstract method that loads the query strategy with the specified ID from the database.
        """
        pass

    @abstractmethod
    def load_or_create_query_strategy(self, query_strategy_name, obj):
        """
        Abstract method that loads the query strategy with the specified parameters from the database. If the
        query strategy does not exist yet, it is created and then returned to the invoker.
        """
        pass


def _fetch_data_of_descriptor(data, descriptor: dict):
    """
    This method fetches the data from the specified list that matches the specified descriptor. The descriptor is a
    dictionary that contains key-value pairs that must match the data in order to be fetched.

    Parameters:
        data (list): list of dictionaries that contain the data
        descriptor (dict): dictionary that contains the key-value pairs that must match the data

    Returns:
        dict: the data that matches the descriptor
    """
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
    """Data File Benchmark Connector

    This class is an implementation of the BenchmarkConnector interface that stores all data in files. This involves
    the Active Learning Setting and Scenario as well as the Learner and query Strategy with all used parameters.
    The data is stored in JSON files and can be accessed by the respective methods of this class.

    Args:
        learner_file (str): path to the file storing the learner data
        query_strategy_file (str): path to the file storing the query strategy data
        scenario_file (str): path to the file storing the scenario data
        setting_file (str): path to the file storing the setting data

    Attributes:
        learner_file (str): path to the file storing the learner data
        query_strategy_file (str): path to the file storing the query strategy data
        scenario_file (str): path to the file storing the scenario data
        setting_file (str): path to the file storing the setting data

    """

    base_folder = "alpbench/"
    learner_file = base_folder + "learner.json"
    query_strategy_file = base_folder + "query_strategy.json"
    scenario_file = base_folder + "scenario.json"
    setting_file = base_folder + "setting.json"

    def __init__(self, learner_file=None, query_strategy_file=None, scenario_file=None, setting_file=None):
        self.learner_file = learner_file if learner_file is not None else DataFileBenchmarkConnector.learner_file
        if query_strategy_file is not None:
            self.query_strategy_file = query_strategy_file
        else:
            self.query_strategy_file = DataFileBenchmarkConnector.query_strategy_file
        self.scenario_file = scenario_file if scenario_file is not None else DataFileBenchmarkConnector.scenario_file
        self.setting_file = setting_file if setting_file is not None else DataFileBenchmarkConnector.setting_file

        # ensure files exist and are at least empty json arrays
        import os

        for file in [self.scenario_file, self.setting_file, self.learner_file, self.query_strategy_file]:
            p = Path(file)
            os.makedirs(p.parent, exist_ok=True)

            if not os.path.isfile(file):
                with open(file, "w") as f:
                    f.write("[]")
        with open(self.scenario_file) as f:
            self.scenarios = json.load(f)
        with open(self.setting_file) as f:
            self.settings = json.load(f)
        with open(self.learner_file) as f:
            self.learners = json.load(f)
        with open(self.query_strategy_file) as f:
            self.query_strategies = json.load(f)

    def cleanup(self):
        self.dump()

    def dump(self):
        dump_data = [
            (self.setting_file, self.settings),
            (self.scenario_file, self.scenarios),
            (self.learner_file, self.learners),
            (self.query_strategy_file, self.query_strategies),
        ]
        for dd in dump_data:
            with open(dd[0], "w") as f:
                json.dump(obj=dd[1], fp=f, indent=4)

    def load_scenario(self, scenario_id: int):
        """
        This method loads the scenario with the specified ID from the database.

        Parameters:
            scenario_id (int): ID of the scenario to load

        Returns:
            ActiveLearningScenario: the loaded scenario
        """
        stored_scenario = _fetch_data_of_descriptor(self.scenarios, {"scenario_id": scenario_id})
        if stored_scenario is None:
            raise BaseException("No scenario could be found with ID " + str(scenario_id))

        setting = self.load_setting(stored_scenario.pop("setting_id"))
        stored_scenario["setting"] = setting

        return ActiveLearningScenario(**stored_scenario)

    def load_or_create_scenario(self, openml_id, test_split_seed, train_split_seed, seed, setting_id):
        """
        This method loads the scenario with the specified parameters from the database. If the scenario does not exist
        yet, it is created and then returned to the invoker.

        Parameters:
            openml_id (int): ID of the openml dataset
            test_split_seed (int): seed for the test split
            train_split_seed (int): seed for the train split
            seed (int): seed for the scenario
            setting_id (int): ID of the setting

        Returns:
            ActiveLearningScenario: the loaded or created scenario
        """
        descriptor = {
            "openml_id": openml_id,
            "test_split_seed": test_split_seed,
            "train_split_seed": train_split_seed,
            "seed": seed,
            "setting_id": setting_id,
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
        """
        This method loads the setting with the specified ID from the database.

        Parameters:
            setting_id (int): ID of the setting to load

        Returns:
            ActiveLearningSetting: the loaded setting
        """
        stored_setting = _fetch_data_of_descriptor(self.settings, {"setting_id": setting_id})

        if stored_setting is None:
            raise BaseException("No setting could be found with ID " + str(setting_id))

        return ActiveLearningSetting(**stored_setting)

    def load_setting_by_name(self, setting_name):
        """
        This method loads the setting with the specified name from the database.

        Parameters:
            setting_name (str): name of the setting to load

        Returns:
            ActiveLearningSetting: the loaded setting
        """
        stored_setting = _fetch_data_of_descriptor(self.settings, {"setting_name": setting_name})

        if stored_setting is None:
            raise BaseException("No setting could be found with name " + setting_name)

        return ActiveLearningSetting(**stored_setting)

    def load_or_create_setting(
        self, name, labeled_train_size, train_type, test_size, number_of_iterations, number_of_queries, factor
    ):
        """
        This method checks whether the specified setting already exists. If so, it just fetches the data from the
        database and returns an instance. If not, the specified setting is added to the database and then also returned.

        Parameters:
            name (str): name of the setting
            labeled_train_size (str): labeled training size
            train_type (str): type of training
            test_size (str): test size
            number_of_iterations (int): number of iterations
            number_of_queries (int): number of queries
            factor (int): factor

        Returns:
            ActiveLearningSetting: the loaded or created setting
        """
        setting_descriptor = {
            "setting_name": name,
            "setting_labeled_train_size": labeled_train_size,
            "setting_train_type": train_type,
            "setting_test_size": test_size,
            "number_of_iterations": number_of_iterations,
            "number_of_queries": number_of_queries,
            "factor": factor,
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
        """
        This method loads the learner with the specified name from the database.

        Parameters:
            learner_name (str): name of the learner to load

        Returns:
            object: the loaded learner
        """
        stored_learner = _fetch_data_of_descriptor(self.learners, {"learner_name": learner_name})

        if stored_learner is None:
            raise BaseException("No learner could be found with name " + learner_name)

        return instantiate_class_by_fqn(
            stored_learner["learner_class"], json.loads(stored_learner["learner_parameterization"])
        )

    def load_learner(self, learner_id):
        """
        This method loads the learner with the specified ID from the database.

        Parameters:
            learner_id (int): ID of the learner to load

        Returns:
            object: the loaded learner
        """
        stored_learner = _fetch_data_of_descriptor(self.learners, {"learner_id": learner_id})

        if stored_learner is None:
            raise BaseException("No learner could be found with ID " + str(learner_id))

        return instantiate_class_by_fqn(
            stored_learner["learner_class"], json.loads(stored_learner["learner_parameterization"])
        )

    def load_or_create_learner(self, learner_name, obj):
        """
        This method checks whether the specified learner already exists in the database. If not, the specified setting
        is added to the database and then also returned to the invoker.

        Parameters:
            learner_name (str): name of the learner
            obj (object): the learner to load or create

        Returns:
            object: the loaded or created learner
        """
        data = obj.get_params()

        learner_descriptor = {
            "learner_class": fullname(obj),
            "learner_parameterization": json.dumps(data, cls=CustomEncoder),
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

    def load_query_strategy_by_name(self, query_strategy_name):
        """
        This method loads the query strategy with the specified name from the database.

        Parameters:
            query_strategy_name (str): name of the query strategy to load

        Returns:
            object: the loaded query strategy
        """
        stored_query_strategy = _fetch_data_of_descriptor(
            self.query_strategies, {"query_strategy_name": query_strategy_name}
        )

        if stored_query_strategy is None:
            raise BaseException("No query strategy could be found with ID " + query_strategy_name)

        return instantiate_class_by_fqn(
            stored_query_strategy["query_strategy_class"],
            json.loads(stored_query_strategy["query_strategy_parameterization"]),
        )

    def load_query_strategy(self, query_strategy_id):
        """
        This method loads the query strategy with the specified ID from the database.

        Parameters:
            query_strategy_id (int): ID of the query strategy to load

        Returns:
            object: the loaded query strategy
        """
        stored_query_strategy = _fetch_data_of_descriptor(
            self.query_strategies, {"query_strategy_id": query_strategy_id}
        )

        if stored_query_strategy is None:
            raise BaseException("No query strategy could be found with ID " + str(query_strategy_id))

        return instantiate_class_by_fqn(
            stored_query_strategy["query_strategy_class"],
            json.loads(stored_query_strategy["query_strategy_parameterization"]),
        )

    def load_or_create_query_strategy(self, query_strategy_name, obj):
        """
        This method checks whether the specified query strategy already exists in the database. If not, the specified
        query strategy including its parameterization is added to the database and then also returned to the invoker.

        Parameters:
            query_strategy_name (str): name of the query strategy
            obj (object): the query strategy to load or create

        Returns:
            object: the loaded or created query strategy
        """
        query_strategy_descriptor = {
            "query_strategy_class": fullname(obj),
            "query_strategy_parameterization": json.dumps(obj.get_params()),
        }
        stored_query_strategy = _fetch_data_of_descriptor(self.query_strategies, query_strategy_descriptor)

        # check whether the specified setting already exists. if so, fetch its id from the database and return an
        # instance of that setting

        if stored_query_strategy is None:
            # The specified setting does not yet exist so create it in the database and then return it to the invoker.
            max_id = 0
            for query_strategy in self.query_strategies:
                max_id = max(query_strategy["query_strategy_id"], max_id)

            query_strategy_descriptor["query_strategy_id"] = max_id + 1
            query_strategy_descriptor["query_strategy_name"] = query_strategy_name
            self.query_strategies += [query_strategy_descriptor]
        else:
            query_strategy_descriptor["query_strategy_name"] = stored_query_strategy["query_strategy_name"]

        return query_strategy_descriptor["query_strategy_name"], obj


class MySQLBenchmarkConnector(BenchmarkConnector):
    """MySQL Benchmark Connector

    This class is an implementation of the BenchmarkConnector interface that stores all data in a MySQL database. This
    involves the Active Learning Setting and Scenario as well as the Learner and query Strategy with all used
    parameters. The data is stored in MySQL tables and can be accessed by the respective methods of this class.

    Args:
        host (str): host of the MySQL server
        user (str): user of the MySQL server
        password (str): password of the MySQL server
        database (str): database to use
        use_ssl (bool): whether to use SSL

    Attributes:
        host (str): host of the MySQL server
        user (str): user of the MySQL server
        password (str): password of the MySQL server
        database (str): database to use
        use_ssl (bool): whether to use SSL
        con (mysql.connector.connection.MySQLConnection): connection to the MySQL database
        scenario_table (str): name of the table storing the scenarios
        setting_table (str): name of the table storing the settings
        learner_table (str): name of the table storing the learners
        query_strategy_table (str): name of the table storing the query strategies
    """

    scenario_table = "salt_scenario"
    setting_table = "salt_setting"
    learner_table = "salt_learner"
    query_strategy_table = "salt_query_strategy"

    def __init__(self, host, user, password, database, use_ssl):
        super().__init__()
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.use_ssl = use_ssl

        self.con = mysql.connector.connect(
            host=host, user=user, password=password, database=database, ssl_disabled=not use_ssl
        )

        setting_table_query = (
            f"CREATE TABLE IF NOT EXISTS {MySQLBenchmarkConnector.setting_table} ("
            f"setting_id INT(10) NOT NULL AUTO_INCREMENT PRIMARY KEY, "
            f"setting_name VARCHAR(250) UNIQUE, setting_labeled_train_size VARCHAR(50), "
            f"setting_train_type VARCHAR(250), setting_test_size VARCHAR(50), "
            f"number_of_iterations INT(10), number_of_queries INT(10), factor INT(10))"
        )
        scenario_table_query = (
            f"CREATE TABLE IF NOT EXISTS {MySQLBenchmarkConnector.scenario_table} ("
            f"scenario_id INT(10) NOT NULL AUTO_INCREMENT PRIMARY KEY, openml_id INT(10), "
            f"test_split_seed INT(10), train_split_seed INT(10), seed INT(10), "
            f"labeled_indices LONGTEXT, test_indices LONGTEXT, setting_id INT(10))"
        )
        learner_table_query = (
            f"CREATE TABLE IF NOT EXISTS {MySQLBenchmarkConnector.learner_table} ("
            f"learner_id INT(10) NOT NULL AUTO_INCREMENT PRIMARY KEY, "
            f"learner_name VARCHAR(100) UNIQUE, learner_class VARCHAR(250), "
            f"learner_parameterization TEXT)"
        )
        query_strategy_table_q = (
            f"CREATE TABLE IF NOT EXISTS {MySQLBenchmarkConnector.query_strategy_table}"
            f" (query_strategy_id INT(10) NOT NULL AUTO_INCREMENT PRIMARY KEY, "
            f"query_strategy_name VARCHAR(100) UNIQUE,"
            f"query_strategy_class VARCHAR(250), "
            f"query_strategy_parameterization TEXT)"
        )

        cursor = self.con.cursor()
        for q in [setting_table_query, scenario_table_query, learner_table_query, query_strategy_table_q]:
            cursor.execute(q)

    def close(self):
        self.con.close()

    def load_scenario(self, scenario_id):
        """
        This method loads the scenario with the specified ID from the database.

        Parameters:
            scenario_id (int): ID of the scenario to load

        Returns:
            ActiveLearningScenario: the loaded scenario
        """
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
        """
        This method loads the scenario with the specified parameters from the database. If the scenario does not exist
        yet, it is created and then returned to the invoker.

        Parameters:
            openml_id (int): ID of the openml dataset
            test_split_seed (int): seed for the test split
            train_split_seed (int): seed for the train split
            seed (int): seed for the scenario
            setting_id (int): ID of the setting

        Returns:
            ActiveLearningScenario: the loaded or created scenario
        """
        setting = self.load_setting(setting_id)
        scenario_data = {
            "openml_id": openml_id,
            "test_split_seed": test_split_seed,
            "train_split_seed": train_split_seed,
            "seed": seed,
            "setting_id": setting_id,
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
        """
        This method loads the setting with the specified ID from the database.

        Parameters:
            setting_id (int): ID of the setting to load

        Returns:
            ActiveLearningSetting: the loaded setting
        """
        query = format_select_query(MySQLBenchmarkConnector.setting_table, {"setting_id": setting_id})
        cursor = self.con.cursor(buffered=True, dictionary=True)
        cursor.execute(query)
        res = cursor.fetchall()
        if len(res) > 0:
            return ActiveLearningSetting(**res[0])
        else:
            raise Exception("Setting with ID " + str(setting_id) + " could not be found.")

    def load_setting_by_name(self, setting_name):
        """
        This method loads the setting with the specified name from the database.

        Parameters:
            setting_name (str): name of the setting to load

        Returns:
            ActiveLearningSetting: the loaded setting
        """
        query = format_select_query(MySQLBenchmarkConnector.setting_table, {"setting_name": setting_name})
        cursor = self.con.cursor(buffered=True, dictionary=True)
        cursor.execute(query)
        res = cursor.fetchall()
        if len(res) > 0:
            return ActiveLearningSetting(**res[0])
        else:
            raise Exception("Setting with name " + str(setting_name) + " could not be found.")

    def load_or_create_setting(
        self, name, labeled_train_size, train_type, test_size, number_of_iterations, number_of_queries, factor
    ):
        """
        This method checks whether the specified setting already exists. If so, it just fetches the data from the
        database and returns an instance. If not, the specified setting is added to the database and then also returned
        to the invoker.

        Parameters:
            name (str): name of the setting
            labeled_train_size (str): labeled training size
            train_type (str): type of training
            test_size (str): test size
            number_of_iterations (int): number of iterations
            number_of_queries (int): number of queries
            factor (int): factor

        Returns:
            ActiveLearningSetting: the loaded or created setting
        """
        setting_descriptor = {
            "setting_name": name,
            "setting_labeled_train_size": labeled_train_size,
            "setting_train_type": train_type,
            "setting_test_size": test_size,
            "number_of_iterations": number_of_iterations,
            "number_of_queries": number_of_queries,
            "factor": factor,
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
            return ActiveLearningSetting(**setting_descriptor)

        # The specified setting does not yet exist so create it in the database and then return it to the invoker.
        query = format_insert_query(MySQLBenchmarkConnector.setting_table, setting_descriptor)
        cursor = self.con.cursor()
        cursor.execute(query)
        inserted_id = cursor.lastrowid
        self.con.commit()

        return ActiveLearningSetting(
            setting_id=inserted_id,
            setting_name=name,
            setting_test_size=test_size,
            setting_train_type=train_type,
            setting_labeled_train_size=labeled_train_size,
            number_of_iterations=number_of_iterations,
            number_of_queries=number_of_queries,
            factor=factor,
        )

    def load_learner_by_name(self, learner_name):
        """
        This method loads the learner with the specified name from the database.

        Parameters:
            learner_name (str): name of the learner to load

        Returns:
            object: the loaded learner
        """
        query = format_select_query(MySQLBenchmarkConnector.learner_table, {"learner_name": learner_name})
        cursor = self.con.cursor(buffered=True, dictionary=True)
        cursor.execute(query)
        res = cursor.fetchall()
        if len(res) > 0:
            learner_data = res[0]
            return instantiate_class_by_fqn(
                learner_data["learner_class"], json.loads(learner_data["learner_parameterization"])
            )
        else:
            raise Exception("Learner with name " + str(learner_name) + " unknown")

    def load_learner(self, learner_id):
        """
        This method loads the learner with the specified ID from the database.

        Parameters:
            learner_id (int): ID of the learner to load

        Returns:
            object: the loaded learner
        """
        query = format_select_query(MySQLBenchmarkConnector.learner_table, {"learner_id": learner_id})
        cursor = self.con.cursor(buffered=True, dictionary=True)
        cursor.execute(query)
        res = cursor.fetchall()
        if len(res) > 0:
            learner_data = res[0]
            return instantiate_class_by_fqn(
                learner_data["learner_class"], json.loads(learner_data["learner_parameterization"])
            )
        else:
            raise Exception("Learner with ID " + str(learner_id) + " unknown")

    def load_or_create_learner(self, learner_name, obj):
        """
        This method checks whether the specified learner already exists in the database. If not, the specified setting
        is added to the database and then also returned to the invoker.

        Parameters:
            learner_name (str): name of the learner
            obj (object): the learner to load or create

        Returns:
            object: the loaded or created learner
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

    def load_query_strategy_by_name(self, query_strategy_name):
        """
        This method loads the query strategy with the specified name from the database.

        Parameters:
            query_strategy_name (str): name of the query strategy to load

        Returns:
            object: the loaded query strategy
        """
        query = format_select_query(
            MySQLBenchmarkConnector.query_strategy_table, {"query_strategy_name": query_strategy_name}
        )
        cursor = self.con.cursor(buffered=True, dictionary=True)
        cursor.execute(query)
        res = cursor.fetchall()
        if len(res) > 0:
            query_strategy_data = res[0]
            return instantiate_class_by_fqn(
                query_strategy_data["query_strategy_class"],
                json.loads(query_strategy_data["query_strategy_parameterization"]),
            )
        else:
            raise Exception("query strategy with name " + str(query_strategy_name) + " unknown")

    def load_query_strategy(self, query_strategy_id):
        """
        This method loads the query strategy with the specified ID from the database.

        Parameters:
            query_strategy_id (int): ID of the query strategy to load

        Returns:
            object: the loaded query strategy
        """
        query = format_select_query(
            MySQLBenchmarkConnector.query_strategy_table, {"query_strategy_id": query_strategy_id}
        )
        cursor = self.con.cursor(buffered=True, dictionary=True)
        cursor.execute(query)
        res = cursor.fetchall()
        if len(res) > 0:
            query_strategy_data = res[0]
            return instantiate_class_by_fqn(
                query_strategy_data["query_strategy_class"],
                json.loads(query_strategy_data["query_strategy_parameterization"]),
            )
        else:
            raise Exception("query strategy with ID " + str(query_strategy_id) + " unknown")

    def load_or_create_query_strategy(self, query_strategy_name, obj):
        """
        This method checks whether the specified query strategy already exists in the database. If not, the specified
        query strategy including its parameterization is added to the database and then also returned to the invoker.

        Parameters:
            query_strategy_name (str): name of the query strategy
            obj (object): the query strategy to load or create

        Returns:
            object: the loaded or created query strategy
        """
        query_strategy_descriptor = {
            "query_strategy_class": fullname(obj),
            "query_strategy_parameterization": json.dumps(obj.get_params()),
        }

        # check whether the specified setting already exists. if so, fetch its id from the database and return an
        # instance of that setting
        query_check = format_select_query(MySQLBenchmarkConnector.query_strategy_table, query_strategy_descriptor)
        cursor = self.con.cursor(buffered=True, dictionary=True)
        cursor.execute(query_check)
        res_check = cursor.fetchall()
        cursor.close()

        if len(res_check) < 1:
            # The specified setting does not yet exist so create it in the database and then return it to the invoker.
            query_strategy_descriptor["query_strategy_name"] = query_strategy_name
            query = format_insert_query(MySQLBenchmarkConnector.query_strategy_table, query_strategy_descriptor)
            cursor = self.con.cursor()
            cursor.execute(query)
            self.con.commit()
        else:
            query_strategy_descriptor["query_strategy_name"] = res_check[0]["query_strategy_name"]

        return query_strategy_descriptor["query_strategy_name"], obj


class CustomEncoder(json.JSONEncoder):
    """CustomEncoder

    This class is a custom JSON encoder that is used to encode the parameters of a learner or query strategy.
    """

    def default(self, obj):
        """
        This method is called by the JSON encoder to encode the specified object.

        Parameters:
            obj (object): object to encode

        Returns:
            str: the encoded object
        """
        if isinstance(obj, type):
            return str(obj)  # Convert class references to their string representation
        return json.JSONEncoder.default(self, obj)
