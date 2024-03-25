

class Scenario:

    def __init__(self):
        self.scenario_id = 0
        self.openml_id = 0
        self.seed = 0
        self.labeled_indices = []
        self.test_indices = []
        self.setting_id = 0

        # actual data
        self.X = None
        self.y = None

    def create_or_load(self, openml_id, seed, setting_id):
        self.openml_id = openml_id
        self.seed = seed
        self.setting_id = setting_id

        # TODO: Lookup in stored data whether this combination is already known otherwise just create a new scenario
        #  and store it in the database

    def load_scenario(self, scenario_id):
        self.scenario_id = scenario_id

        # TODO: load scenario from database

    def get_scenario_id(self):
        return self.scenario_id

    def get_openml_id(self):
        return self.openml_id

    def get_setting_id(self):
        return self.setting_id

    def get_seed(self):
        return self.seed

    def get_labeled_instances(self):
        return self.labeled_indices

    def get_test_indices(self):
        return self.test_indices

    def get_labeled_train_data(self):
        return self.X[self.labeled_indices], self.y[self.labeled_indices]

    def get_unlabeled_train_data(self):
        combined_train_labeled_test = self.labeled_indices + self.test_indices
        return self.X - self.X[combined_train_labeled_test], self.y - self.y[combined_train_labeled_test]

    def get_test_data(self):
        return self.X[self.test_indices], self.y[self.test_indices]
