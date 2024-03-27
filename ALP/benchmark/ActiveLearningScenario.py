import numpy as np
import openml
import sklearn.preprocessing as preprocessing
import sklearn.impute as impute

from ALP.benchmark.ActiveLearningSetting import ActiveLearningSetting


class ActiveLearningScenario:

    def __init__(self, scenario_id, openml_id, test_split_seed, train_split_seed, seed, labeled_indices, test_indices,
                 setting: ActiveLearningSetting):
        self.scenario_id = scenario_id
        self.openml_id = openml_id
        self.test_split_seed = test_split_seed
        self.train_split_seed = train_split_seed
        self.seed = seed
        self.labeled_indices = labeled_indices
        self.test_indices = test_indices
        self.setting = setting
        self.imputer = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
        self.encoder = preprocessing.OrdinalEncoder()

        # actual data
        ds = openml.datasets.get_dataset(openml_id)
        # print("dataset info loaded")
        df = ds.get_data()[0]
        # prepare label column as numpy array
        X = np.array(df.drop(columns=[ds.default_target_attribute]).values)
        y = np.array(df[ds.default_target_attribute].values)
        if y.dtype != int:
            y_int = np.zeros(len(y)).astype(int)
            vals = np.unique(y)
            for i, val in enumerate(vals):
                mask = y == val
                y_int[mask] = i
            y = y_int
        self.X = X
        self.y = y

    def preprocess_data(self):
        self.X = self.encoder.fit_transform(self.X)
        self.X = self.imputer.fit_transform(self.X)

    def get_scenario_id(self):
        return self.scenario_id

    def get_openml_id(self):
        return self.openml_id

    def get_setting(self):
        return self.setting

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
        mask = np.array([True] * len(self.X))
        mask[combined_train_labeled_test] = False
        return self.X[mask], self.y[mask]

    def get_test_data(self):
        return self.X[self.test_indices], self.y[self.test_indices]

    def get_data_split(self):
        X_l, y_l = self.get_labeled_train_data()
        X_u, y_u = self.get_unlabeled_train_data()
        X_test, y_test = self.get_test_data()
        return X_l, y_l, X_u, y_u, X_test, y_test
