import numpy as np
import openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from ALP.benchmark.ActiveLearningSetting import ActiveLearningSetting


def create_dataset_split(X, y, test_split_seed, test_split_size: float, train_split_seed, train_split_size,
                         train_split_type):
    # initialize list of indices
    indices = np.arange(0, len(X))

    # split data into train and test and retrieve test_indices to be returned later
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, indices,
                                                                                     test_size=test_split_size,
                                                                                     random_state=test_split_seed)
    # determine the proportion of unlabeled data, also in case the train split is given in terms of an absolute number
    # of labeled data points
    unlabeled_size = 1-train_split_size
    if train_split_type == "absolute":
        unlabeled_size = 1 - train_split_size/len(X_train)

    # split data into labeled and unlabeled
    X_l, X_u, y_l, y_u, labeled_indices, unlabeled_indices = train_test_split(X_train, y_train, train_indices,
                                                                              test_size=unlabeled_size,
                                                                              random_state=train_split_seed)

    return labeled_indices.tolist(), test_indices.tolist()


class ActiveLearningScenario:

    def __init__(self, scenario_id, openml_id, test_split_seed, train_split_seed, seed, setting: ActiveLearningSetting,
                 labeled_indices=None, test_indices=None):
        self.scenario_id = scenario_id
        self.openml_id = openml_id
        self.test_split_seed = test_split_seed
        self.train_split_seed = train_split_seed
        self.seed = seed
        self.labeled_indices = labeled_indices
        self.test_indices = test_indices
        self.setting = setting

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
        X = OrdinalEncoder().fit_transform(X)
        X = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(X)
        self.X = X
        self.y = LabelEncoder().fit_transform(y)

        if test_indices is None or labeled_indices is None:
            self.labeled_indices, self.test_indices = create_dataset_split(X, y, test_split_seed,
                                                                           setting.setting_test_size, train_split_seed,
                                                                           setting.setting_labeled_train_size,
                                                                           setting.setting_train_type)

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

    def __repr__(self):
        params = dict(self.__dict__)
        params.pop("X")
        params.pop("y")
        return "<ActiveLearningScenario> " + str(params)
