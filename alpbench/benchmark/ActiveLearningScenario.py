import numpy as np
import openml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from alpbench.benchmark.ActiveLearningSetting import ActiveLearningSetting


def create_dataset_split(
    X, y, test_split_seed, test_split_size: float, train_split_seed, train_split_size, train_split_type, factor
):
    """This method creates a split of the data into labeled, unlabeled and test data. The type of the split can be
    either absolute (i.e., a fixed number of labeled data points) or relative (i.e., a fixed share of the training
    data). The split is stratified according to the labels. The labeled data is guaranteed to contain at least one
    instance of each class. Further, if a factor is given, the number of labeled data points is determined by the
    number of classes times the factor.

    Args:
        X (numpy.ndarray): data
        y (numpy.ndarray): labels
        test_split_seed (int): seed for the test split
        test_split_size (float): size of the test data
        train_split_seed (int): seed for the train split
        train_split_size (float): size of the labeled training data
        train_split_type (str): type of the size parameter: number of data points or share of the (training) dataset
        factor (int): task-dependent factor

    Returns:
        labeled_indices (list): indices of the labeled data
        test_indices (list): indices of the test data
    """

    # initialize list of indices
    indices = np.arange(0, len(X))

    # split data into train and test and retrieve test_indices to be returned later
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, indices, test_size=test_split_size, random_state=test_split_seed, stratify=y
    )
    # determine the proportion of unlabeled data, also in case the train split is given in terms of an absolute number
    # of labeled data points
    unlabeled_size = 1 - train_split_size
    if train_split_type == "absolute":
        if factor != -1:
            train_split_size = factor * len(np.unique(y))
            unlabeled_size = 1 - train_split_size / len(X_train)
        else:
            unlabeled_size = 1 - train_split_size / len(X_train)

    # split data into labeled and unlabeled
    X_l, X_u, y_l, y_u, labeled_indices, unlabeled_indices = train_test_split(
        X_train, y_train, train_indices, test_size=unlabeled_size, random_state=train_split_seed, stratify=y_train
    )

    if len(np.unique(y[labeled_indices])) != len(np.unique(y)):
        # make sure that each class within y is at least once in the labeled data
        for i in np.unique(y):
            if i not in y_l:
                ids = np.where(y_u == i)[0]
                np.random.seed(train_split_seed)
                idx_in_yu = np.random.choice(ids)
                idx = unlabeled_indices[idx_in_yu]
                labeled_indices = np.append(labeled_indices, idx)

    assert len(np.unique(y[labeled_indices])) == len(
        np.unique(y)
    ), "Not all classes are represented in the labeled data"

    return labeled_indices.tolist(), test_indices.tolist()


class ActiveLearningScenario:
    """Active Learning Scenario

    The active learning scenario defines the data and the setting of one active learning setup.  The scenario is
    initialized with the openml id of the dataset, the test split, train split and the seed for reproducibility, the
    setting, and optionally labeled and test indices.

    Args:
        scenario_id (int): id of the scenario in the database
        openml_id (int): id of the dataset on openml
        test_split_seed (int): seed for the test split
        train_split_seed (int): seed for the train split
        seed (int): seed for reproducibility
        setting (ActiveLearningSetting): active learning setting
        labeled_indices (list): indices of the labeled data
        test_indices (list): indices of the test data

    Attributes:
        scenario_id (int): id of the scenario in the database
        openml_id (int): id of the dataset on openml
        test_split_seed (int): seed for the test split
        train_split_seed (int): seed for the train split
        seed (int): seed for reproducibility
        setting (ActiveLearningSetting): active learning setting
        labeled_indices (list): indices of the labeled data
        test_indices (list): indices of the test data

    """

    def __init__(
        self,
        scenario_id,
        openml_id,
        test_split_seed,
        train_split_seed,
        seed,
        setting: ActiveLearningSetting,
        labeled_indices: list = None,
        test_indices: list = None,
    ):
        self.scenario_id = scenario_id
        self.openml_id = openml_id
        self.test_split_seed = test_split_seed
        self.train_split_seed = train_split_seed
        self.seed = seed
        self.labeled_indices = labeled_indices
        self.test_indices = test_indices
        self.setting = setting

        # actual data
        ds = openml.datasets.get_dataset(
            openml_id, download_data=True, download_qualities=True, download_features_meta_data=True
        )
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
        X = SimpleImputer(missing_values=np.nan, strategy="mean").fit_transform(X)

        # filter X for duplicates
        _, unique_indices = np.unique(X, axis=0, return_index=True)

        self.X = X[unique_indices]
        self.y = LabelEncoder().fit_transform(y)[unique_indices]

        if test_indices is None or labeled_indices is None:
            self.labeled_indices, self.test_indices = create_dataset_split(
                self.X,
                self.y,
                test_split_seed,
                setting.setting_test_size,
                train_split_seed,
                setting.setting_labeled_train_size,
                setting.setting_train_type,
                setting.factor,
            )

    def get_scenario_id(self):
        """
        Get the scenario id.
        """
        return self.scenario_id

    def get_openml_id(self):
        """
        Get the openml id.
        """
        return self.openml_id

    def get_setting(self):
        """
        Get the setting.
        """
        return self.setting

    def get_seed(self):
        """
        Get the seed.
        """
        return self.seed

    def get_labeled_instances(self):
        """
        Get the labeled instances.
        """
        return self.labeled_indices

    def get_test_indices(self):
        """
        Get the test indices.
        """
        return self.test_indices

    def get_labeled_train_data(self):
        """
        Get the labeled training data.
        """
        return self.X[self.labeled_indices], self.y[self.labeled_indices]

    def get_unlabeled_train_data(self):
        """
        Get the unlabeled training data (X and y).
        """
        combined_train_labeled_test = self.labeled_indices + self.test_indices
        mask = np.array([True] * len(self.X))
        mask[combined_train_labeled_test] = False
        return self.X[mask], self.y[mask]

    def get_test_data(self):
        """
        Get the test data.
        """
        return self.X[self.test_indices], self.y[self.test_indices]

    def get_data_split(self):
        """
        Get labeled, unlabeled and test data.
        """
        X_l, y_l = self.get_labeled_train_data()
        X_u, y_u = self.get_unlabeled_train_data()
        X_test, y_test = self.get_test_data()
        return X_l, y_l, X_u, y_u, X_test, y_test

    def __repr__(self):
        params = dict(self.__dict__)
        params.pop("X")
        params.pop("y")
        return "<ActiveLearningScenario> " + str(params)
