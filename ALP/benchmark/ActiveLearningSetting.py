class ActiveLearningSetting:
    """Active Learning Setting

    The active learning setting defines constraints and design choices of one active learning setup. This involves the
    size of the labeled training data, the size of the test data, the number of iterations, the number of samples
    queried per iteration, and a task-dependent factor in case a dynamic setting is considered (i.e. number of samples
    queried depends on the number of classes of the given dataset).

    Args:
        setting_id (int): id of the setting in the database
        setting_name (str): descriptor of the setting
        setting_labeled_train_size (float): size of the labeled training size
        setting_train_type (str): type of the size parameter: number of data points or share of the (training) dataset
        setting_test_size (float): size of the test data (always given as a share of the full dataset)
        number_of_iterations (int): number of iterations
        number_of_samples (int): number of samples queried per iteration
        factor (int): task-dependent factor

    Attributes:
        setting_id (int): id of the setting in the database
        setting_name (str): descriptor of the setting
        setting_labeled_train_size (float): size of the labeled training size
        setting_train_type (str): type of the size parameter: number of data points or share of the (training) dataset
        setting_test_size (float): size of the test data (always given as a share of the full dataset)
        number_of_iterations (int): number of iterations
        number_of_samples (int): number of samples queried per iteration
        factor (int): task-dependent factor
    """

    def __init__(
        self,
        setting_id,
        setting_name,
        setting_labeled_train_size,
        setting_train_type,
        setting_test_size,
        number_of_iterations,
        number_of_samples,
        factor,
    ):
        # id of the setting in the database
        self.setting_id = setting_id
        # descriptor of the setting
        self.setting_name = setting_name
        # size of the labeled training size
        self.setting_labeled_train_size = float(setting_labeled_train_size)
        # type of the size parameter: number of data points or share of the (training) dataset
        self.setting_train_type = setting_train_type
        # size of the test data (always given as a share of the full dataset)
        self.setting_test_size = float(setting_test_size)
        # number of iterations
        self.number_of_iterations = number_of_iterations
        # number of samples queried per iteration
        self.number_of_samples = number_of_samples
        # task-dependent factor
        self.factor = factor

    def from_dict(setting: dict):
        """
        Create an ActiveLearningSetting object from a dictionary.

        Args:
            setting (dict): dictionary containing the setting information

        Returns:
            ActiveLearningSetting: active learning setting object
        """
        return ActiveLearningSetting(
            setting_id=setting["setting_id"],
            setting_name=setting["setting_name"],
            setting_labeled_train_size=setting["setting_labeled_train_size"],
            setting_train_type=setting["setting_train_type"],
            setting_test_size=setting["setting_test_size"],
            number_of_iterations=setting["number_of_iterations"],
            number_of_samples=setting["number_of_samples"],
            factor=setting["factor"],
        )

    def get_setting_id(self):
        """
        Get the setting id.
        """
        return self.setting_id

    def get_setting_name(self):
        """
        Get the setting name.
        """
        return self.setting_name

    def get_factor(self):
        """
        Get the factor.
        """
        return self.factor

    def get_setting_labeled_train_size(self):
        """
        Get the size of the labeled training data.
        """
        return self.setting_labeled_train_size

    def get_setting_train_type(self):
        """
        Get the training type, absolute or relative.
        """
        return self.setting_train_type

    def get_setting_test_size(self):
        """
        Get the size of the test data.
        """
        return self.setting_test_size

    def get_number_of_iterations(self):
        """
        Get the number of iterations.
        """
        return self.number_of_iterations

    def get_number_of_samples(self):
        """
        Get the number of samples queried per iteration.
        """
        return self.number_of_samples

    def __repr__(self):
        return "<ActiveLearningSetting> " + str(self.__dict__)
