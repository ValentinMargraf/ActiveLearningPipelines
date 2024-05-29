class ActiveLearningSetting:
    """
    The active learning setting defines constraints and design choices of one active learning setup.
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
        return self.setting_id

    def get_setting_name(self):
        return self.setting_name

    def get_setting_labeled_train_size(self):
        return self.setting_labeled_train_size

    def get_setting_train_type(self):
        return self.setting_train_type

    def get_setting_test_size(self):
        return self.setting_test_size

    def get_number_of_iterations(self):
        return self.number_of_iterations

    def get_number_of_samples(self):
        return self.number_of_samples

    def __repr__(self):
        return "<ActiveLearningSetting> " + str(self.__dict__)
