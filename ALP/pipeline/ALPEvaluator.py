from ALP.benchmark.BenchmarkConnector import BenchmarkConnector
from ALP.benchmark.Observer import Observer
from ALP.pipeline.ActiveLearningPipeline import ActiveLearningPipeline
from ALP.pipeline.Oracle import Oracle


class ALPEvaluator:
    """ALPEvaluator

    This class is used to evaluate an active learning pipeline (ALP) on a specific benchmark setting. This involves
    fitting the ALP on the training data and then evaluating it on the test data. All results can be reproduced
    by setting the test_split_seed, train_split_seed, and seed.

    Args:
        benchmark_connector (BenchmarkConnector): The benchmark connector to use.
        setting_name (str): The name of the setting.
        learner_name (str): The name of the learner.
        sampling_strategy_name (str): The name of the sampling strategy.
        openml_id (int): The openml id of the benchmark scenario.
        test_split_seed (int): The seed for splitting the test data.
        train_split_seed (int): The seed for splitting the train data.
        seed (int): The seed for the random operatioms.

    Attributes:
        benchmark_connector (BenchmarkConnector): The benchmark connector to use.
        observer_list (list): The list of observers to use.
        setting (ActiveLearningSetting): The setting to use.
        scenario (ActiveLearningScenario): The scenario to use.
        setting_name (str): The name of the setting.
        learner_name (str): The name of the learner.
        sampling_strategy_name (str): The name of the sampling strategy.
        openml_id (int): The openml id of the benchmark scenario.
        test_split_seed (int): The seed for splitting the test data.
        train_split_seed (int): The seed for splitting the train data.
        seed (int): The seed for the random operatioms.
    """

    def __init__(
        self,
        benchmark_connector: BenchmarkConnector,
        setting_name=None,
        learner_name=None,
        sampling_strategy_name=None,
        openml_id=None,
        test_split_seed=0,
        train_split_seed=0,
        seed=0,
    ):
        self.benchmark_connector = benchmark_connector
        self.observer_list = []
        self.setting = None
        self.scenario = None

        # setting specifications
        self.setting_name = setting_name

        # scenario specifications
        self.openml_id = openml_id
        self.test_split_seed = test_split_seed
        self.train_split_seed = train_split_seed
        self.seed = seed

        # pipeline specifications
        self.learner_name = learner_name
        self.sampling_strategy_name = sampling_strategy_name

    def _load_setting_and_scenario(self):
        """Load the setting and scenario from the database using the benchmark connector.

        Parameters:
            None

        Returns:
            None
        """
        assert self.openml_id is not None and self.setting_name is not None
        self.setting = self.benchmark_connector.load_setting_by_name(self.setting_name)
        self.scenario = self.benchmark_connector.load_or_create_scenario(
            openml_id=self.openml_id,
            test_split_seed=self.test_split_seed,
            train_split_seed=self.train_split_seed,
            seed=self.seed,
            setting_id=self.setting.get_setting_id(),
        )

    def fit(self):
        """Fits the active learning pipeline on the training data for the
        given amount of iterations and returns the fitted pipeline.

        Parameters::
            None
        Returns:
            ActiveLearningPipeline: The fitted active learning pipeline.
        """
        self._load_setting_and_scenario()

        X_l, y_l, X_u, y_u, X_test, y_test = self.scenario.get_data_split()
        learner = self.benchmark_connector.load_learner_by_name(self.learner_name)
        sampling_strategy = self.benchmark_connector.load_sampling_strategy_by_name(self.sampling_strategy_name)

        oracle = Oracle(X_u=X_u, y_u=y_u)
        alp = ActiveLearningPipeline(
            learner=learner,
            sampling_strategy=sampling_strategy,
            num_iterations=self.setting.get_number_of_iterations(),
            observer_list=self.observer_list,
            num_samples_per_iteration=self.setting.get_number_of_samples(),
        )
        alp.active_fit(X_l=X_l, y_l=y_l, X_u=X_u, oracle=oracle)

        return alp

    def with_setting(self, setting_name: str):
        self.setting_name = setting_name
        return self

    def with_learner(self, learner_name: str):
        self.learner_name = learner_name
        return self

    def with_sampling_strategy(self, sampling_strategy_name: str):
        self.sampling_strategy_name = sampling_strategy_name
        return self

    def with_test_split_seed(self, test_split_seed: int):
        self.test_split_seed = test_split_seed
        return self

    def with_train_split_seed(self, train_split_seed: int):
        self.train_split_seed = train_split_seed
        return self

    def with_openml_id(self, openml_id):
        self.openml_id = openml_id
        return self

    def with_observer(self, observer: Observer):
        self.observer_list += [observer]
        return self

    def get_test_data(self):
        self._load_setting_and_scenario()
        return self.scenario.get_test_data()
