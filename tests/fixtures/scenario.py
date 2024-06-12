import pytest

from ALP.benchmark.ActiveLearningSetting import ActiveLearningSetting
from ALP.benchmark.ActiveLearningScenario import ActiveLearningScenario

SCENARIO_ID = 1
OPENML_ID = 31
LEN_DATASET = 1000
TEST_SPLIT_SEED = 42
TRAIN_SPLIT_SEED = 43
SEED = 44
LABELED_INDICES = [0, 1, 2, 3]
TEST_INDICES = [5, 6, 7, 8, 9]

SETTING_ID = 1
SETTING_NAME = "TestSetting"
SETTING_TRAIN_SIZE = 0.4
SETTING_TRAIN_TYPE = "proportion"
SETTING_TEST_SIZE = 0.3
NUMBER_OF_IT = 10
NUMBER_OF_QUERIES = 5
FACTOR = None


@pytest.fixture
def setting():
    return ActiveLearningSetting(
        setting_id=SETTING_ID,
        setting_name=SETTING_NAME,
        setting_labeled_train_size=SETTING_TRAIN_SIZE,
        setting_train_type=SETTING_TRAIN_TYPE,
        setting_test_size=SETTING_TEST_SIZE,
        number_of_iterations=NUMBER_OF_IT,
        number_of_queries=NUMBER_OF_QUERIES,
        factor=FACTOR
    )


@pytest.fixture
@pytest.mark.usefixtures("setting")
def scenario(setting):
    return ActiveLearningScenario(
        scenario_id=SCENARIO_ID,
        openml_id=OPENML_ID,
        test_split_seed=TEST_SPLIT_SEED,
        train_split_seed=TRAIN_SPLIT_SEED,
        seed=SEED,
        labeled_indices=LABELED_INDICES,
        test_indices=TEST_INDICES,
        setting=setting
    )
