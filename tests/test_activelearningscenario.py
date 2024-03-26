import pytest

from ALP.benchmark.ActiveLearningScenario import ActiveLearningScenario
from ALP.benchmark.ActiveLearningSetting import ActiveLearningSetting

SCENARIO_ID = 1
OPENML_ID = 31
LEN_DATASET = 1000
TEST_SPLIT_SEED = 42
TRAIN_SPLIT_SEED = 43
SEED = 44
LABELED_INDICES = [0, 1, 2, 3]
TEST_INDICES = [5, 6, 7, 8, 9]


SETTING_ID = 1337
SETTING_NAME = "TestSetting"
SETTING_TRAIN_SIZE = 0.4
SETTING_TRAIN_TYPE = "proportion"
SETTING_TEST_SIZE = 0.3
NUMBER_OF_IT = 10
NUMBER_OF_SAMPLES = 5


@pytest.fixture
def scenario():
    alsetting = ActiveLearningSetting(setting_id=SETTING_ID, setting_name=SETTING_NAME,
                                      setting_labeled_train_size=SETTING_TRAIN_SIZE,
                                      setting_train_type=SETTING_TRAIN_TYPE, setting_test_size=SETTING_TEST_SIZE,
                                      number_of_iterations=NUMBER_OF_IT, number_of_samples=NUMBER_OF_SAMPLES)

    return ActiveLearningScenario(scenario_id=SCENARIO_ID, openml_id=OPENML_ID, test_split_seed=TEST_SPLIT_SEED,
                                  train_split_seed=TRAIN_SPLIT_SEED, seed=SEED, labeled_indices=LABELED_INDICES,
                                  test_indices=TEST_INDICES, setting=alsetting)


def test_get_setting_id(scenario):
    assert scenario.get_scenario_id() == SCENARIO_ID


def test_get_openml_id(scenario):
    assert scenario.get_openml_id() == OPENML_ID


def test_get_test_data(scenario):
    X, y = scenario.get_test_data()
    assert len(X) == len(TEST_INDICES)
    assert len(X) == len(TEST_INDICES)


def test_get_labeled_train_data(scenario):
    X, y = scenario.get_labeled_train_data()
    assert len(X) == len(LABELED_INDICES)
    assert len(X) == len(LABELED_INDICES)


def test_get_unlabeled_train_data(scenario):
    X, y = scenario.get_unlabeled_train_data()
    exp_len = LEN_DATASET - len(LABELED_INDICES) - len(TEST_INDICES)
    assert len(X) == exp_len
    assert len(y) == exp_len


def test_get_setting(scenario):
    setting = scenario.get_setting()
    assert str(type(setting)) == "<class 'ALP.benchmark.ActiveLearningSetting.ActiveLearningSetting'>"
    assert setting.get_setting_id() == SETTING_ID
