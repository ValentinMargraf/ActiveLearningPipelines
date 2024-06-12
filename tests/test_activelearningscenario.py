import pytest

from ALP.benchmark.ActiveLearningScenario import create_dataset_split
from fixtures.scenario import SCENARIO_ID, OPENML_ID, TEST_INDICES, LABELED_INDICES, LEN_DATASET, SETTING_ID


@pytest.mark.usefixtures("scenario")
def test_get_setting_id(scenario):
    assert scenario.get_scenario_id() == SCENARIO_ID


@pytest.mark.usefixtures("scenario")
def test_get_openml_id(scenario):
    assert scenario.get_openml_id() == OPENML_ID


@pytest.mark.usefixtures("scenario")
def test_get_test_data(scenario):
    X, y = scenario.get_test_data()
    assert len(X) == len(TEST_INDICES)
    assert len(X) == len(TEST_INDICES)


@pytest.mark.usefixtures("scenario")
def test_get_labeled_train_data(scenario):
    X, y = scenario.get_labeled_train_data()
    assert len(X) == len(LABELED_INDICES)
    assert len(X) == len(LABELED_INDICES)


@pytest.mark.usefixtures("scenario")
def test_get_unlabeled_train_data(scenario):
    X, y = scenario.get_unlabeled_train_data()
    exp_len = LEN_DATASET - len(LABELED_INDICES) - len(TEST_INDICES)
    assert len(X) == exp_len
    assert len(y) == exp_len


@pytest.mark.usefixtures("scenario")
def test_get_setting(scenario):
    setting = scenario.get_setting()
    assert str(type(setting)) == "<class 'ALP.benchmark.ActiveLearningSetting.ActiveLearningSetting'>"
    assert setting.get_setting_id() == SETTING_ID


@pytest.mark.usefixtures("scenario")
def test_create_dataset_split(scenario):
    labeled_indices, test_indices = create_dataset_split(scenario.X, scenario.y, scenario.test_split_seed,
                                                         scenario.setting.setting_test_size,
                                                         scenario.train_split_seed,
                                                         scenario.setting.setting_labeled_train_size,
                                                         scenario.setting.setting_train_type, scenario.setting.factor)
    assert len(labeled_indices) > 0
    assert len(test_indices) > 0
