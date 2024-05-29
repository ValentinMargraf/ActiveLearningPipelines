import pytest

from ALP.benchmark.ActiveLearningSetting import ActiveLearningSetting

SETTING_ID = 42
SETTING_NAME = "TestSetting"
SETTING_TRAIN_SIZE = 0.4
SETTING_TRAIN_TYPE = "proportion"
SETTING_TEST_SIZE = 0.3
NUMBER_OF_IT = 10
NUMBER_OF_SAMPLES = 5


@pytest.fixture
def setting():
    return ActiveLearningSetting(
        setting_id=SETTING_ID,
        setting_name=SETTING_NAME,
        setting_labeled_train_size=SETTING_TRAIN_SIZE,
        setting_train_type=SETTING_TRAIN_TYPE,
        setting_test_size=SETTING_TEST_SIZE,
        number_of_iterations=NUMBER_OF_IT,
        number_of_samples=NUMBER_OF_SAMPLES,
    )


def test_get_setting_id(setting):
    assert setting.get_setting_id() == SETTING_ID


def test_get_setting_name(setting):
    assert setting.get_setting_name() == SETTING_NAME


def test_get_setting_labeled_train_size(setting):
    assert setting.get_setting_labeled_train_size() == SETTING_TRAIN_SIZE


def test_get_setting_train_type(setting):
    assert setting.get_setting_train_type() == SETTING_TRAIN_TYPE


def test_get_setting_test_size(setting):
    assert setting.get_setting_test_size() == SETTING_TEST_SIZE


def test_get_number_of_iterations(setting):
    assert setting.get_number_of_iterations() == NUMBER_OF_IT


def test_get_number_of_samples(setting):
    assert setting.get_number_of_samples() == NUMBER_OF_SAMPLES
