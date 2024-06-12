import pytest

from fixtures.scenario import SETTING_ID, SETTING_NAME, SETTING_TRAIN_SIZE, SETTING_TRAIN_TYPE, SETTING_TEST_SIZE, \
    NUMBER_OF_SAMPLES, NUMBER_OF_IT, FACTOR


@pytest.mark.usefixtures("setting")
def test_get_setting_id(setting):
    assert setting.get_setting_id() == SETTING_ID


@pytest.mark.usefixtures("setting")
def test_get_setting_name(setting):
    assert setting.get_setting_name() == SETTING_NAME


@pytest.mark.usefixtures("setting")
def test_get_setting_labeled_train_size(setting):
    assert setting.get_setting_labeled_train_size() == SETTING_TRAIN_SIZE


@pytest.mark.usefixtures("setting")
def test_get_setting_train_type(setting):
    assert setting.get_setting_train_type() == SETTING_TRAIN_TYPE


@pytest.mark.usefixtures("setting")
def test_get_setting_test_size(setting):
    assert setting.get_setting_test_size() == SETTING_TEST_SIZE


@pytest.mark.usefixtures("setting")
def test_get_number_of_iterations(setting):
    assert setting.get_number_of_iterations() == NUMBER_OF_IT


@pytest.mark.usefixtures("setting")
def test_get_number_of_samples(setting):
    assert setting.get_number_of_samples() == NUMBER_OF_SAMPLES


@pytest.mark.usefixtures("setting")
def test_get_factor(setting):
    assert setting.get_factor() == FACTOR


@pytest.mark.usefixtures("setting")
def test_repr(setting):
    assert str(setting).startswith("<ActiveLearningSetting>")
