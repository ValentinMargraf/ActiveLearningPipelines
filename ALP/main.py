import numpy as np

from ALP.benchmark.ActiveLearningScenario import ActiveLearningScenario
from ALP.benchmark.ActiveLearningSetting import ActiveLearningSetting

labeled_idx = np.arange(5, 10)
test_idx = np.arange(25, 50)

setting = ActiveLearningSetting(setting_id=1, setting_test_size=.2, setting_train_type="proportion",
                                setting_name="Test", setting_labeled_train_size=0.1, number_of_iterations=5,
                                number_of_samples=5)

scenario = ActiveLearningScenario(scenario_id=1, openml_id=31, train_split_seed=42, test_split_seed=43, seed=44,
                                  labeled_indices=labeled_idx, test_indices=test_idx, setting=setting)
