from sklearn.ensemble import RandomForestClassifier as RF

from ALP.benchmark.ActiveLearningScenario import ActiveLearningScenario
from ALP.benchmark.ActiveLearningSetting import ActiveLearningSetting
from ALP.pipeline.ActiveLearningPipeline import ActiveLearningPipeline
from ALP.pipeline.Oracle import Oracle
from ALP.pipeline.SamplingStrategy import (
    BatchBaldSampling,
    DiscriminativeSampling,
    EntropySampling,
    ExpectedAveragePrecision,
    LeastConfidentSampling,
    MarginSampling,
    MinMarginSampling,
    MonteCarloEERLogLoss,
    MonteCarloEERMisclassification,
    PowerMarginSampling,
    QueryByCommitteeEntropySampling,
    QueryByCommitteeKLSampling,
    RandomMarginSampling,
    RandomSamplingStrategy,
    TypicalClusterSampling,
    WeightedClusterSampling,
)

# import pytest


SCENARIO_ID = 1
OPENML_ID = 31
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


def test_pipeline_executions():
    print("load setting")
    alsetting = ActiveLearningSetting(setting_id=SETTING_ID, setting_name=SETTING_NAME,
                                      setting_labeled_train_size=SETTING_TRAIN_SIZE,
                                      setting_train_type=SETTING_TRAIN_TYPE, setting_test_size=SETTING_TEST_SIZE,
                                      number_of_iterations=NUMBER_OF_IT, number_of_samples=NUMBER_OF_SAMPLES)

    print("load scenario")
    alscenario = ActiveLearningScenario(scenario_id=SCENARIO_ID, openml_id=OPENML_ID, test_split_seed=TEST_SPLIT_SEED,
                                        train_split_seed=TRAIN_SPLIT_SEED, seed=SEED, labeled_indices=LABELED_INDICES,
                                        test_indices=TEST_INDICES, setting=alsetting)

    print("get data split")
    X_l, y_l, X_u, y_u, X_test, y_test = alscenario.get_data_split()

    print("defined sampling strategies")
    sampling_strategies = [RandomSamplingStrategy(42), EntropySampling(42), MarginSampling(42),
                           LeastConfidentSampling(42),
                           MonteCarloEERLogLoss(42),
                           MonteCarloEERMisclassification(42), DiscriminativeSampling(42),
                           QueryByCommitteeEntropySampling(42, 10), QueryByCommitteeKLSampling(42, 10),
                           BatchBaldSampling(42, 10), PowerMarginSampling(42), RandomMarginSampling(42),
                           MinMarginSampling(42), ExpectedAveragePrecision(42), TypicalClusterSampling(42),
                           WeightedClusterSampling(42)]
    sampling_strategies = [RandomSamplingStrategy(42)]

    print("setup learner")
    learner = RF(n_estimators=10)

    print("test sampling strategies")
    for sampling_strategy in sampling_strategies:
        print("current sampling strategy: ", sampling_strategy)
        ALP = ActiveLearningPipeline(learner=learner, sampling_strategy=sampling_strategy, init_budget=10,
                                     num_iterations=1, num_samples_per_iteration=10)

        oracle = Oracle(X_u, y_u)
        ALP.active_fit(X_l, y_l, X_u, oracle)