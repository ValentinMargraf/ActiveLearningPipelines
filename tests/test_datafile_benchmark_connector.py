import pytest
from sklearn.ensemble import RandomForestClassifier

from alpbench.benchmark.BenchmarkConnector import DataFileBenchmarkConnector
from alpbench.evaluation.experimenter.DefaultSetup import ensure_default_setup
from alpbench.pipeline.QueryStrategy import RandomQueryStrategy

base_folder = "temp/"
learner_file = base_folder + "learner.json"
query_strategy_file = base_folder + "query_strategy.json"
scenario_file = base_folder + "scenario.json"
setting_file = base_folder + "setting.json"


@pytest.fixture
@pytest.mark.usefixtures("scenario")
def benchmark_connector(scenario):
    bc = DataFileBenchmarkConnector(
        learner_file=learner_file,
        query_strategy_file=query_strategy_file,
        scenario_file=scenario_file,
        setting_file=setting_file,
    )
    ensure_default_setup(bc)
    bc.load_or_create_scenario(
        scenario.openml_id,
        test_split_seed=scenario.test_split_seed,
        train_split_seed=scenario.train_split_seed,
        seed=scenario.seed,
        setting_id=scenario.setting.setting_id,
    )
    bc.dump()
    return DataFileBenchmarkConnector(
        learner_file=learner_file,
        query_strategy_file=query_strategy_file,
        scenario_file=scenario_file,
        setting_file=setting_file,
    )


def test_load_learner(benchmark_connector):
    learner = benchmark_connector.load_learner(1)
    assert learner is not None


def test_load_learner_by_name(benchmark_connector):
    learner = benchmark_connector.load_learner_by_name("svm_lin")
    assert learner is not None


def test_load_query_strategy(benchmark_connector):
    query_strategy = benchmark_connector.load_query_strategy(1)
    assert query_strategy is not None


def test_load_query_strategy_by_name(benchmark_connector):
    query_strategy = benchmark_connector.load_query_strategy_by_name("random")
    assert query_strategy is not None


def test_load_setting(benchmark_connector):
    setting = benchmark_connector.load_setting(1)
    assert setting is not None


def test_load_setting_by_name(benchmark_connector):
    setting = benchmark_connector.load_setting_by_name("small")
    assert setting is not None


def test_load_scenario(benchmark_connector):
    scenario = benchmark_connector.load_scenario(1)
    assert scenario is not None


def test_fail_load_nonexistent_setting(benchmark_connector):
    with pytest.raises(BaseException):
        benchmark_connector.load_setting(-1)


def test_fail_load_nonexistent_scenario(benchmark_connector):
    with pytest.raises(BaseException):
        benchmark_connector.load_scenario(-1)


def test_fail_load_nonexistent_setting_by_name(benchmark_connector):
    with pytest.raises(BaseException):
        benchmark_connector.load_setting_by_name("foe")


def test_create_new_setting(benchmark_connector):
    setting = benchmark_connector.load_or_create_setting("test", 0.5, "absolute", 0.3, 2, 2, 3)
    assert setting.get_setting_id() == 6


def test_fail_load_nonexistent_learner(benchmark_connector):
    with pytest.raises(BaseException):
        benchmark_connector.load_learner_by_name("foe")


def test_fail_load_nonexistent_learner_by_id(benchmark_connector):
    with pytest.raises(BaseException):
        benchmark_connector.load_learner(-1)


def test_load_new_learner(benchmark_connector):
    benchmark_connector.load_or_create_learner("foe", RandomForestClassifier(n_estimators=42))


def test_fail_load_nonexistent_query_strategy_by_name(benchmark_connector):
    with pytest.raises(BaseException):
        benchmark_connector.load_query_strategy_by_name("foe")


def test_fail_load_nonexistent_query_strategy(benchmark_connector):
    with pytest.raises(BaseException):
        benchmark_connector.load_query_strategy(-1)


def test_load_new_query_strategy(benchmark_connector):
    benchmark_connector.load_or_create_query_strategy("foe", RandomQueryStrategy(seed=10))


def test_init_default_file_paths():
    dfbc = DataFileBenchmarkConnector()
    assert dfbc.learner_file != DataFileBenchmarkConnector.base_folder + DataFileBenchmarkConnector.learner_file
    dfbc.cleanup()
