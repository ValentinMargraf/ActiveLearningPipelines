import pytest

from alpbench.benchmark.BenchmarkConnector import DataFileBenchmarkConnector
from alpbench.evaluation.experimenter.DefaultSetup import ensure_default_setup

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
    setting = benchmark_connector.load_setting(1)
    assert setting is not None
