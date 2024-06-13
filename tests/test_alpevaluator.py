import pytest

from alpbench.benchmark.BenchmarkConnector import DataFileBenchmarkConnector
from alpbench.benchmark.Observer import PrintObserver
from alpbench.evaluation.experimenter.DefaultSetup import ensure_default_setup
from alpbench.pipeline.ALPEvaluator import ALPEvaluator

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


@pytest.fixture
@pytest.mark.usefixtures("benchmark_connector")
def alpevaluator(benchmark_connector):
    return ALPEvaluator(benchmark_connector)


def test_with_learner(alpevaluator):
    alpevaluator.with_learner("rf")
    assert alpevaluator.learner_name == "rf"


def test_with_query_strategy(alpevaluator):
    alpevaluator.with_query_strategy("random")
    assert alpevaluator.query_strategy_name == "random"


def test_with_setting(alpevaluator):
    alpevaluator.with_setting("small")
    assert alpevaluator.setting_name == "small"


def test_with_test_split_seed(alpevaluator):
    alpevaluator.with_test_split_seed(1337)
    assert alpevaluator.test_split_seed == 1337


def test_with_train_split_seed(alpevaluator):
    alpevaluator.with_train_split_seed(42)
    assert alpevaluator.train_split_seed == 42


def test_with_openml_id(alpevaluator):
    alpevaluator.with_openml_id(31)
    assert alpevaluator.openml_id == 31


def test_with_observer(alpevaluator):
    observer = PrintObserver(None, None)
    alpevaluator.with_observer(observer)
    assert observer in alpevaluator.observer_list


def test_get_test_data(alpevaluator):
    alpevaluator.with_setting("small")
    alpevaluator.with_openml_id(31)
    alpevaluator.get_test_data()


def test_fit(alpevaluator):
    alpevaluator.with_setting("small")
    alpevaluator.with_openml_id(31)
    alpevaluator.with_learner("rf_entropy")
    alpevaluator.with_query_strategy("random")
    alpevaluator.fit()
