import numpy as np
import pytest
from pytorch_tabnet.tab_model import TabNetClassifier
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier

from alpbench.util.ensemble_constructor import Ensemble


def eval_ensemble_with_learner(learner):
    X, y = np.random.random((10, 10)), np.random.randint(0, 2, 10)
    learner.fit(X, y)
    ens = Ensemble(learner, 10, 10)
    ens.fit(X, y)
    ens.predict(X)


@pytest.mark.usefixtures("benchmark_connector")
def test_init_ensemble(benchmark_connector):
    for bench_learner in benchmark_connector.learners:
        learner = benchmark_connector.load_learner_by_name(bench_learner["learner_name"])
        eval_ensemble_with_learner(learner)


@pytest.mark.usefixtures("benchmark_connector")
def test_init_tabpfn_ensemble(benchmark_connector):
    learner = TabPFNClassifier()
    eval_ensemble_with_learner(learner)


@pytest.mark.usefixtures("benchmark_connector")
def test_init_tabpfn_ensemble_larger_dataset(benchmark_connector):
    learner = TabPFNClassifier()
    X, y = np.random.random((1200, 10)), np.random.randint(0, 2, 1200)
    learner.fit(X[0:999], y[0:999])
    ens = Ensemble(learner, 10, 10)
    ens.fit(X, y)
    ens.predict(X)


@pytest.mark.usefixtures("benchmark_connector")
def test_init_tabnet_ensemble(benchmark_connector):
    learner = TabNetClassifier(verbose=0)
    eval_ensemble_with_learner(learner)
