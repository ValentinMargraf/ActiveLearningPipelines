import numpy as np
import pytest

from alpbench.util.ensemble_constructor import Ensemble


@pytest.mark.usefixtures("benchmark_connector")
def test_init_ensemble(benchmark_connector):
    X, y = np.random.random((10, 10)), np.random.randint(0, 2, (10, 1))
    for bench_learner in benchmark_connector.learners:
        learner = benchmark_connector.load_learner_by_name(bench_learner["learner_name"])
        learner.fit(X, y)
        ens = Ensemble(learner, 10, 10)
        ens.fit(X, y)
        ens.predict(X)
