import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ALP.util.ensemble_constructor import Ensemble
from ALP.util.pytorch_tabnet.tab_model import TabNetClassifier
from ALP.util.tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier


@pytest.mark.usefixtures("benchmark_connector")
def test_init_ensemble(benchmark_connector):
    X, y = np.random.random((10, 10)), np.random.randint(0, 2, (10, 1))
    for l in benchmark_connector.learners:
        learner = benchmark_connector.load_learner_by_name(l["learner_name"])
        learner.fit(X, y)
        ens = Ensemble(learner, 10, 10)
        ens.fit(X, y)
        ens.predict(X)

