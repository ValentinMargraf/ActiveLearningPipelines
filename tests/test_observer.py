import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ALP.benchmark.Observer import PrintObserver


def test_labeling_statistics():
    X, y = np.random.random((10, 10)), np.random.randint(0, 2, (10, 1))
    po = PrintObserver(X, y)
    po.observe_data(1, X, y, X, y, X)

    po.observe_data(1, np.empty((10, 0)), np.empty((1, 0)), X, y, X)


def test_model_stats():
    X, y = np.random.random((10, 10)), np.random.randint(0, 2, (10, 1))
    po = PrintObserver(X, y)
    rf = RandomForestClassifier()
    rf.fit(X, y)
    po.observe_model(1, rf)
