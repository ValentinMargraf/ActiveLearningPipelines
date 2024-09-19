import numpy as np
from sklearn.ensemble import RandomForestClassifier

from alpbench.benchmark.Observer import PrintObserver
from alpbench.evaluation.experimenter.LogTableObserver import LogTableObserver, SparseLogTableObserver


class MockupResultProcessor:
    def __init__(self):
        pass

    def process_logs(self, data):
        pass


def test_labeling_statistics():
    X, y = np.random.random((10, 10)), np.random.randint(0, 2, (10, 1))
    po = PrintObserver(X, y)
    po.observe_data(1, X, y, X, y, X, np.arange(len(y)))

    po.observe_data(1, np.empty((10, 0)), np.empty((1, 0)), X, y, X, np.arange(len(y)))


def test_model_stats():
    X, y = np.random.random((10, 10)), np.random.randint(0, 2, (10, 1))
    po = PrintObserver(X, y)
    rf = RandomForestClassifier()
    rf.fit(X, y)
    po.observe_model(1, rf)


def test_logtableobserver():
    X, y = np.random.random((10, 10)), np.random.randint(0, 2, (10, 1))
    lto = LogTableObserver(MockupResultProcessor(), X, y)
    rf = RandomForestClassifier()
    rf.fit(X, y)
    lto.observe_data(1, X, y, X, y, X, np.arange(len(y)))
    lto.observe_model(1, rf)


def test_sparselogtableobserver():
    X, y = np.random.random((10, 10)), np.random.randint(0, 2, (10, 1))
    lto = SparseLogTableObserver(MockupResultProcessor(), X, y)
    rf = RandomForestClassifier()
    rf.fit(X, y)
    lto.observe_data(1, X, y, X, y, X, np.arange(len(y)))
    lto.observe_model(1, rf)
    lto.log_data({})
    lto.log_model({})
