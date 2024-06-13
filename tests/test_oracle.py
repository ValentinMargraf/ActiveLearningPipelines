import numpy as np

from alpbench.pipeline.Oracle import Oracle


def test_oracle_setter():
    oracle = Oracle()

    X_u = np.random.random((10, 10))
    y_u = np.random.randint(0, 2, 10)
    oracle.set_data(X_u, y_u)

    assert (oracle.X_u == X_u).all()
    assert (oracle.y_u == y_u).all()
