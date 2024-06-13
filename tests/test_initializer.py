import numpy as np

from alpbench.pipeline.Initializer import RandomInitializer

NUM_SAMPLES = 2


def test_random_initializer():
    rand_init = RandomInitializer(42)
    X_u = np.random.random((10, 10))
    sample = rand_init.sample(X_u=X_u, num_samples=NUM_SAMPLES)
    assert len(sample) == NUM_SAMPLES
