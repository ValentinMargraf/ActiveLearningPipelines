from ALP.benchmark.BenchmarkSuite import OpenMLBenchmarkSuite


def test_openml_benchmark_suite_loader():
    dids = OpenMLBenchmarkSuite(openml_benchmark_id=99).get_openml_dataset_ids()
    assert len(dids) == 72
