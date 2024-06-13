from alpbench.benchmark.BenchmarkSuite import OpenMLBenchmarkSuite, TabZillaBenchmarkSuite


def test_openml_benchmark_suite_loader():
    dids = OpenMLBenchmarkSuite(openml_benchmark_id=99).get_openml_dataset_ids()
    assert len(dids) == 72


def test_tabzilla_benchmark_suite_loader():
    dids = TabZillaBenchmarkSuite().get_openml_dataset_ids()
    assert len(dids) == 34
