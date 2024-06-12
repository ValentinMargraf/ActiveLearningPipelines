from abc import ABC, abstractmethod

import openml


class BenchmarkSuite(ABC):
    """Benchmark Suite

    Abstract class for benchmark suites. A benchmark suite is a collection of datasets from OpenML.

    Args:
        name (str): name of the benchmark suite

    Attributes:
        name (str): name of the benchmark suite
    """

    def __init__(self, name="TabZilla"):
        self.name = name

    @abstractmethod
    def get_openml_dataset_ids(self):
        """
        Abstract method for getting the OpenML dataset ids of the benchmark suite.
        """
        pass


class TabZillaBenchmarkSuite(BenchmarkSuite):
    """TabZilla Benchmark Suite

    This benchmark suite contains the dataset ids of the TabZilla benchmark suite
    (https://github.com/naszilla/tabzilla).

    Args:
        None

    Attributes:
        name (str): name of the benchmark suite
    """

    def __init__(self):
        super().__init__(name="TabZilla")

    def get_openml_dataset_ids(self):
        """
        Get the OpenML dataset ids of the TabZilla benchmark suite.

        Returns:
            list: list of OpenML dataset ids
        """
        return [
            11,
            14,
            22,
            25,
            29,
            31,
            46,
            51,
            54,
            151,
            334,
            470,
            846,
            934,
            1043,
            1067,
            1169,
            1459,
            1468,
            1486,
            1489,
            1493,
            1494,
            1567,
            4134,
            4538,
            23512,
            40536,
            40981,
            41027,
            41143,
            41147,
            41150,
            41159,
        ]


class OpenMLBenchmarkSuite(BenchmarkSuite):
    """OpenML Benchmark Suite

    This benchmark suite contains the dataset ids of the OpenML-CC18 benchmark suite
    (https://openml.github.io/openml-python/main/examples/20_basic/simple_suites_tutorial.html).

    Args:
        openml_benchmark_id (int): id of the OpenML benchmark suite

    Attributes:
        openml_benchmark_id (int): id of the OpenML benchmark suite
        name (str): name of the benchmark suite
        openml_dataset_ids (list): list of OpenML dataset ids
    """

    def __init__(self, openml_benchmark_id, name="OpenML-BenchmarkSuite"):
        super(OpenMLBenchmarkSuite, self).__init__(name=name + "-" + str(openml_benchmark_id))
        self.openml_benchmark_id = openml_benchmark_id

        benchmark = openml.study.get_suite(openml_benchmark_id)
        tasks = openml.tasks.list_tasks(task_id=benchmark.tasks)

        self.openml_dataset_ids = list()
        for k, t in tasks.items():
            self.openml_dataset_ids.append(t["did"])

    def get_openml_dataset_ids(self):
        """
        Get the OpenML dataset ids of the OpenML-CC18 benchmark suite.

        Returns:
            list: list of OpenML dataset ids
        """
        return self.openml_dataset_ids
