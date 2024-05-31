from abc import ABC, abstractmethod

import openml


class BenchmarkSuite(ABC):
    """
    A BenchmarkSuite is a collection of openml dataset ids to be paired with different active learning problems.
    """

    def __init__(self, name="ALPBench-BenchmarkSuite"):
        self.name = name

    @abstractmethod
    def get_openml_dataset_ids(self):
        pass


class TabZillaBenchmarkSuite(BenchmarkSuite):
    """
    This is a benchmark suite consisting of the datasets that have been used in the TabZilla paper.
    """
    def __init__(self):
        super().__init__(name="TabZilla")

    def get_openml_dataset_ids(self):
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
            41159
        ]


class OpenMLBenchmarkSuite(BenchmarkSuite):
    """
    This benchmark suite allows for easy access to all datasets contained in an OpenML benchmark, which is identified
    via a openml_benchmark_id.
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
        return self.openml_dataset_ids


class OpenMLCC18BenchmarkSuite(OpenMLBenchmarkSuite):
    """
    This is a short-handle to access the OpenMLCC-18 benchmark suite from OpenML's team.
    """
    def __init__(self):
        super().__init__(99, "OpenMLCC-18")

