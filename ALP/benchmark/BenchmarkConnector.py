from abc import ABC, abstractmethod


class BenchmarkConnector(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def load_scenario(self, scenario_id):
        pass

    @abstractmethod
    def load_or_create_scenario(self, openml_id, seed, setting_id):
        pass
