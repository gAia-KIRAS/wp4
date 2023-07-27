from abc import ABC, abstractmethod

from src.config.config import Config
from src.io_manager.io_manager import IO


class Module:
    def __init__(self, config: Config, io: IO):
        self._config = config
        self._io = io

        self._records = self._io.get_records()
        self._time_limit = self._config.time_limit

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def run_on_server(self):
        pass
