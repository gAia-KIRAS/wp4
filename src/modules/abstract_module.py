from abc import abstractmethod

from config.config import Config
from io_manager import IO


class Module:
    def __init__(self, config: Config, io: IO):
        self._config = config
        self._io = io

        self._records = self._io.get_records()
        self._time_limit = self._config.time_limit

    @abstractmethod
    def run(self, on_the_server: bool = False) -> None:
        """
        Runs the module.

        Args:
            on_the_server (bool): if True, the module is being executed on the server.
        """
        pass
