import os
import yaml


class Config:
    """
    The config class is responsible for loading the config.yaml file and providing the configuration to the other
    classes. The config.yaml file is located in the root directory of the project.

    Attributes:
        _root: string with the root directory of the project
        _config_path: string with the path to the configuration file
        _config: dictionary with the configuration
        _profiling_active: boolean indicating whether profiling is active
        _profiling_browser: boolean indicating whether the profiling results should be opened in the browser
        _execute_module: string with the module to be executed
        _time_limit: integer with the time limit for the execution of the module (minutes)
        _filters: dictionary with the filters to be applied to the inventory. Can have keys: ['year', 'tile', 'product']
        _execute_where: string with the location of the execution. local, server. Can also be: update_server, update_local
        _nci_conf: dictionary with the configuration for the NCI module
        _cd_conf: dictionary with the configuration for the CD module
        _eval_conf: dictionary with the configuration for the evaluation module

    """
    def __init__(self):
        # Use parent directory as root
        self._root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._config_path = os.path.join(self._root, 'config.yaml')
        self._config = self.load_config()

        # Profiling
        self._profiling_active = self._config['profiling']['active']
        self._profiling_browser = self._config['profiling']['browser']

        # Execution:
        self._execute_module = self._config['execute']['module']
        self._time_limit = self._config['execute']['time_limit']
        self._filters = self._config['execute']['filters']
        self._execute_where = self._config['execute']['where']

        # Module parameters
        self._nci_conf = self._config['nci']
        self._cd_conf = self._config['cd']
        self._eval_conf = self._config['eval']

    def load_config(self) -> dict:
        """
        Loads the config.yaml file and returns the content as a dictionary.
        Returns:
            dict: The content of the config.yaml file as a dictionary.
        """
        with open(self._config_path) as f:
            # use safe_load instead load
            config = yaml.safe_load(f)
        return config

    @property
    def profiling_active(self) -> bool:
        return self._profiling_active

    @property
    def profiling_browser(self) -> bool:
        return self._profiling_browser

    @property
    def nci_conf(self) -> dict:
        return self._nci_conf

    @property
    def cd_conf(self) -> dict:
        return self._cd_conf

    @property
    def eval_conf(self) -> dict:
        return self._eval_conf

    @property
    def execute(self) -> str:
        return self._execute_module

    @property
    def time_limit(self) -> int:
        return self._time_limit

    @property
    def filters(self) -> dict:
        return self._filters

    @property
    def execution_where(self) -> str:
        return self._execute_where
