import tensorflow as tf
from src.config.config import Config
from src.io_manager import IO
from src.utils import ImageRef


class NCI:
    def __init__(self, config: Config, io: IO):
        self._config = config
        self._io = io

    def run(self):
        print('Running NCI')
        # Initialize a tensorflow session

        # Create a tensorflow constant
        hello = tf.constant('Hello, TensorFlow!')

    def compute_nci(self, image: ImageRef):
        self._io.check_existence_on_local()

