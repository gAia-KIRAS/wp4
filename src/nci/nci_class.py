import tensorflow as tf

class NCI:
    def __init__(self, config):
        self._config = config

    def run(self):
        print('Running NCI')
        # Initialize a tensorflow session

        # Create a tensorflow constant
        hello = tf.constant('Hello, TensorFlow!')
