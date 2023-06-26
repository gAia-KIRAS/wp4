import warnings

from src.config.config import Config
from src.io_manager import IO
from src.utils import ImageRef
import tensorflow as tf
import rasterio as rio


class NCI:
    """
    Class for the calculation of Neighborhood Correlation Images on consecutive Sentinel-2 images.

    Attributes:
        _config (Config): Config object with the configuration parameters
        _io (IO): IO object with the input/output parameters
        _n_size (int): size of the neighborhood
    """
    def __init__(self, config: Config, io: IO):
        self._config = config
        self._io = io

        self._n_size = self._config.nci_conf['n_size']

    @property
    def n_size(self) -> int:
        return self._n_size

    @n_size.setter
    def n_size(self, value: int):
        self._n_size = value

    def run(self):
        print('Running NCI')
        # Initialize a tensorflow session

        # Create a tensorflow constant
        hello = tf.constant('Hello, TensorFlow!')

    def build_and_check_paths(self, image_1: ImageRef, image_2: ImageRef) -> (str, str):
        """
        Checks if the paths to the images exist on the local machine. If not, raises an error.

        Args:
            image_1: ImageRef object with the first image
            image_2: ImageRef object with the second image

        Returns:
            path_1: string with the filepath to the first image
            path_2: string with the filepath to the second image
        """
        self._io.check_inputs_with_metadata(image_1.tile_ref)
        self._io.check_inputs_with_metadata(image_2.tile_ref)

        dir_1 = f'{self._io.config.base_local_dir}/{image_1.rel_dir()}'
        dir_2 = f'{self._io.config.base_local_dir}/{image_2.rel_dir()}'
        self._io.check_existence_on_local(dir_1, dir=True)
        self._io.check_existence_on_local(dir_2, dir=True)

        filepath_1 = f'{dir_1}/{image_1.filename}'
        filepath_2 = f'{dir_2}/{image_2.filename}'
        self._io.check_existence_on_local(filepath_1, dir=False)
        self._io.check_existence_on_local(filepath_2, dir=False)

        return filepath_1, filepath_2

    def save_nci(self, nci: tf.Tensor, image: ImageRef):
        """
        Saves the NCI to the server. The NCI is saved according to the following rules:
        - saved using tf.saved_model.save
        - will be loaded using tf.saved_model.load
        - for image_1 -> image_2 will be saved in the directory of image_1
        - will have name: 'nci' + image_1.year + image_1.tile + image_1.date
        - will be placed in directory image_1.year + image_1.tile + 'nci'

        Args:
            nci: tf.Tensor with the NCI to save
            image: ImageRef object for the first image in the pair
        """
        # Check if the directory exists. If not, the IO method already creates it
        dir = f'{self._io.config.base_server_dir}/{image.tile_ref.to_subpath()}'
        self._io.check_existence_on_server(dir, dir=True)

        # Save the NCI to the local machine first
        new_image = ImageRef(f'nci_{image.year}_{image.tile}_{image.product}',
                             year=image.year, tile=image.tile, product='nci')
        local_dir = f'{self._io.config.base_local_dir}/{new_image.rel_dir()}'
        self._io.check_existence_on_local(local_dir, dir=True)
        filepath = f'{local_dir}/{new_image.filename}'
        try:
            self._io.check_existence_on_local(filepath, dir=False)
            warnings.warn(f'NCI {new_image.filename} already exists. Overwriting it.')
        except FileNotFoundError:
            pass
        tf.saved_model.save(nci, filepath)

        # Upload the NCI to the server
        self._io.upload_file(new_image)

        # Delete the NCI from the local machine
        self._io.delete_local_file(new_image)

    def compute_nci(self, image_1: ImageRef, image_2: ImageRef) -> (tf.Tensor, tf.Tensor, tf.Tensor):
        """
        Computes the NCI between two images. Definiton of the NCI:
        - https://linkinghub.elsevier.com/retrieve/pii/S0034425705002919
        We delete the intermediate tf.Tensors to free up memory as soon as they are not needed anymore.

        Args:
            image_1: ImageRef object with the first image
            image_2: ImageRef object with the second image

        Returns:
            nci: tf.Tensor with the final NCI. Contains three channels:
                - 1st channel: correlation r
                - 2nd channel: slope a
                - 3rd channel: intercept b
        """
        filepath_1, filepath_2 = self.build_and_check_paths(image_1, image_2)

        # Both are locally available, we can compute the NCI
        with rio.open(filepath_1) as reader:
            r_1 = tf.convert_to_tensor(reader.read(1), dtype=tf.float32, name='r_1')
        with rio.open(filepath_2) as reader:
            r_2 = tf.convert_to_tensor(reader.read(1), dtype=tf.float32, name='r_2')

        # Compute image of means according to filter size
        mean_1 = self.apply_convolutions(r_1, self._n_size, filter_values=1 / (self._n_size ** 2))
        mean_2 = self.apply_convolutions(r_2, self._n_size, filter_values=1 / (self._n_size ** 2))

        centered_1 = tf.subtract(r_1, mean_1)
        del r_1
        centered_2 = tf.subtract(r_2, mean_2)
        del r_2

        cov = self.apply_convolutions(tf.multiply(centered_1, centered_2), self._n_size, filter_values=1 / (self._n_size ** 2 - 1))

        std_1 = tf.sqrt(self.apply_convolutions(tf.square(centered_1), self._n_size, filter_values=1 / (self._n_size ** 2 - 1)))
        std_2 = tf.sqrt(self.apply_convolutions(tf.square(centered_2), self._n_size, filter_values=1 / (self._n_size ** 2 - 1)))

        r = tf.divide(cov, tf.multiply(std_1, std_2))
        del std_2

        a = tf.divide(cov, tf.square(std_1))
        del cov, std_1
        b = tf.subtract(mean_2, tf.multiply(a, mean_1))
        del mean_1, mean_2

        # Put the three images in one tensor
        nci = tf.stack([r, a, b], axis=2)
        del r, a, b

        return nci

    @staticmethod
    def apply_convolutions(raster, filter_size, filter_values=None):
        """
        Applies a 2D convolution to a raster. The filter is a square matrix of ones by default, and
        a matrix of values equal to filter_values if this is specified.

        Args:
            raster: tf.Tensor with the raster to apply the convolution to
            filter_size: int with the size of the filter. The filter is a square matrix of size filter_size x filter_size
            filter_values: (optional) float with the value to fill the filter with. If None, the filter is a matrix of ones

        Returns:
            output: tf.Tensor with the result of the convolution
        """
        filter = tf.ones((filter_size, filter_size, 1, 1), dtype=tf.float32)
        if filter_values is not None:
            filter = tf.scalar_mul(filter_values, filter)

        # Apply the filter to the raster
        output = tf.nn.conv2d(
            tf.reshape(raster, (1, raster.shape[0], raster.shape[1], 1)),
            filter,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )
        return tf.squeeze(output)
