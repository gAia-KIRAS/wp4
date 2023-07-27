import warnings
import gdal
import numpy as np
import pandas as pd
import tensorflow as tf
import time

from src.config.config import Config
from src.config.io_config import IOConfig
from src.io.io_manager import IO
from src.modules.abstract_module import Module
from src.utils import ImageRef, TileRef, timestamp, RECORDS_FILE_COLUMNS


class NCI(Module):
    """
    Class for the calculation of Neighborhood Correlation Images on consecutive Sentinel-2 images.

    Attributes:
        _config (Config): Config object with the configuration parameters
        _io (IO): IO object with the input/output parameters
        _records (pd.DataFrame): records DataFrame, as defined in IO.get_records()
        _time_limit (int): time limit in minutes for run method
        _n_size (int): size of the neighborhood
    """

    def run_on_server(self):
        pass

    def __init__(self, config: Config, io: IO):
        super().__init__(config, io)

        self._n_size = self._config.nci_conf['n_size']

    @property
    def n_size(self) -> int:
        return self._n_size

    @n_size.setter
    def n_size(self, value: int):
        self._n_size = value

    def run(self):
        images_df = self._io.filter_all_images(image_type='crop', filters=self._config.filters)

        # Get all images that still have no computed NCI
        nci_done = self._records.loc[
            (self._records['from'] == 'crop') & (self._records['to'] == 'nci'),
            ['year', 'tile', 'product', 'filename_from']
        ].rename(columns={'filename_from': 'filename'})
        images_df = images_df.merge(nci_done, how='left', on=['year', 'tile', 'product', 'filename'], indicator=True)
        to_compute = images_df.loc[images_df['_merge'] == 'left_only',
        ['year', 'tile', 'product', 'filename']].sort_values(by=['year', 'tile', 'product', 'filename'])
        image_refs = [ImageRef(row.filename, row.year, row.tile, row.product, type='crop')
                      for row in to_compute.itertuples()]

        print(f'Filters: {self._config.filters}')
        print(f'Intersecting {len(image_refs)} images with the AOI.\n')
        print(f'Time limit: {self._time_limit} minutes.')
        print(f'{len(images_df) - len(image_refs)} images have already been intersected.\n')

        start_timestamp, start_time = timestamp(), time.time()
        i = 0
        while i < len(image_refs) - 1 and time.time() - start_time < self._time_limit * 60:
            print(f' -- Processing image {i + 1} of {len(image_refs)} '
                  f'({round((i + 1) * 100 / len(image_refs), 2)}%). Time elapsed: {round((time.time() - start_time) / 60, 2)} minutes. --')
            image_1 = image_refs[i]
            image_2 = image_refs[i + 1]

            # Download images (if not already downloaded)
            self._io.download_file(image_1)
            self._io.download_file(image_2)

            # NCI computed locally
            nci_image = self.compute_and_save_nci(image_1, image_2)

            # Delete first image from local machine. Second image will be used for the next iteration
            self._io.delete_local_file(image_1)

            # Upload NCI to server
            self._io.upload_file(nci_image)

            # Delete image from local machine
            self._io.delete_local_file(nci_image)

            # Update records
            record = ['crop', 'nci', image_1.tile, image_1.year, image_1.product, timestamp(),
                      image_1.filename, nci_image.filename, 1]
            record_df = pd.DataFrame({k: [v] for (k, v) in zip(RECORDS_FILE_COLUMNS, record)})
            self._records = pd.concat([self._records, record_df], ignore_index=True)
            self._io.save_records(self._records)
            i += 1

        # Delete image_2 from local machine
        if i > 0:
            self._io.delete_local_file(image_2)

        print(f'\nEnd of process. Time elapsed: {round((time.time() - start_time) / 60, 2)} minutes.')
        print(f'Processed {i} images during the execution.')
        unsuccessful = len(self._records.loc[
                               (self._records['success'] == 0) & (self._records['from'] == 'crop') &
                               (self._records['to'] == 'nci') &
                               self._records['timestamp'].between(start_timestamp, timestamp())])

        print(f'Unsuccessful computations: {unsuccessful}')

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

    def save_nci(self, nci: tf.Tensor, image: ImageRef, srs: str) -> ImageRef:
        """
        Saves the NCI locally. The NCI is saved according to the following rules:
        - saved as a .tif file. Therefore, when loaded, it is an image with 3 channels
        - the spatial reference system is the same as the one of the image
        - LZW compression is used
        - NCI for (image_1 -> image_2) will be saved in the directory of image_1
        - will have name: - 'nci{neigh_size}_{image_1.product}_{image_1.year}_{image_1.tile}_{image_1.date}.tif'

        Args:
            nci: tf.Tensor with the NCI to save. Shape: (image.height, image.width, 4)
            image: ImageRef object for the first image in the pair
            srs: string with the spatial reference system of the image

        Returns:
            image: ImageRef object with the new saved image
        """
        # Check if the directory exists. If not, the IO method already creates it
        dir = f'{self._io.config.base_local_dir}/nci/{image.tile_ref.to_subpath()}'
        self._io.check_existence_on_local(dir, dir=True)

        filename_aux = f"nci{self._n_size}_{image.product}_{image.year}_{image.tile}_{image.extract_date()}_aux.tif"
        filepath_aux = f'{dir}/{filename_aux}'

        filename = f'nci{self._n_size}_{image.product}_{image.year}_{image.tile}_{image.extract_date()}.tif'
        filepath = f'{dir}/{filename}'

        # Check if image already exists. If so, overwrite it
        try:
            self._io.check_existence_on_local(filepath, dir=False)
            warnings.warn(f'\nNCI {filename} already exists. Overwriting it.')
        except FileNotFoundError:
            pass

        nci_numpy = nci.numpy()

        # Save an auxiliary .tif file
        driver = gdal.GetDriverByName('GTiff')
        n_bands, rows, cols = nci.shape
        dataset = driver.Create(filepath_aux, cols, rows, n_bands, gdal.GDT_Float32)
        dataset.SetProjection(srs)
        for i in range(n_bands):
            band = dataset.GetRasterBand(i + 1)
            if i == 0:
                band.SetNoDataValue(np.nan)
            band.WriteArray(nci_numpy[i])
        dataset.FlushCache()
        dataset = None

        # Use gdal translate to compress the image using LZW
        gdal.Translate(filepath, filepath_aux, options='-co COMPRESS=LZW')

        # Delete the auxiliary file
        self._io.delete_local_file(ImageRef(filename_aux, tile_ref=image.tile_ref, type='nci'))

        return ImageRef(filename, tile_ref=image.tile_ref, type='nci')

    def compute_and_save_nci(self, image_1: ImageRef, image_2: ImageRef) -> ImageRef:
        """
        Computes and saves the NCI between two images. See more details about the computation in compute_nci method.

        Args:
            image_1: ImageRef object with the first image
            image_2: ImageRef object with the second image

        Returns:
            image_ref: ImageRef object referencing the new saved NCI image
        """
        filepath_1, filepath_2 = self.build_and_check_paths(image_1, image_2)

        # Both are locally available, we can compute the NCI
        r_1 = gdal.Open(filepath_1).ReadAsArray()
        r_2 = gdal.Open(filepath_2).ReadAsArray()

        # Get SRS from image_1
        srs = gdal.Open(filepath_1).GetProjection()

        nci_result = self.compute_nci(r_1, r_2)
        new_image = self.save_nci(nci_result, image_1, srs)

        return new_image

    def compute_nci(self, r_1: np.ndarray, r_2: np.ndarray) -> tf.Tensor:
        """
        Computes the NCI between two images. Definition of the NCI:
        - https://linkinghub.elsevier.com/retrieve/pii/S0034425705002919
        We delete the intermediate tf.Tensors to free up memory as soon as they are not needed anymore.
        Additionally, saves the centered second image.

        Args:
            r_1: np.ndarray with the first image
            r_2: np.ndarray with the second image

        Returns:
            nci: tf.Tensor with the NCI between the two images. Shape: (4, image.height, image.width)
                 The three bands correspond to the correlation r, the intercept a, and the slope b.
        """
        r_1 = tf.convert_to_tensor(r_1, dtype=tf.float32, name='r_1')
        r_2 = tf.convert_to_tensor(r_2, dtype=tf.float32, name='r_2')

        # Compute image of means according to filter size, and center the image (substract mean)
        mean_1 = self.apply_convolutions(r_1, self._n_size, filter_values=1 / (self._n_size ** 2))
        mean_2 = self.apply_convolutions(r_2, self._n_size, filter_values=1 / (self._n_size ** 2))
        centered_1 = tf.subtract(r_1, mean_1)
        del r_1
        centered_2 = tf.subtract(r_2, mean_2)
        del r_2

        # Compute covariance and standard deviations
        cov = self.apply_convolutions(tf.multiply(centered_1, centered_2), self._n_size,
                                      filter_values=1 / (self._n_size ** 2 - 1))
        std_1 = tf.sqrt(
            self.apply_convolutions(tf.square(centered_1), self._n_size, filter_values=1 / (self._n_size ** 2 - 1)))
        std_2 = tf.sqrt(
            self.apply_convolutions(tf.square(centered_2), self._n_size, filter_values=1 / (self._n_size ** 2 - 1)))
        del centered_1

        # Compute NCI
        r = tf.divide(cov, tf.multiply(std_1, std_2))
        del std_2
        a = tf.divide(cov, tf.square(std_1))
        del cov, std_1
        b = tf.subtract(mean_2, tf.multiply(a, mean_1))
        del mean_1, mean_2

        # Stack the four images in one tensor
        nci_result = tf.stack([r, a, b, centered_2], axis=0)
        del r, a, b, centered_2

        return nci_result

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


if __name__ == '__main__':
    config = Config()
    io_config = IOConfig()
    io = IO(io_config)
    nci = NCI(config, io)

    tile_ref = TileRef(2020, '33TUM', 'NDVI_raw')
    image_refs, _ = io.list_files_on_server(tile_ref, image_type='crop')
    image_1 = image_refs[0]
    image_1.type = 'crop'
    image_2 = image_refs[1]
    image_2.type = 'crop'

    io.download_file(image_1)
    io.download_file(image_2)

    result = nci.compute_and_save_nci(image_1, image_2)
