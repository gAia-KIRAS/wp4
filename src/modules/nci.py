import warnings
from osgeo import gdal
import numpy as np
import pandas as pd
import torch
import time

from config.config import Config
from config.io_config import IOConfig
from io_manager.io_manager import IO
from modules.abstract_module import Module
from utils import ImageRef, TileRef, timestamp, RECORDS_FILE_COLUMNS, FakeTFTypeHints, rename_product, RAW_IMAGE_SIZES, \
    CROP_IMAGE_LIMITS

try:
    import tensorflow as tf
except ImportError:
    tf = FakeTFTypeHints()


class NCI(Module):
    """
    Class for the calculation of Neighborhood Correlation Images on consecutive Sentinel-2 images.

    Attributes:
        _config (Config): Config object with the configuration parameters
        _io (IO): IO object with the input/output parameters
        _n_size (int): size of the neighborhood
        _conv_lib (str): library to use in the convolution calculations ('torch' or 'tf')
    """

    def __init__(self, config: Config, io: IO):
        super().__init__(config, io)

        self._n_size = self._config.nci_conf['n_size']
        self._conv_lib = self._config.nci_conf['conv_lib']

    @property
    def n_size(self) -> int:
        return self._n_size

    @n_size.setter
    def n_size(self, value: int):
        self._n_size = value

    def run(self, on_the_server: bool = False) -> None:
        """
        Compute NCI and avg vs neigh for images of all TileRefs (year + product + tile)
        1. Create a list of all the images that still have to be intersected. List is filtered according to config.
        2. For each pair of consecutive images (if time_limit is not over):
            - Downloads the RAW images from the server if it's not available locally
            - Calculate the NCI between the two images and save it
            - Delete the first image locally
            - Uploads the NCI image to the server
            - Deletes both the NCI and first image locally
            - Add image to the records
        3. Save the records to a CSV file

        Args:
            on_the_server (bool, optional): If True, the computation is done on the server. Defaults to False.
        """
        images_df = self._io.filter_all_images(image_type='raw', filters=self._config.filters)

        # Get all images that still have no computed NCI
        nci_done = self._records.loc[
            (self._records['from'] == 'raw') & (self._records['to'] == 'nci'),
            ['year', 'tile', 'product', 'filename_from']
        ].rename(columns={'filename_from': 'filename'})
        images_df = images_df.merge(nci_done, how='left', on=['year', 'tile', 'product', 'filename'], indicator=True)
        to_compute = images_df.loc[images_df['_merge'] == 'left_only',
        ['year', 'tile', 'product', 'filename']].sort_values(by=['year', 'tile', 'product', 'filename'])
        image_refs = [ImageRef(row.filename, row.year, row.tile, row.product, type='raw')
                      for row in to_compute.itertuples()]

        print(f'Filters: {self._config.filters}')
        print(f'Computing NCI for {len(image_refs)} images.\n')
        print(f'Time limit: {self._time_limit} minutes.')
        print(f'{len(images_df) - len(image_refs)} images already have NCI computed.\n')

        start_timestamp, start_time = timestamp(), time.time()
        i = 0
        while i < len(image_refs) - 1 and time.time() - start_time < self._time_limit * 60:
            print(f' -- Processing image {i + 1} of {len(image_refs)} '
                  f'({round((i + 1) * 100 / len(image_refs), 2)}%). Time elapsed: {round((time.time() - start_time) / 60, 2)} minutes. --')
            image_1 = image_refs[i]
            image_2 = image_refs[i + 1]

            if not on_the_server:
                # Download images (if not already downloaded)
                self._io.download_file(image_1)
                self._io.download_file(image_2)

            # NCI computed locally
            nci_image = self.compute_and_save_nci(image_1, image_2, on_the_server=on_the_server)

            if not on_the_server:
                # Delete first image from local machine. Second image will be used for the next iteration
                self._io.delete_local_file(image_1)

                # Upload NCI to server
                self._io.upload_file(nci_image)

                # Delete image from local machine
                self._io.delete_local_file(nci_image)

            # Update records
            record = ['raw', 'nci', image_1.tile, image_1.year, image_1.product, timestamp(),
                      image_1.filename, nci_image.filename, 1]
            record_df = pd.DataFrame({k: [v] for (k, v) in zip(RECORDS_FILE_COLUMNS, record)})
            self._records = pd.concat([self._records, record_df], ignore_index=True)
            self._io.save_records(self._records)
            i += 1

        # Delete image_2 from local machine
        if i > 0 and not on_the_server:
            self._io.delete_local_file(image_2)

        print(f'\nEnd of process. Time elapsed: {round((time.time() - start_time) / 60, 2)} minutes.')
        print(f'Processed {i} images during the execution.')
        unsuccessful = len(self._records.loc[
                               (self._records['success'] == 0) & (self._records['from'] == 'crop') &
                               (self._records['to'] == 'nci') &
                               self._records['timestamp'].between(start_timestamp, timestamp())])

        print(f'Unsuccessful computations: {unsuccessful}')

    def build_and_check_paths(self, image_1: ImageRef, image_2: ImageRef, on_the_server: bool = False) -> (str, str):
        """
        Checks if the paths to the images exist on the local machine. If not, raises an error.

        Args:
            image_1: ImageRef object with the first image
            image_2: ImageRef object with the second image
            on_the_server (bool, optional): If True, the computation is done on the server. Defaults to False.

        Returns:
            path_1: string with the filepath to the first image
            path_2: string with the filepath to the second image
        """
        if on_the_server:
            dir_1 = f'{self._io.build_remote_dir_for_image(image_1)}'
            dir_2 = f'{self._io.build_remote_dir_for_image(image_2)}'
            filepath_1 = f'{dir_1}/{image_1.filename}'
            filepath_2 = f'{dir_2}/{image_2.filename}'
        else:
            dir_1 = f'{self._io.config.base_local_dir}/{image_1.rel_dir()}'
            dir_2 = f'{self._io.config.base_local_dir}/{image_2.rel_dir()}'
            filepath_1 = f'{dir_1}/{image_1.filename}'
            filepath_2 = f'{dir_2}/{image_2.filename}'

        self._io.check_existence_on_local(filepath_1, dir_name=False)
        self._io.check_existence_on_local(filepath_2, dir_name=False)

        return filepath_1, filepath_2

    def save_nci(self, nci: tf.Tensor | torch.Tensor, image: ImageRef, srs: str, on_the_server: bool = False) \
            -> ImageRef:
        """
        Saves the NCI locally. The NCI is saved according to the following rules:
        - saved as a .tif file. Therefore, when loaded, it is an image with 4 channels
        - the spatial reference system is the same as the one of the image
        - LZW compression is used
        - NCI for (image_1 -> image_2) will be saved in the directory of image_1
        - will have name: - 'nci{neigh_size}_{image_1.product}_{image_1.year}_{image_1.tile}_{image_1.date}.tif'

        Args:
            nci: tf or torch Tensor with the NCI to save. Shape: (4, image.height, image.width)
            image: ImageRef object for the first image in the pair
            srs: string with the spatial reference system of the image
            on_the_server (bool, optional): If True, the computation is done on the server. Defaults to False.

        Returns:
            image: ImageRef object with the new saved image
        """
        filename = f'nci{self._n_size}_{rename_product.get(image.product, image.product)}_' \
                   f'{image.year}_{image.tile}_{image.extract_date()}.tif'
        filename_aux = filename.replace('.tif', '_aux.tif')
        return_image_ref = ImageRef(filename, tile_ref=image.tile_ref, type='nci')

        # Build paths to save the NCI
        if on_the_server:
            save_dir = self._io.build_remote_dir_for_image(return_image_ref)
        else:
            save_dir = f'{self._io.config.base_local_dir}/nci/{image.tile_ref.to_subpath()}'

        self._io.check_existence_on_local(save_dir, dir_name=True)

        filepath = f'{save_dir}/{filename}'
        filepath_aux = f'{save_dir}/{filename_aux}'

        # Check if image already exists. If so, overwrite it
        try:
            self._io.check_existence_on_local(filepath, dir_name=False)
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

        return return_image_ref

    def compute_and_save_nci(self, image_1: ImageRef, image_2: ImageRef, on_the_server: bool = False) -> ImageRef:
        """
        Computes and saves the NCI between two images. See more details about the computation in compute_nci method.

        Args:
            image_1: ImageRef object with the first image
            image_2: ImageRef object with the second image
            on_the_server: If True, the module is being run on the server, otherwise locally

        Returns:
            image_ref: ImageRef object referencing the new saved NCI image
        """
        filepath_1, filepath_2 = self.build_and_check_paths(image_1, image_2, on_the_server=on_the_server)

        # Both are locally available, we can compute the NCI
        r_1 = gdal.Open(filepath_1).ReadAsArray()
        r_2 = gdal.Open(filepath_2).ReadAsArray()

        # Get SRS from image_1
        srs = gdal.Open(filepath_1).GetProjection()

        # Crop images
        r_1 = self.crop_image(r_1, image_1)
        r_2 = self.crop_image(r_2, image_2)

        # Compute NCI using the selected library
        if self._conv_lib == 'torch':
            nci_result = self.compute_nci_with_torch(r_1, r_2)
        elif self._conv_lib == 'tf':
            nci_result = self.compute_nci_with_tf(r_1, r_2)
        else:
            raise ValueError(f'Convolution library {self._conv_lib} not recognized')

        # Save the NCI
        new_image = self.save_nci(nci_result, image_1, srs, on_the_server=on_the_server)

        return new_image

    @staticmethod
    def crop_image(raster, image_ref: ImageRef):
        # Check dimensions
        assert raster.shape == RAW_IMAGE_SIZES.get(image_ref.tile), \
            f'Image {image_ref.filename} has wrong dimensions. Expected: {RAW_IMAGE_SIZES.get(image_ref.tile)}'

        # Get dimensions from utils map
        min_i, max_i, min_j, max_j = CROP_IMAGE_LIMITS.get(image_ref.tile)

        # Crop image
        raster = raster[min_i:max_i, min_j:max_j]

        return raster


    def compute_nci_with_torch(self, r_1: np.ndarray, r_2: np.ndarray) -> torch.Tensor:
        """
        Computes the NCI between two images using PyTorch. Definiton of the NCI:
        - https://linkinghub.elsevier.com/retrieve/pii/S0034425705002919
        We delete the intermediate torch.Tensors to free up memory as soon as they are not needed anymore.
        Additionally, saves the centered second image.

        Args:
            r_1: np.ndarray with the first image
            r_2: np.ndarray with the second image

        Returns:
            nci: torch .Tensor with the NCI between the two images. Shape: (4, image.height, image.width)
                 The four bands correspond to the correlation r, the intercept a, the slope b, and pix_vs_avg.
        """
        # Convert type np.uint16 to np.int32
        r_1 = r_1.astype(np.int32)
        r_2 = r_2.astype(np.int32)

        # Build tensors
        r_1 = torch.from_numpy(r_1)
        r_2 = torch.from_numpy(r_2)

        # Convert type torch.int16 to torch.float32
        r_1 = r_1.type(torch.float32)
        r_2 = r_2.type(torch.float32)

        # Compute image of means according to filter size, and center the image (substract mean)
        mean_1 = self.apply_convolutions_torch(r_1, self._n_size, filter_values=1 / (self._n_size ** 2))
        mean_2 = self.apply_convolutions_torch(r_2, self._n_size, filter_values=1 / (self._n_size ** 2))

        # Subtract mean to the images
        centered_1 = torch.sub(r_1, mean_1)
        del r_1
        centered_2 = torch.sub(r_2, mean_2)
        del r_2

        cov = self.apply_convolutions_torch(torch.mul(centered_1, centered_2), self._n_size,
                                            filter_values=1 / (self._n_size ** 2 - 1))

        std_1 = torch.sqrt(
            self.apply_convolutions_torch(torch.square(centered_1), self._n_size, filter_values=1 / (self._n_size ** 2 - 1)))
        std_2 = torch.sqrt(
            self.apply_convolutions_torch(torch.square(centered_2), self._n_size, filter_values=1 / (self._n_size ** 2 - 1)))
        del centered_1

        r = torch.div(cov, torch.mul(std_1, std_2))
        del std_2
        a = torch.div(cov, torch.square(std_1))
        del cov, std_1
        b = torch.sub(mean_2, torch.mul(a, mean_1))
        del mean_1, mean_2

        # Stack the four images in one tensor
        nci_result = torch.stack([r, a, b, centered_2], dim=0)
        del r, a, b, centered_2

        return nci_result

    def compute_nci_with_tf(self, r_1: np.ndarray, r_2: np.ndarray) -> tf.Tensor:
        """
        Computes the NCI between two images using Tensorflow. Definition of the NCI:
        - https://linkinghub.elsevier.com/retrieve/pii/S0034425705002919
        We delete the intermediate tf.Tensors to free up memory as soon as they are not needed anymore.
        Additionally, saves the centered second image.

        Args:
            r_1: np.ndarray with the first image
            r_2: np.ndarray with the second image

        Returns:
            nci: tf.Tensor with the NCI between the two images. Shape: (4, image.height, image.width)
                 The four bands correspond to the correlation r, the intercept a, the slope b, and pix_vs_avg.
        """
        assert not isinstance(tf, FakeTFTypeHints), \
            f'Tensorflow is not installed. Change the convolution library (nci/conv_lib parameter) to "torch".'

        r_1 = tf.convert_to_tensor(r_1, dtype=tf.float32, name='r_1')
        r_2 = tf.convert_to_tensor(r_2, dtype=tf.float32, name='r_2')

        # Compute image of means according to filter size, and center the image (substract mean)
        mean_1 = self.apply_convolutions_tf(r_1, self._n_size, filter_values=1 / (self._n_size ** 2))
        mean_2 = self.apply_convolutions_tf(r_2, self._n_size, filter_values=1 / (self._n_size ** 2))
        centered_1 = tf.subtract(r_1, mean_1)
        del r_1
        centered_2 = tf.subtract(r_2, mean_2)
        del r_2

        # Compute covariance and standard deviations
        cov = self.apply_convolutions_tf(tf.multiply(centered_1, centered_2), self._n_size,
                                         filter_values=1 / (self._n_size ** 2 - 1))
        std_1 = tf.sqrt(
            self.apply_convolutions_tf(tf.square(centered_1), self._n_size, filter_values=1 / (self._n_size ** 2 - 1)))
        std_2 = tf.sqrt(
            self.apply_convolutions_tf(tf.square(centered_2), self._n_size, filter_values=1 / (self._n_size ** 2 - 1)))
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
    def apply_convolutions_tf(raster: tf.Tensor, filter_size: int, filter_values=None) -> tf.Tensor:
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

    @staticmethod
    def apply_convolutions_torch(raster, filter_size, filter_values=None):
        filter = torch.ones((1, 1, filter_size, filter_size))
        if filter_values is not None:
            filter = filter.multiply(filter_values)

        # Apply the filter to the raster
        output = torch.nn.functional.conv2d(
            torch.reshape(raster, (1, 1, raster.size()[0], raster.size()[1])),
            filter,
            stride=1,
            padding='same'
        )
        return torch.squeeze(output)
