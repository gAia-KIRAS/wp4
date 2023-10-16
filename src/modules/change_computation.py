import time
import warnings

import pandas as pd
from osgeo import gdal
import ruptures as rpt
from config.config import Config
from io_manager.io_manager import IO
from config.io_config import IOConfig
from modules.abstract_module import Module
from utils import ImageRef, reference_nci_images, timestamp, RECORDS_FILE_COLUMNS, RAW_IMAGE_SIZES, \
    CROP_IMAGE_LIMITS, rename_product
import numpy as np


class ChangeComputation(Module):
    """
    Change Computation module. To replace the Change Detection module.
    Does the step nci -> delta
    TODO: documentation
    """

    def __init__(self, config: Config, io: IO):
        super().__init__(config, io)

        self._cd_records = self._io.get_records_cd()
        self._cd_results = self._io.get_results_cd()
        self._all_nci = self._io.list_all_files_of_type('nci')

        self._on_the_server = None
        self._cd_id = None
        self._pelt_penalty = None

    def run(self, on_the_server: bool = False) -> None:

        # Update parameters
        self._on_the_server = on_the_server
        self._cd_id = self._config.cd_conf['cd_id']
        self._pelt_penalty = self._config.cd_conf['penalty']

        # Check filters
        assert not (set(self._config.filters['product']) - {'NDVI_reconstructed'}), \
            'CD can only be applied to NDVI_reconstructed'

        images_df = self._io.filter_all_images(image_type='nci', filters=self._config.filters)
        raw_images_df = self._io.filter_all_images(image_type='raw', filters=self._config.filters)

        # Get all images that still have no computed CD
        nci_done = self._records.loc[
            (self._records['from'] == 'nci') & (self._records['to'] == 'delta'),
            ['year', 'tile', 'product', 'filename_from']
        ].rename(columns={'filename_from': 'filename'})
        images_df = images_df.merge(nci_done, how='left', on=['year', 'tile', 'product', 'filename'], indicator=True)
        to_compute = images_df.loc[images_df['_merge'] == 'left_only',
        ['year', 'tile', 'product', 'filename']].sort_values(by=['year', 'tile', 'product', 'filename'])
        image_refs = [ImageRef(row.filename, row.year, row.tile, row.product, type='nci')
                      for row in to_compute.itertuples()]

        print(f'Filters: {self._config.filters}')
        print(f'Computing Delta for {len(image_refs)} images.\n')
        print(f'Time limit: {self._time_limit} minutes.')
        print(f'{len(images_df) - len(image_refs)} Delta have already been computed.\n')

        start_timestamp, start_time = timestamp(), time.time()
        i = 0
        while i < len(image_refs) - 1 and time.time() - start_time < self._time_limit * 60:
            print(f' -- Processing image {i + 1} of {len(image_refs)} '
                  f'({round((i + 1) * 100 / len(image_refs), 2)}%). Time elapsed: {round((time.time() - start_time) / 60, 2)} minutes. --')
            image_1 = image_refs[i]
            image_2 = image_refs[i + 1]

            # Define corresponding cropped images
            raw_image_1 = raw_images_df.loc[
                (raw_images_df['year'] == image_1.year) &
                (raw_images_df['tile'] == image_1.tile) &
                (raw_images_df['product'] == image_1.product) &
                (raw_images_df['filename'].str.contains(image_1.extract_date()))
                ].iloc[0]
            raw_image_2 = raw_images_df.loc[
                (raw_images_df['year'] == image_2.year) &
                (raw_images_df['tile'] == image_2.tile) &
                (raw_images_df['product'] == image_2.product) &
                (raw_images_df['filename'].str.contains(image_2.extract_date()))
                ].iloc[0]
            raw_image_1 = ImageRef(raw_image_1.filename, raw_image_1.year, raw_image_1.tile, raw_image_1['product'],
                                   type='raw')
            raw_image_2 = ImageRef(raw_image_2.filename, raw_image_2.year, raw_image_2.tile, raw_image_2['product'],
                                   type='raw')

            if not on_the_server:
                # Download images (if not already downloaded)
                self._io.download_file(image_1)
                self._io.download_file(image_2)

                self._io.download_file(raw_image_1)
                self._io.download_file(raw_image_2)

            # Compute delta
            delta_image = self.compute_and_save_delta(image_1, image_2, raw_image_1, raw_image_2)

            if not on_the_server:
                pass
                # Delete first image from local machine. Second image will be used for the next iteration
                self._io.delete_local_file(image_1)
                self._io.delete_local_file(raw_image_1)

                # Upload NCI to server
                self._io.upload_file(delta_image)

                # Delete image from local machine
                self._io.delete_local_file(delta_image)

            # Update records
            record = ['nci', 'delta', image_1.tile, image_1.year, image_1.product, timestamp(),
                      image_1.filename, delta_image.filename, 1]
            record_df = pd.DataFrame({k: [v] for (k, v) in zip(RECORDS_FILE_COLUMNS, record)})
            self._records = pd.concat([self._records, record_df], ignore_index=True)
            self._io.save_records(self._records)
            i += 1

        # Delete image_2 from local machine
        if i > 0 and not on_the_server:
            self._io.delete_local_file(image_2)
            self._io.delete_local_file(raw_image_2)

        print(f'\nEnd of process. Time elapsed: {round((time.time() - start_time) / 60, 2)} minutes.')
        print(f'Processed {i} images during the execution.')
        unsuccessful = len(self._records.loc[
                               (self._records['success'] == 0) & (self._records['from'] == 'nci') &
                               (self._records['to'] == 'delta') &
                               self._records['timestamp'].between(start_timestamp, timestamp())])

        print(f'Unsuccessful computations: {unsuccessful}')

    def compute_and_save_delta(self, image_1: ImageRef, image_2: ImageRef, raw_image_1: ImageRef,
                               raw_image_2: ImageRef) -> ImageRef:
        filepath_1, filepath_2 = self.build_and_check_paths(image_1, image_2)
        filepath_raw_1, filepath_raw_2 = self.build_and_check_paths(raw_image_1, raw_image_2)

        # Read images
        nci_1 = gdal.Open(filepath_1).ReadAsArray()
        nci_2 = gdal.Open(filepath_2).ReadAsArray()

        raw_1 = gdal.Open(filepath_raw_1).ReadAsArray()
        raw_2 = gdal.Open(filepath_raw_2).ReadAsArray()

        srs = gdal.Open(filepath_raw_1).GetProjection()

        # Crop images
        r_1 = self.crop_image(raw_1, image_1)
        r_2 = self.crop_image(raw_2, image_2)
        del raw_1, raw_2

        # Assert images have the same dimension
        bands, rows, cols = nci_1.shape
        assert r_1.shape == (rows, cols), f'Image {image_1.filename} has wrong dimensions. Expected: {(rows, cols)}'
        bands, rows, cols = nci_2.shape
        assert r_2.shape == (rows, cols), f'Image {image_2.filename} has wrong dimensions. Expected: {(rows, cols)}'

        # Add one dimension to the raw images (it is just one band)
        r_1 = np.expand_dims(r_1, axis=0)
        r_2 = np.expand_dims(r_2, axis=0)

        # Append raw images as extra band to the NCI images
        # The five bands are: NCI.r, NCI.a, NCI.b, avg_vs_diff, raw
        nci_1 = np.concatenate((nci_1, r_1), axis=0)
        nci_2 = np.concatenate((nci_2, r_2), axis=0)

        # Compute delta band-wise. Delta is the relative difference between the two images:
        # delta = (nci_2 - nci_1) / nci_1
        delta = np.zeros((nci_1.shape[0], nci_1.shape[1], nci_1.shape[2]))
        for i in range(nci_1.shape[0]):
            delta[i] = np.divide(np.subtract(nci_2[i], nci_1[i]), nci_1[i])

        # Save delta
        delta_image = self._save_delta(delta, image_1, srs=srs)

        return delta_image

    def _save_delta(self, delta: np.ndarray, image_ref: ImageRef, srs) -> ImageRef:
        filename = f'delta_{rename_product.get(image_ref.product, image_ref.product)}_' \
                   f'{image_ref.year}_{image_ref.tile}_{image_ref.extract_date()}.tif'
        filename_aux = filename.replace('.tif', '_aux.tif')
        return_image_ref = ImageRef(filename, tile_ref=image_ref.tile_ref, type='delta')

        # Build paths
        if self._on_the_server:
            save_dir = f'{self._io.build_remote_dir_for_image(return_image_ref)}'
        else:
            save_dir = f'{self._io.config.base_local_dir}/{return_image_ref.rel_dir()}'

        self._io.check_existence_on_local(save_dir, dir=True)

        filepath = f'{save_dir}/{filename}'
        filepath_aux = f'{save_dir}/{filename_aux}'

        # Check if image already exists. If so, overwrite it
        try:
            self._io.check_existence_on_local(filepath, dir=False)
            warnings.warn(f'\nNCI {filename} already exists. Overwriting it.')
        except FileNotFoundError:
            pass

        # Save an auxiliary .tif file
        driver = gdal.GetDriverByName('GTiff')
        n_bands, rows, cols = delta.shape
        dataset = driver.Create(filepath_aux, cols, rows, n_bands, gdal.GDT_Float32)
        for i in range(n_bands):
            band = dataset.GetRasterBand(i + 1)
            if i == 0:
                band.SetNoDataValue(np.nan)
            band.WriteArray(delta[i])
        dataset.FlushCache()
        dataset = None

        # Use gdal translate to compress the image using LZW
        gdal.Translate(filepath, filepath_aux, options='-co COMPRESS=LZW')

        # Delete the auxiliary file
        self._io.delete_local_file(ImageRef(filename_aux, tile_ref=image_ref.tile_ref, type='delta'))

        return return_image_ref

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
        if self._on_the_server:
            dir_1 = f'{self._io.build_remote_dir_for_image(image_1)}'
            dir_2 = f'{self._io.build_remote_dir_for_image(image_2)}'
            filepath_1 = f'{dir_1}/{image_1.filename}'
            filepath_2 = f'{dir_2}/{image_2.filename}'
        else:
            dir_1 = f'{self._io.config.base_local_dir}/{image_1.rel_dir()}'
            dir_2 = f'{self._io.config.base_local_dir}/{image_2.rel_dir()}'
            filepath_1 = f'{dir_1}/{image_1.filename}'
            filepath_2 = f'{dir_2}/{image_2.filename}'

        self._io.check_existence_on_local(filepath_1, dir=False)
        self._io.check_existence_on_local(filepath_2, dir=False)

        return filepath_1, filepath_2

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
