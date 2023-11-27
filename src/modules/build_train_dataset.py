import time

import pandas as pd
from osgeo import gdal

from utils import ImageRef, RAW_IMAGE_SIZES, CROP_IMAGE_LIMITS, CROP_LIMITS_INSIDE_CROPPED
from modules.abstract_module import Module


class BuildTrainDataset(Module):
    """
    Module for building the train dataset for the aggregation formula.

    Attributes:
        _on_the_server (bool): if True, the module is being executed on the server.
        _raw_images_df (pd.DataFrame): dataframe with the raw images.
    """
    def __init__(self, config, io):
        super().__init__(config, io)

        self._on_the_server = None

        self._raw_images_df = self._io.filter_all_images(image_type='raw', filters={})

    def get_features_for_point(self, row) -> list:
        """
        Gets the features for a given point. The features are the NCI values, the delta values and the raw NDVI value.

        Args:
            row: (named tuple) the row of the train_inv dataframe.

        Returns:
            features (list): the features
        """
        i, j = row.i, row.j
        date = pd.to_datetime(row.detected_breakpoint, format='%Y-%m-%d')
        year = row.year
        tile = row.tile

        filepath_nci = f'nci3_NDVIrec_{year}_{tile}_{date.strftime("%Y%m%d")}.tif'
        filepath_delta = f'delta_NDVIrec_{year}_{tile}_{date.strftime("%Y%m%d")}.tif'

        image_ref_nci = ImageRef(filepath_nci, year=year, tile=tile, product='NDVI_reconstructed', type='nci')
        image_ref_delta = ImageRef(filepath_delta, year=year, tile=tile, product='NDVI_reconstructed', type='delta')

        raw_image = self._raw_images_df.loc[
            (self._raw_images_df['year'] == image_ref_nci.year) &
            (self._raw_images_df['tile'] == image_ref_nci.tile) &
            (self._raw_images_df['product'] == image_ref_nci.product) &
            (self._raw_images_df['filename'].str.contains(image_ref_nci.extract_date()))
            ].iloc[0]
        image_ref_raw = ImageRef(raw_image.filename, raw_image.year, raw_image.tile, raw_image['product'], type='raw')

        # Download images:
        if not self._on_the_server:
            self._io.download_file(image_ref_nci)
            self._io.download_file(image_ref_delta)
            self._io.download_file(image_ref_raw)

        nci = self._io.load_tif_as_ndarray(image_ref_nci)
        delta = self._io.load_tif_as_ndarray(image_ref_delta)

        if self._on_the_server:
            dir_raw = f'{self._io.build_remote_dir_for_image(image_ref_raw)}'
            filepath_raw = f'{dir_raw}/{image_ref_raw.filename}'
            raw = gdal.Open(filepath_raw).ReadAsArray()
        else:
            raw = self._io.load_tif_as_ndarray(image_ref_raw)

        # Crop raw image
        raw = self.crop_image(raw, image_ref_raw)

        min_i, max_i, min_j, max_j = CROP_LIMITS_INSIDE_CROPPED.get(image_ref_raw.tile)
        i -= min_i
        j -= min_j

        # Get features
        nci_values = nci[:, i, j]
        delta_values = delta[:, i, j]
        raw_ndvi = raw[i, j]

        # Build as list and return it
        features = []
        features.extend(nci_values)
        features.extend(delta_values)
        features += [raw_ndvi]

        return features

    def crop_image(self, raster, image_ref: ImageRef):
        # Check dimensions
        assert raster.shape == RAW_IMAGE_SIZES.get(image_ref.tile), \
            f'Image {image_ref.filename} has wrong dimensions. Expected: {RAW_IMAGE_SIZES.get(image_ref.tile)}'

        # Get dimensions from utils map
        min_i, max_i, min_j, max_j = CROP_IMAGE_LIMITS.get(image_ref.tile)

        # Crop image
        raster = raster[min_i:max_i, min_j:max_j]

        return raster

    def run(self, on_the_server=False):
        """
        Run the module.
        1. Read the train_inv.csv file.
        2. Iterate over the dataset and get the features for each point.
        3. Save the features in a csv file.

        Args:
            on_the_server: (bool) if True, the module is being executed on the server and an assertion error is raised.

        """
        print(f'Starting {self.__class__.__name__}')
        print(f'Start time: {pd.Timestamp.now()}')

        self._on_the_server = on_the_server
        assert not self._on_the_server, f'{self.__class__.__name__} is not implemented to run on the server.'
        train_inv_path = f'{self._io.config.base_local_dir}/operation_records/train_inv.csv'
        train_inv = pd.read_csv(train_inv_path)

        base_columns = ['i', 'j', 'year', 'tile', 'date', 'y']
        feature_names = [f'nci_{i}' for i in range(4)] + [f'delta_{i}' for i in range(5)] + ['raw']
        df = pd.DataFrame(columns=base_columns + feature_names)

        # Iterate over dataset and get features
        for index, row in enumerate(train_inv.itertuples()):
            print(f'Processing {index} of {len(train_inv)}')
            start_time = time.time()
            features = self.get_features_for_point(row)
            new_row = pd.DataFrame([[row.i, row.j, row.year, row.tile, row.detected_breakpoint, row.y] + features],
                                   columns=base_columns + feature_names)
            df = pd.concat([df, new_row], ignore_index=True)
            print(f' -- took {round(time.time() - start_time, 2)} seconds -- ')

        df.to_csv(f'{self._io.config.base_local_dir}/operation_records/train_features.csv', index=False)


if __name__ == '__main__':
    from config.config import Config
    from io_manager.io_manager import IO
    from config.io_config import IOConfig

    # config = Config()
    # io_config = IOConfig()
    # io = IO(io_config)
    #
    # get_features()
