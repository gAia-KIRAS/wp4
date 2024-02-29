import time
from typing import Tuple, Any

import pandas as pd
from numpy import ndarray
from osgeo import osr, ogr, gdal
from config.config import Config
from io_manager.io_manager import IO
from modules.abstract_module import Module
from utils import ImageRef, CROP_IMAGE_SIZES, CROP_LIMITS_INSIDE_CROPPED, reference_crop_images, timestamp, \
    coefficients_log_reg
import numpy as np


import os
os.environ['PROJ_LIB'] = '/home/siposova/miniconda3/envs/wp4_env/share/proj'


class ChangeDetection(Module):
    """
    Class that performs the Change Detection. It performs the step NCI + Delta -> CPROB + results_CD.csv
    This means that it takes the Delta images and the NCI images, and computes the change probability images. On top
    of these images, it applies a threshold to detect the change events and saves the predicted events in the
    results_CD file.

    If the c_prob.tif file already exists, only the application of the threshold is performed, and the
    existing image is used (not recomputed).

    There are two outputs:
    - results_cd.csv: tabular file with the detected events that have prob > threshold
    - c_prob.tif: for each image, a raster file with the probability of change for each pixel

    The results_cd file has the following columns:
    - version: version ID of the CD run
    - threshold: threshold used to filter the detected events
    - tile: tile ID
    - year: year of the image
    - row: row index of the pixel
    - column: column index of the pixel
    - date: date of the prediction
    - probability: prediction probability
    - timestamp: timestamp of the detection. When the prediction was executed
    - lat: latitude coordinate of the pixel
    - lon: longitude coordinate of the pixel

    The c_prob.tif file has only one band, which is the probability of change for each pixel.

    Attributes:
        _cd_records: (pd.DataFrame) the records of the CD runs.
        _cd_results: (pd.DataFrame) the CD results dataframe.
        _all_delta: (pd.DataFrame) contains information on all the delta images available.
        _on_the_server: (bool) if True, the module is being executed on the server.
        _cd_id: (str) the ID of the CD run. Becomes "version" in the results_cd file.
        _threshold: (float) the threshold used to filter the detected events. Must be in [0, 1].
        _type: (str) the type of CD to perform. Can be 'basic_mean', 'nci_logic' or 'log_reg'.
        _reference_images: (dict) dictionary with the reference images.
        _spatial_info_crop_images: (dict) dictionary with the spatial information of the reference images.
    """

    def __init__(self, config: Config, io: IO):
        super().__init__(config, io)

        self._cd_records = self._io.get_records_cd()
        self._cd_results = self._io.get_results_cd()
        self._all_delta = self._io.list_all_files_of_type('delta')

        self._on_the_server = None
        self._cd_id = self._config.cd_conf['cd_id']
        self._threshold = self._config.cd_conf['threshold']
        self._type = self._config.cd_conf['type']

        self._reference_images = {
            tile: gdal.Open(f'{self._io.config.base_local_dir}/{reference_crop_images[tile]}')
            for tile in self._io.config.available_tiles
        }
        self._spatial_info_crop_images = {
            tile: (self._reference_images[tile].GetProjection(),
                   self._reference_images[tile].GetGeoTransform(),
                   self._build_transform_inverse(self._reference_images[tile]),
                   self._build_transform(self._reference_images[tile]))

            for tile in self._io.config.available_tiles
        }

    @staticmethod
    def _build_transform_inverse(reference_image: gdal.Dataset) -> osr.CoordinateTransformation:
        """
        Builds the inverse transformation from the reference image to the original image.
        Args:
            reference_image: reference image

        Returns:
            inverse transformation
        """
        print(' -- Building inverse transformation')
        source = osr.SpatialReference(wkt=reference_image.GetProjection())
        target = osr.SpatialReference()
        target.ImportFromEPSG(4326)
        transform = osr.CoordinateTransformation(source, target)
        return transform

    @staticmethod
    def _build_transform(reference_image: gdal.Dataset) -> osr.CoordinateTransformation:
        """
        Builds the transformation from the original image to the reference image.
        Args:
            reference_image: reference image

        Returns:
            transformation
        """
        print(' -- Building transformation')
        source = osr.SpatialReference()
        source.ImportFromEPSG(4326)
        target = osr.SpatialReference(wkt=reference_image.GetProjection())
        transform = osr.CoordinateTransformation(source, target)
        return transform

    def pixel_to_latlon(self, tile, i, j) -> Tuple[float, float]:
        """
        Find the lat and lon coordinates of the pixel (i, j) in the original image.

        Args:
            tile: tile ID
            i: row index of the pixel
            j: column index of the pixel

        Returns:
            tuple: lat and lon coordinates of the pixel
        """
        world_j, world_i = self.pixel_to_world(tile, i, j)
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(world_j, world_i)
        point.Transform(self._spatial_info_crop_images[tile][2])
        return point.GetX(), point.GetY()

    def pixel_to_world(self, tile, i, j) -> Tuple[float, float]:
        """
        Transforms the pixel coordinates to world coordinates using the reference image.
        Args:
            tile: tile ID
            i: row index of the pixel
            j: column index of the pixel

        Returns:
            j and i coordinates of the pixel in the world coordinates
        """
        geo_matrix = self._spatial_info_crop_images[tile][1]
        return j * geo_matrix[1] + geo_matrix[0], i * geo_matrix[5] + geo_matrix[3]

    def run(self, on_the_server: bool = False) -> None:
        """
        Runs the Change Detection module. Follows the steps:
        1. Get all the images to process.
        2. Iterate over the images and perform the CD.
            2.1 If the c_prob image is already computed for the image, only the threshold is applied.
            2.2 If the c_prob image is not computed, it is computed and the threshold is applied.
        3. Save the results and the records.

        Args:
            on_the_server: (bool) if True, the module is executed on the server.

        """

        # Update parameters
        self._on_the_server = on_the_server

        # Check filters
        assert not (set(self._config.filters['product']) - {'NDVI_reconstructed'}),\
            'CD can only be applied to NDVI_reconstructed'

        # Get all tiles and dates
        cd_done = self._cd_records.loc[
            (self._cd_records['cd_id'] == self._cd_id) &
            (self._cd_records['threshold'] == self._threshold),
            ['tile', 'filename_from', 'year']
        ].rename(columns={'filename_from': 'filename'}).values.tolist()

        years_filter = self._config.filters['year'] if self._config.filters['year'] else self._io.config.available_years
        tiles_filter = self._config.filters['tile'] if self._config.filters['tile'] else self._io.config.available_tiles

        cd_todo = self._all_delta.loc[
            (self._all_delta['tile'].isin(tiles_filter)) &
            (self._all_delta['year'].isin(years_filter)),
            ['tile', 'filename', 'year']
        ].values.tolist()
        # Convert to list of tuples
        cd_todo = [(tile, filename, year) for tile, filename, year in cd_todo]
        cd_done = [(tile, filename, year) for tile, filename, year in cd_done]
        cd_todo = sorted(list(set(cd_todo) - set(cd_done)))

        print(f'Filters: {self._config.filters}')
        print(f'CD ID: {self._cd_id}  -  Threshold: {self._threshold}')
        print(f'Number of images to process: {len(cd_todo)}')

        start_timestamp, start_time = timestamp(), time.time()
        i = 0
        while i < len(cd_todo) and time.time() - start_time < self._time_limit * 60:
            print(f' -- Processing image {i + 1} of {len(cd_todo)}. {round((time.time() - start_time) / 60, 1)} '
                  f'min elapsed.\n ---- Image name: {cd_todo[i][1]}')
            tile, filename, year = cd_todo[i]
            image_delta = ImageRef(filename, year, tile, 'NDVI_reconstructed', type='delta')
            image_nci = ImageRef(filename.replace('delta', 'nci3'), year, tile, 'NDVI_reconstructed', type='nci')
            image_cprob = ImageRef(filename.replace('delta', 'cprob'), tile_ref=image_delta.tile_ref, type='cprob')

            # Load the time-series for the subtile
            detected_events = self.perform_cd(image_delta, image_nci, image_cprob)

            # Add detected events to the results
            self.add_results(image_cprob, detected_events, image_cprob.extract_date())

            # Update the records
            self.update_records(image_delta, image_cprob, len(detected_events))

            i += 1

        # Save records and results
        self._io.save_records_cd(self._cd_records)
        self._io.save_results_cd(self._cd_results)

        print(f' -- CD finished. {i} images processed in {time.time() - start_time} seconds.')

    def perform_cd(self, delta_imref: ImageRef, nci_imref: ImageRef, cprob_imref: ImageRef) -> list:
        """
        Performs the change-detection on the given image.
        Checks if the c_prob image exists. Otherwise, computes it.
        Args:
            delta_imref: ImageRef of the delta image
            nci_imref: ImageRef of the nci image
            cprob_imref: ImageRef of the c_prob image

        Returns:
            detected_events: list of detected events and the associated probabilites
        """
        cprob_dir = f'{self._io.config.base_local_dir}/{cprob_imref.rel_dir()}'
        cprob_filepath = f'{cprob_dir}/{cprob_imref.filename}'

        try:
            self._io.check_existence_on_local(cprob_filepath, dir_name=False)
        except FileNotFoundError:
            self.compute_c_prob(delta_imref, nci_imref)

        # Load the c_prob.tif file
        c_prob = self._io.load_tif_as_ndarray(cprob_imref)

        detected_events, detected_probs = self.apply_filter(c_prob)

        return list(zip(detected_events, detected_probs))

    def apply_filter(self, c_prob: np.ndarray) -> tuple[ndarray, ndarray]:
        """
        Applies the threshold to the c_prob.tif file and returns the detected events and the associated probabilities.

        Args:
            c_prob: (np.ndarray) corresponding to the c_prob image

        Returns:
            detected_events: list of detected events
            detected_probs: list of probabilities associated to the detected events

        """
        print(f' ---- Applying threshold {self._threshold} to c_prob.tif file.')
        # Get index of the pixels with prob > threshold
        detected_events = np.asarray(np.where(c_prob >= self._threshold)).T.tolist()
        detected_probs = c_prob[np.where(c_prob >= self._threshold)]
        return detected_events, detected_probs

    def compute_c_prob(self, delta_imref: ImageRef, nci_imref: ImageRef) -> None:
        """
        Compute the change probability, saving it to the c_prob.tif file.
        Apply the corresponding logic depending on the type of CD to perform. Possible types:
        - basic_mean: average of the correlation, pixel vs average and ndvi differences
        - nci_logic: average of the correlation, pixel vs average and ndvi differences, but set to 0 the pixels with
        slope > 0
        - log_reg: use the aggregation given by the logistic regression model

        Args:
            delta_imref: ImageRef of the delta image
            nci_imref: ImageRef of the nci image

        """
        print(f' ---- Computing c_prob.tif file.')

        if not self._on_the_server:
            # Download the delta image
            self._io.download_file(delta_imref)
            if self._type == 'log_reg':
                self._io.download_file(nci_imref)
        delta = self._io.load_tif_as_ndarray(delta_imref)
        if self._type == 'log_reg':
            nci = self._io.load_tif_as_ndarray(nci_imref)

        n_bands = delta.shape[0]
        delta[np.isnan(delta)] = 0

        if self._type in ['basic_mean', 'nci_logic']:
            for b in range(n_bands):
                # Clip with the 5th and 95th percentiles
                v_min, v_max = np.percentile(delta[b, :, :], [5, 95])
                delta[b, :, :] = np.clip(delta[b, :, :], v_min, v_max)

                # Take mask of the pixels with slope > 0
                mask = delta[1, :, :] > 0

                # Normalize to [0, 1]
                delta[b, :, :] = (delta[b, :, :] - v_min) / (v_max - v_min)

            # Average the bands 0 (correlation), 3 (pixel vs average), 4 (ndvi differences)
            c_prob = np.mean(delta[[0, 3, 4], :, :], axis=0)

        if self._type == 'nci_logic':
            # Set all pixels with slope > 0 to c_prob = 0
            c_prob[mask] = 0

        if self._type == 'log_reg':
            c_prob = np.ones(delta.shape[1:]) * coefficients_log_reg['intercept']
            c_prob += coefficients_log_reg['nci_0'] * nci[0, :, :]
            c_prob += coefficients_log_reg['nci_1'] * nci[1, :, :]
            c_prob += coefficients_log_reg['nci_2'] * nci[2, :, :]
            c_prob += coefficients_log_reg['nci_3'] * nci[3, :, :]
            c_prob += coefficients_log_reg['delta_0'] * delta[0, :, :]
            c_prob += coefficients_log_reg['delta_1'] * delta[1, :, :]
            c_prob += coefficients_log_reg['delta_2'] * delta[2, :, :]
            c_prob += coefficients_log_reg['delta_3'] * delta[3, :, :]
            c_prob += coefficients_log_reg['delta_4'] * delta[4, :, :]

            # Apply the logistic function
            c_prob = 1 / (1 + np.exp(-c_prob))

        # Save the c_prob.tif file
        c_prob_filename = f'{delta_imref.filename.replace("delta", "cprob")}'

        # Save image of size = crop image
        cprob_imref = ImageRef(c_prob_filename, tile_ref=delta_imref.tile_ref, type='cprob')

        raster_shape = CROP_IMAGE_SIZES[cprob_imref.tile]
        raster = np.empty(raster_shape, dtype=np.float32)
        raster.fill(np.nan)
        min_i, max_i, min_j, max_j = CROP_LIMITS_INSIDE_CROPPED.get(cprob_imref.tile)

        raster[min_i:max_i, min_j:max_j] = c_prob
        del c_prob

        crs, geotransform, _, _ = self._spatial_info_crop_images[delta_imref.tile_ref.tile]
        self._io.save_ndarray_as_tif(raster, cprob_imref, crs=crs, geotransform=geotransform)

    def add_results(self, image_ref: ImageRef, detected_events: list, date: str) -> None:
        """
        Adds the detected events to the results file. One row per detected event.

        Args:
            image_ref: ImageRef of the image
            detected_events: list of detected events and the associated probabilites
            date: date of the image. Will be the date of the detected breakpoint
        """
        new_records = {
            'row': [i for (i, j), prob in detected_events],
            'column': [j for (i, j), prob in detected_events],
            'probability': [prob for _, prob in detected_events]
        }
        new_records = pd.DataFrame(new_records)
        new_records['version'] = self._cd_id
        new_records['threshold'] = self._threshold
        new_records['tile'] = image_ref.tile
        new_records['year'] = image_ref.year
        new_records['date'] = date
        new_records['date'] = pd.to_datetime(new_records['date'], format="%Y%m%d")
        new_records['timestamp'] = timestamp()
        new_records['timestamp'] = pd.to_datetime(new_records['timestamp'], format="%Y%m%d_%H%M%S")
        new_records[['lat', 'lon']] = new_records.apply(
            lambda x: self.pixel_to_latlon(image_ref.tile, x['row'], x['column']), axis=1, result_type='expand')

        self._cd_results = pd.concat([self._cd_results, new_records], ignore_index=True).reset_index(drop=True)

    def update_records(self, image_from: ImageRef, image_to: ImageRef, n_detected_events: int) -> None:
        """
        Updates the records file with the new CD run.

        Args:
            image_from: ImageRef of the delta image
            image_to: ImageRef of the c_prob image
            n_detected_events: number of detected events
        """
        record = [self._cd_id, self._threshold, image_from.tile, image_from.year, image_from.filename, image_to
        .filename, n_detected_events, timestamp()]
        self._cd_records.loc[len(self._cd_records)] = record

