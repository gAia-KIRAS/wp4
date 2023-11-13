import time
from typing import Tuple

import pandas as pd
from osgeo import osr, ogr, gdal
from config.config import Config
from io_manager.io_manager import IO
from config.io_config import IOConfig
from modules.abstract_module import Module
from utils import ImageRef, CROP_IMAGE_SIZES, CROP_LIMITS_INSIDE_CROPPED, reference_crop_images, timestamp
import numpy as np

# import os
# os.environ['PROJ_LIB'] = '/home/salva/miniconda3/envs/wp4_env/share/proj'


class ChangeDetection(Module):
    """
    Change-Detection module. The input are the Delta images, which is a 5-band time-series.

    If the c_prob.tif file already exists, only the application of the threshold is performed, and the
    existing image is used (not recomputed).

    There are two outputs:
    - results_cd.csv: tabular file with the detected events that have prob > threshold
    - c_prob.tif: raster file with the probability of change for each pixel

    The results_cd file has the following columns:
    - cd_id: ID of the CD run
    - threshold: threshold used to filter the detected events
    - tile: tile ID
    - subtile: subtile ID
    - i: row index of the pixel
    - j: column index of the pixel
    - timestamp: timestamp of the CD run
    - detected_breakpoint: timestamp of the detected breakpoint
    - d_prob: probability of change

    The c_prob.tif file has only one band, which is the probability of change for each pixel.
    """

    def __init__(self, config: Config, io: IO):
        super().__init__(config, io)

        self._cd_records = self._io.get_records_cd()
        self._cd_results = self._io.get_results_cd()
        self._all_delta = self._io.list_all_files_of_type('delta')

        self._on_the_server = None
        self._cd_id = None
        self._threshold = None
        self._type = None
        print(f'Loading reference images')
        self._reference_images = {
            tile: gdal.Open(f'{self._io.config.base_local_dir}/{reference_crop_images[tile]}')
            for tile in self._io.config.available_tiles
        }
        print(f'Building spatial info for reference images')
        self._spatial_info_crop_images = {
            tile: (self._reference_images[tile].GetProjection(),
                   self._reference_images[tile].GetGeoTransform(),
                   self._build_transform_inverse(self._reference_images[tile]))
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
        # Update parameters
        self._on_the_server = on_the_server
        self._cd_id = self._config.cd_conf['cd_id']
        self._threshold = self._config.cd_conf['threshold']
        self._type = self._config.cd_conf['type']

        # Check filters
        assert not (set(self._config.filters['product']) - {'NDVI_reconstructed'}), \
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
            image_cprob = ImageRef(filename.replace('delta', 'cprob'), tile_ref=image_delta.tile_ref, type='cprob')

            # Load the time-series for the subtile
            detected_events = self.perform_cd(image_delta, image_cprob)

            # Add detected events to the results
            self.add_results(image_cprob, detected_events, image_cprob.extract_date())

            # Update the records
            self.update_records(image_delta, image_cprob, len(detected_events))

            i += 1

        # Save records and results
        self._io.save_records_cd(self._cd_records)
        self._io.save_results_cd(self._cd_results)

        print(f' -- CD finished. {i} images processed in {time.time() - start_time} seconds.')

    def perform_cd(self, delta_imref: ImageRef, cprob_imref: ImageRef) -> list:
        """
        Performs the change-detection on the given image.
        Checks if the c_prob image exists. Otherwise, computes it.
        Args:
            delta_imref: ImageRef of the delta image
            cprob_imref: ImageRef of the c_prob image

        Returns:
            detected_events: list of detected events and the associated probabilites
        """
        cprob_dir = f'{self._io.config.base_local_dir}/{cprob_imref.rel_dir()}'
        cprob_filepath = f'{cprob_dir}/{cprob_imref.filename}'

        try:
            self._io.check_existence_on_local(cprob_filepath, dir=False)
        except FileNotFoundError:
            self.compute_c_prob(delta_imref)

        # Load the c_prob.tif file
        c_prob = self._io.load_tif_as_ndarray(cprob_imref)

        detected_events, detected_probs = self.apply_filter(c_prob)

        return list(zip(detected_events, detected_probs))

    def apply_filter(self, c_prob):
        print(f' ---- Applying threshold {self._threshold} to c_prob.tif file.')
        # Get index of the pixels with prob > threshold
        detected_events = np.asarray(np.where(c_prob >= self._threshold)).T.tolist()
        detected_probs = c_prob[np.where(c_prob >= self._threshold)]
        return detected_events, detected_probs

    def compute_c_prob(self, delta_imref: ImageRef) -> None:
        print(f' ---- Computing c_prob.tif file.')

        if not self._on_the_server:
            # Download the delta image
            self._io.download_file(delta_imref)
        delta = self._io.load_tif_as_ndarray(delta_imref)

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

        crs, geotransform, _ = self._spatial_info_crop_images[delta_imref.tile_ref.tile]
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
            'i': [i for (i, j), prob in detected_events],
            'j': [j for (i, j), prob in detected_events],
            'd_prob': [prob for _, prob in detected_events]
        }
        new_records = pd.DataFrame(new_records)
        new_records['cd_id'] = self._cd_id
        new_records['threshold'] = self._threshold
        new_records['tile'] = image_ref.tile
        new_records['year'] = image_ref.year
        new_records['detected_breakpoint'] = date
        new_records['timestamp'] = timestamp()
        new_records[['lat', 'lon']] = new_records.apply(
            lambda row: self.pixel_to_latlon(image_ref.tile, row['i'], row['j']), axis=1, result_type='expand')

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

    def _check_if_subtile_is_available(self, subtile):
        filepath = f'{self._io.config.base_local_dir}/ts/ts_{subtile}.pkl'
        try:
            self._io.check_existence_on_local(filepath, dir=False)
        except FileNotFoundError:
            return None
        print(f' -- Loading time-series for subtile {subtile} from local file.')
        return self._io.load_pickle(filepath)

    def _save_subtile_ts(self, subtile, signal, dates):
        print(f' -- Saving time-series for subtile {subtile}.')
        dirpath = f'{self._io.config.base_local_dir}/ts'
        self._io.check_existence_on_local(dirpath, dir=True)
        filepath = f'{dirpath}/ts_{subtile}.pkl'
        self._io.save_pickle((signal, dates), filepath)
