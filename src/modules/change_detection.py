import time
from typing import Tuple

import pandas as pd
from osgeo import gdal
import ruptures as rpt
from config.config import Config
from io_manager.io_manager import IO
from config.io_config import IOConfig
from modules.abstract_module import Module
from utils import ImageRef, subtiles, reference_nci_images, timestamp
import numpy as np


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

    def run(self, on_the_server: bool = False) -> None:
        # Update parameters
        self._on_the_server = on_the_server
        self._cd_id = self._config.cd_conf['cd_id']
        self._threshold = self._config.cd_conf['threshold']

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
            ~self._all_delta['filename'].isin(cd_done['filename']) &
            (self._all_delta['tile'].isin(tiles_filter)) &
            (self._all_delta['year'].isin(years_filter)),
            ['tile', 'filename', 'year']
        ].values.tolist()
        cd_todo = set(cd_todo) - set(cd_done)

        print(f'Filters: {self._config.filters}')
        print(f'CD ID: {self._cd_id}  -  Threshold: {self._threshold}')
        print(f'Number of images to process: {len(cd_todo)}')

        start_timestamp, start_time = timestamp(), time.time()
        i = 0
        while i < len(cd_todo) and time.time() - start_time < self._time_limit * 60:
            print(f' -- Processing image {i + 1} of {len(cd_todo)}. {time.time() - start_time} seconds elapsed.')
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
        print(f' ---- Computing c_prob.tif file for {delta_imref}.')

        if not self._on_the_server:
            # Download the delta image
            self._io.download_file(delta_imref)
        delta = self._io.load_tif_as_ndarray(delta_imref)

        n_bands = delta.shape[0]
        delta[np.isnan(delta)] = 0
        for b in range(n_bands):
            # Clip with the 5th and 95th percentiles
            v_min, v_max = np.percentile(delta[b, :, :], [5, 95])
            delta[b, :, :] = np.clip(delta[b, :, :], v_min, v_max)

            # Normalize to [0, 1]
            delta[b, :, :] = (delta[b, :, :] - v_min) / (v_max - v_min)

        # Average the bands 0 (correlation), 3 (pixel vs average), 4 (ndvi differences)
        c_prob = np.mean(delta[[0, 3, 4], :, :], axis=0)

        # Save the c_prob.tif file
        c_prob_filename = f'{delta_imref.filename.replace("delta", "cprob")}'

        c_prob_imref = ImageRef(c_prob_filename, tile_ref=delta_imref.tile_ref, type='cprob')
        self._io.save_ndarray_as_tif(c_prob, c_prob_imref)

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
            'd_prob': [prob for (i, j), prob in detected_events]
        }
        new_records = pd.DataFrame(new_records)
        new_records['cd_id'] = self._cd_id
        new_records['threshold'] = self._threshold
        new_records['tile'] = image_ref.tile
        new_records['detected_breakpoint'] = date
        new_records['timestamp'] = timestamp()

        self._cd_results = pd.concat([self._cd_results, new_records], ignore_index=True).reset_index(drop=True)

    def update_records(self, image_from: ImageRef, image_to: ImageRef, n_detected_events: int) -> None:
        """
        Updates the records file with the new CD run.

        Args:
            image_from: ImageRef of the delta image
            image_to: ImageRef of the c_prob image
            n_detected_events: number of detected events
        """
        record = [self._cd_id, self._threshold, image_from.tile, image_from.filename, image_to
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

    def create_test_image(self):
        image = ImageRef("nci3_NDVIrec_2020_33TUM_20200101.tif", 2020, '33TUM', 'NDVI_reconstructed', 'nci')
        self._io.download_file(image)
        base = gdal.Open(f'{self._io.config.base_local_dir}/{image.rel_filepath()}').ReadAsArray()
        print(base.shape)

        # Crop the image to [1000x1000] pixels and save it as test_cd.tif
        base = base[:, -500:, -500:]
        print(base.shape)
        # Save the image

        filepath_aux = f'{self._io.config.base_local_dir}/testing/test_cd_aux.tif'
        filepath = f'{self._io.config.base_local_dir}/testing/test_cd.tif'

        # Save an auxiliary .tif file
        driver = gdal.GetDriverByName('GTiff')
        n_bands, rows, cols = base.shape
        dataset = driver.Create(filepath_aux, cols, rows, n_bands, gdal.GDT_Float32)
        for i in range(n_bands):
            band = dataset.GetRasterBand(i + 1)
            if i == 0:
                band.SetNoDataValue(np.nan)
            band.WriteArray(base[i])
        dataset.FlushCache()
        dataset = None

        # Use gdal translate to compress the image using LZW
        gdal.Translate(filepath, filepath_aux, options='-co COMPRESS=LZW')

    def experiments(self):
        data = self.load_image(2)

        print(f'Applying CD to {data.shape} array.')

        # self.bfast_cd(data)
        self.ruptures_cd(data)

    def ruptures_cd(self, data):
        """
        According to documentation:
        penalty value: "As a rule of thumb, the more noise, samples or dimensions, the larger this parameter should be."
        bic = sigma*sigma*np.log(T)*d gives also a starting value for the penalty
        """
        # Get image dimensions
        n_images, n_bands, rows, cols = data.shape

        res = np.ndarray(shape=(rows, cols), dtype=object)
        total_pixels = rows * cols
        for i in range(rows):
            for j in range(cols):
                if i % 100 == 0 and j % 100 == 0:
                    print(f'Processed {i * cols + j} / {total_pixels} pixels.')

                ts = data[:, :, i, j]
                algo = rpt.Pelt(model="rbf").fit(ts)
                result = algo.predict(pen=0.1)
                # Save to res
                res[i, j] = str(result)

        print(res)

    def load_image(self, extra_images=1):
        base = gdal.Open(f'{self._io.config.base_local_dir}/testing/test_cd.tif').ReadAsArray()
        # Take only first band
        # base = base[0]
        # Create random image of size 7849 * 3463
        base = np.random.rand(4, 7849, 3463)
        base = np.expand_dims(base, axis=0)
        data = base.copy()
        for i in range(extra_images):
            # if i == 5:
            #     base[0, 0, 0] = 100
            data = np.concatenate((data, base), axis=0)

        return data

    def create_and_number_subdivisions(self):
        subtiles = {}
        for tile in self._io.config.available_tiles:
            image = ImageRef(reference_nci_images[tile], 2020, tile, 'NDVI_reconstructed', 'nci')
            # self._io.download_file(image)
            base = gdal.Open(f'{self._io.config.base_local_dir}/{image.rel_filepath()}').ReadAsArray()
            print(f'tile: {tile}\nImage shape: {base.shape}')
            original_pix = base.shape[1] * base.shape[2]

            ilim = 0
            pixel_count = 0
            sub_count = 0
            while ilim < base.shape[1]:
                ilim += 1000
                jlim = 0
                while jlim < base.shape[2]:
                    jlim += 1000
                    name = f'{tile}_{sub_count}'
                    subtiles[name] = [ilim - 1000, ilim, jlim - 1000, jlim]
                    # print(f'[{ilim-1000}:{ilim}, {jlim-1000}:{jlim}]')
                    # print(f'This gives an image of size: {base[:, ilim-1000:ilim, jlim-1000:jlim].shape}')
                    pixel_count += base[:, ilim - 1000:ilim, jlim - 1000:jlim].shape[1] * \
                                   base[:, ilim - 1000:ilim, jlim - 1000:jlim].shape[2]
                    sub_count += 1

            assert pixel_count == original_pix, f'Pixel count does not match. Original: {original_pix}, new: {pixel_count}'

        print(subtiles)


if __name__ == '__main__':
    config = Config()
    io_config = IOConfig()
    io_manager = IO(io_config)

    cd = ChangeDetection(config, io_manager)
    # cd.create_test_image()
    cd.experiments()
    # cd.create_and_number_subdivisions()
