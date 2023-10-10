import time
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
    Change-Detection module. The input are the NCI images, which is a 4-band time-series of satellite images.
    The output is a list of detected events, with the following information:
    - cd_id: ID of the change-detection operation
    - tile: tile of the detected event
    - subtile: subtile of the detected event
    - pixel_i: i coordinate of the detected event
    - pixel_j: j coordinate of the detected event
    - timestamp: timestamp of the detected event
    - date: date of the detected event
    - subproduct: subproduct of the detected event (r, a, b, m)
    """

    def __init__(self, config: Config, io: IO):
        super().__init__(config, io)

        self._cd_records = self._io.get_records_cd()
        self._cd_results = self._io.get_results_cd()
        self._all_delta = self._io.list_all_files_of_type('delta')

        self._on_the_server = None
        self._cd_id = None
        self._pelt_penalty = None

    def run(self, on_the_server: bool = False) -> None:
        # Each tile is divided in [1000x1000] pixels. Starting from the top left corner, we take the first [1000x1000]
        # pixels and apply CD to them. Then we take the next [1000x1000] pixels and apply CD to them, and so on.
        # The right-most subtile can have width < 1000, and the bottom-most subtile can have height < 1000.

        # Update parameters
        self._on_the_server = on_the_server
        self._cd_id = self._config.cd_conf['cd_id']
        self._pelt_penalty = self._config.cd_conf['penalty']

        # Check filters
        assert not self._config.filters['year'], 'Year filter is not supported for CD'
        assert not (set(self._config.filters['product']) - {'NDVI_reconstructed'}), \
            'CD can only be applied to NDVI_reconstructed'

        # Tiles to execute
        tiles = self._config.filters['tile'] \
            if self._config.filters['tile'] is not None else self._io.config.available_tiles

        # Get the subtiles that have not been processed yet
        all_subtiles = subtiles.keys()
        done_subtiles = self._cd_records.loc[
            (self._cd_records.cd_id == self._cd_id), 'subtile'].values
        todo_subtiles = set(all_subtiles) - set(done_subtiles)
        todo_subtiles = sorted([s for s in todo_subtiles if s[:5] in tiles], key=lambda x: int(x.split('_')[1]))

        print(f'Filters: {self._config.filters}')
        print(f'CD ID: {self._cd_id}')
        print(f'Number of subtiles to process: {len(todo_subtiles)}')

        start_timestamp, start_time = timestamp(), time.time()
        i = 0
        while i < len(todo_subtiles) and time.time() - start_time < self._time_limit:
            print(f' -- Processing subtile {todo_subtiles[i]} ({i + 1} of {len(todo_subtiles)}). ')
            subtile = todo_subtiles[i]

            # Load the time-series for the subtile
            signal, dates = self.load_subtile_ts(subtile)

            # Run the CD algorithm
            detected_events = self.perform_cd(signal, dates)

            # Add detected events to the results
            self.add_results(subtile, detected_events)

            # Update the records
            self.update_records(subtile)

        # Save records and results
        self._io.save_records_cd(self._cd_records)
        self._io.save_results_cd(self._cd_results)

    def perform_cd(self, signal, dates):
        detected_events = []
        ts_length, n_bands, imax, jmax = signal.shape
        assert ts_length == len(dates), f'Raster shape {signal.shape} does not match dates length {len(dates)}'
        print(f' -- Start CD at {timestamp()} --')
        for i in range(imax):
            for j in range(jmax):
                if (i * jmax + j) % 10000 == 0:
                    print(f' -- -- Processed {round(100 * (i * jmax + j) / (imax * jmax))}% pixels.')
                    print(f' -- -- Found {len(detected_events)} events so far. {timestamp()}')

                ts = signal[:, :, i, j]
                pelt = rpt.Pelt(model='rbf')
                result = pelt.fit_predict(ts, pen=self._pelt_penalty)

                if len(result) == 1:
                    continue
                for r in result[:-1]:
                    # print(f' -- -- -- Detected event at {dates[r]}')
                    # Leave field 'subproduct' empty because CD is applied to all features
                    detected_events.append((i, j, dates[r], ''))
        return detected_events

    def add_results(self, subtile, detected_events):
        imin, imax, jmin, jmax = subtiles[subtile]
        for i, j, date, subproduct in detected_events:
            pixel_id = imin + i, jmin + j
            record = [self._cd_id, subtile[:5], subtile, pixel_id[0], pixel_id[1], timestamp(), date, subproduct]
            assert len(record) == len(self._cd_results.columns), \
                f'Length of record {len(record)} does not match length of columns {len(self._cd_results.columns)}'
            self._cd_results.loc[len(self._cd_results)] = record

    def update_records(self, subtile):
        record = [self._cd_id, subtile[:5], subtile, timestamp(), int(self._on_the_server)]
        self._cd_records.loc[len(self._cd_records)] = record

    def _check_if_subtile_is_available(self, subtile):
        filepath = f'{self._io.config.base_local_dir}/ts/ts_{subtile}.pkl'
        try:
            self._io.check_existence_on_local(filepath, dir=False)
        except FileNotFoundError:
            return None
        print(f' -- Loading time-series for subtile {subtile} from local file.')
        return self._io.load_pickle(filepath)

    def load_subtile_ts(self, subtile):
        print(f' -- Loading time-series for subtile {subtile}.')

        assert subtile in subtiles.keys(), f'{subtile} is not a valid subtile.'

        res = self._check_if_subtile_is_available(subtile)

        if res:
            return res

        # Get Delta images of the file
        ts = self._all_delta.loc[self._all_delta['tile'] == subtile[:5]].sort_values(by=['year', 'date_f'])

        # # TODO: remove testing pipeline
        # ts = ts.loc[ts['filename'].isin(['delta_NDVIrec_2018_33TUM_20180101.tif', 'delta_NDVIrec_2018_33TUM_20180111.tif'])]
        # ts.reset_index(inplace=True)

        # Get the subtile limits
        ilim1, ilim2, jlim1, jlim2 = subtiles[subtile]

        signal = np.ndarray(shape=(ts.shape[0], 5, ilim2 - ilim1, jlim2 - jlim1), dtype=np.float32)

        # Define a dates list
        dates = []
        loading_start_time = time.time()

        for i, row in ts.iterrows():
            image_start_time = time.time()
            print(f' -- -- Loading image {i + 1} / {ts.shape[0]}')

            image = ImageRef(row['filename'], row['year'], row['tile'], row['product'], type='delta')

            if not self._on_the_server:
                # Download the image (if not available locally, handled by IO)
                self._io.download_file(image)

            if self._on_the_server:
                image_dir = self._io.build_remote_dir_for_image(image)
                image_filepath = f'{image_dir}/{image.filename}'
            else:
                image_dir = f'{self._io.config.base_local_dir}/{image.rel_dir()}'
                image_filepath = f'{image_dir}/{image.filename}'

            self._io.check_existence_on_local(image_filepath, dir=False)

            r = gdal.Open(image_filepath).ReadAsArray()

            signal[i] = r[:, ilim1:ilim2, jlim1:jlim2]

            # raster_r[i] = r[0, ilim1:ilim2, jlim1:jlim2]
            # raster_a[i] = r[1, ilim1:ilim2, jlim1:jlim2]
            # raster_b[i] = r[2, ilim1:ilim2, jlim1:jlim2]
            # raster_m[i] = r[3, ilim1:ilim2, jlim1:jlim2]

            # self._io.delete_local_file(image) # TODO: will be active when not testing

            dates.append(image.extract_date())
            print(f' -- -- -- took {round(time.time() - image_start_time, 2)} seconds.')

        # Cut the rasters to the correct size

        print(f' -- Loading completed. Time-series shape: {signal.shape}. '
              f' -- -- Took {round(time.time() - loading_start_time, 2)} seconds.')

        self._save_subtile_ts(subtile, signal, dates)

        return signal, dates

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
