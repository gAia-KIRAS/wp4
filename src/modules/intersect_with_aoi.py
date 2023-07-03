import geopandas as gpd
import warnings
import time

import pandas as pd
from osgeo import gdal

from src.config.config import Config
from src.io.io_manager import IO
from src.utils import ImageRef, TileRef, timestamp, RECORDS_FILE_COLUMNS
from typing import Union


class IntersectAOI:
    """
    This module does the step: RAW -> CROP
    It intersects the Area of Interest (AOI) with the raw images and saves the result as a new .tif file.

    Attributes:
        _io (IO): IO object
        _config (Config): Config object
        _records (pd.DataFrame): records DataFrame, as defined in IO.get_records()
        _time_limit (int): time limit in minutes for run method
    """

    def __init__(self, io: IO, config: Config):
        self._io = io
        self._config = config
        self._check_aoi()
        self._records = self._io.get_records()
        self._time_limit = self._config.time_limit

    def _check_aoi(self):
        """
        Checks if the Area of Interest (AOI) exists locally as a .shp file.
        If not, it is loaded as .gpkg and saved as .shp file. If .gpkg does not exist, an error is thrown.
        """
        filepath_shp = self._io.config.aoi_path['shp']
        try:
            self._io.check_existence_on_local(filepath_shp, dir=False)
        except FileNotFoundError:
            filepath_gpkg = self._io.config.aoi_path['gpkg']

            # .gpkg should exist, otherwise throw an error
            self._io.check_existence_on_local(filepath_gpkg, dir=False)

            warnings.warn(f'\nAOI in .shp format did not exist. Creating it from {filepath_gpkg}.')
            aoi = gpd.read_file(filepath_gpkg)
            aoi.to_file(filepath_shp, driver='ESRI Shapefile')
            self._io.check_existence_on_local(filepath_shp, dir=False)

    def read_aoi(self) -> gpd.GeoDataFrame:
        """
        Reads the Area of Interest (AOI) from the local file. File format is .gpkg and is saved as a GeoDataFrame.

        Returns:
            gdf (GeoDataFrame): GeoDataFrame with the AOI
        """
        filepath = self._io.config.aoi_path['gpkg']
        self._io.check_existence_on_local(filepath, dir=False)
        gdf = gpd.read_file(filepath)
        return gdf

    def intersect(self, image: ImageRef) -> Union[ImageRef, None]:
        """
        Intersects a local .tif file referenced by image (ImageRef) with the Area of Interest AOI.
        Saves the result locally in a new .tif file with the following name:
        - 'crop_{image.product}_{image.year}_{imgae.tile}_{image.date}.tif'

        Args:
            image: ImageRef object with the image to intersect. Needs to have the attribute type set.

        Returns:
            clip_image_ref (ImageRef | None): If the intersection was successful, returns an ImageRef
             object with the clipped image. Has the attribute type set to 'crop'. Otherwise (if the
             raster does not have a valid SRS), returns None.
        """
        print(f'Intersecting image {image} with the AOI.')
        dir = f'{self._io.config.base_local_dir}/{image.rel_dir()}'
        filepath = f'{self._io.config.base_local_dir}/{image.rel_filepath()}'

        # Check existance of the file and the directory
        self._io.check_existence_on_local(dir, dir=True)
        self._io.check_existence_on_local(filepath, dir=False)

        clip_filename = f'crop_{image.product}_{image.year}_{image.tile}_{image.extract_date()}.tif'
        clip_image_ref = ImageRef(clip_filename, tile_ref=image.tile_ref, type='crop')
        clip_dir = f'{self._io.config.base_local_dir}/{clip_image_ref.rel_dir()}'
        clip_filepath = f'{self._io.config.base_local_dir}/{clip_image_ref.rel_filepath()}'

        # Check directory where will be saved exists
        self._io.check_existence_on_local(clip_dir, dir=True)
        # Check if file already exists. If so, warn of overwriting
        try:
            self._io.check_existence_on_local(clip_filepath, dir=False)
            warnings.warn(f'\nFile {clip_filepath} already exists. Overwriting it.')
        except FileNotFoundError:
            pass

        # Open raster and clip it
        raster_to_clip = gdal.Open(filepath)
        # Check SRS of the raster
        if not raster_to_clip.GetProjection():
            warnings.warn(f'\nRaster {filepath} does not have a SRS. Intersection will not be done.')
            return None
        gdal.Warp(
            clip_filepath,
            raster_to_clip,
            format='GTiff',
            cutlineDSName=self._io.config.aoi_path['shp'],
            cropToCutline=True,
            dstNodata=0,
            multithread=True,
            warpOptions=['NUM_THREADS=ALL_CPUS'],
            creationOptions=['COMPRESS=LZW'],
            callback=gdal.TermProgress_nocb
        )
        return clip_image_ref

    def run(self):
        """
        Intersect all images of all TileRefs (year + product + tile) with the Area of Interest (AOI).
        1. Create a list of all the images that still have to be intersected. List is filtered according to config.
        2. For each image to be intersected (if time_limit is not over):
            - Downloads the RAW image from the server if it's not available locally
            - Crops it and saves it locally (CROP step). If not possible (corrupted SRS), mark it as unsuccessful
            in records and continue with the next image.
            - Uploads the CROP image to the server
            - Deletes both the RAW and CROP images locally
            - Add image to the records
        3. Save the records to a CSV file
        """
        all_images_df = self._io.list_all_raw_files()[['year', 'tile', 'product', 'filename']]
        if self._config.filters['product']:
            all_images_df = all_images_df.loc[all_images_df['product'].isin(self._config.filters['product'])]
        if self._config.filters['year']:
            all_images_df = all_images_df.loc[all_images_df['year'].isin(self._config.filters['year'])]
        if self._config.filters['tile']:
            all_images_df = all_images_df.loc[all_images_df['tile'].isin(self._config.filters['tile'])]

        # Get all images that still have to be intersected
        intersected = self._records.loc[
            (self._records['from'] == "raw") & (self._records['to'] == "crop"),
            ['year', 'tile', 'product', 'filename_from']
        ].rename(columns={'filename_from': 'filename'})
        all_images_df = all_images_df.merge(intersected, how='left', indicator=True,
                                            on=['year', 'tile', 'product', 'filename'])
        to_intersect = all_images_df.loc[all_images_df['_merge'] == 'left_only',
        ['year', 'tile', 'product', 'filename']].sort_values(by=['year', 'tile', 'product', 'filename'])
        image_refs = [ImageRef(row.filename, row.year, row.tile, row.product, type='raw')
                      for row in to_intersect.itertuples()]

        print(f'Filters: {self._config.filters}')
        print(f'Intersecting {len(image_refs)} images with the AOI.\n')
        print(f'Time limit: {self._time_limit} minutes.')
        print(f'{len(all_images_df) - len(image_refs)} images have already been intersected.\n')

        start_timestamp, start_time = timestamp(), time.time()
        i = 0
        while i < len(image_refs) and time.time() - start_time < self._time_limit * 60:
            print(f' -- Processing image {i + 1} of {len(image_refs)} '
                  f'({round((i + 1) * 100 / len(image_refs), 2)}%). Time elapsed: {round(time.time() - start_time, 2) / 60}')
            image_ref = image_refs[i]

            # Download the image (if not available locally, handled by IO)
            self._io.download_file(image_ref)
            # Crop and save locally
            crop_image_ref = self.intersect(image_ref)
            # Delete raw image locally
            self._io.delete_local_file(image_ref)
            if crop_image_ref is None:
                record = ['raw', 'crop', image_ref.tile, image_ref.year, image_ref.product, timestamp(), image_ref.filename, None, 0]
                self._records.loc[len(self._records)] = record
                continue
            # Upload the cropped image to the server
            self._io.upload_file(crop_image_ref)
            # Delete cropped image locally
            self._io.delete_local_file(crop_image_ref)
            # Add image to the records
            record = ['raw', 'crop', image_ref.tile, image_ref.year, image_ref.product, timestamp(),
                      image_ref.filename, crop_image_ref.filename, 1]
            record_df = pd.DataFrame({k: [v] for (k, v) in zip(RECORDS_FILE_COLUMNS, record)})
            self._records = pd.concat([self._records, record_df], ignore_index=True)
            self._io.save_records(self._records)
            i += 1

        print(f'\nEnd of process. Time elapsed: {round((time.time() - start_time) / 60, 2)} minutes.')
        print(f'Processed {i} images during the execution.')
        unsuccessful = len(self._records.loc[
                               (self._records['success'] == 0) & (self._records['from'] == 'raw') &
                               (self._records['to'] == 'crop') &
                               self._records['timestamp'].between(start_timestamp, timestamp())])

        print(f'Unsuccessful intersections: {unsuccessful}')
