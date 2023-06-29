import geopandas as gpd
import warnings
from osgeo import gdal

from src.config.config import Config
from src.config.io_config import IOConfig
from src.io_manager import IO
from src.utils import ImageRef, TileRef
from typing import Union


class IntersectAOI:
    """
    This module does the step: RAW -> CROP
    It intersects the Area of Interest (AOI) with the raw images and saves the result as a new .tif file.

    Attributes:
        _io (IO): IO object
        _config (Config): Config object
    """
    def __init__(self, io: IO, config: Config):
        self._io = io
        self._config = config
        self._check_aoi()

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

    def intersect_tile_ref(self, image_type: str, tile_ref: TileRef) -> None:
        """
        Intersect all images of a specific TileRef (year + product + tile) with the Area of Interest (AOI).
        For each image:
            - Downloads the RAW image from the server if it's not available locally
            - Crops it and saves it locally (CROP)
            - Uploads the CROP image to the server
            - Deletes both the RAW and CROP images locally

        Args:
            image_type (str): type of the image to intersect. Can be 'raw' or 'crop'
            tile_ref (TileRef): TileRef object with the tile to intersect. Includes year, product, tile.
        """
        self._io.check_inputs_with_metadata(tile_ref)

        # Get all images of the tile reference
        image_refs, df = self._io.list_sentinel_files(tile_ref)

        print(f'Intersecting {len(image_refs)} images of type {image_type} for {tile_ref} with the AOI.\n')

        unsuccesful_intersections = []
        for i, image_ref in enumerate(image_refs):
            print(f' -- Processing image {i + 1} of {len(image_refs)} ({round((i + 1) * 100 / len(image_refs), 2)}%)')
            # Download the image (if not available locally, handled by IO)
            self._io.download_file(image_ref)
            # Crop and save locally
            crop_image_ref = self.intersect(image_ref)
            if crop_image_ref is None:
                unsuccesful_intersections.append(image_ref)
                continue
            # Upload the cropped image to the server
            self._io.upload_file(crop_image_ref)
            # Delete both the raw and the cropped image locally
            self._io.delete_local_file(image_ref)
            self._io.delete_local_file(crop_image_ref)

        print(f' -- Processed 100% of the images.')
        print(f' -- {len(unsuccesful_intersections)} '
              f'({round(len(unsuccesful_intersections) * 100 / len(image_refs), 2)}%) '
              f'images could not be intersected with the AOI.')
        print(f' -- {len(image_refs) - len(unsuccesful_intersections)} images were intersected with the AOI.')


if __name__ == '__main__':
    io_config = IOConfig()
    config = Config()

    io = IO(io_config)
    # intersect = IntersectAOI(io, config)
    # image = ImageRef(
    #     '33_T_UM_2021_10_S2A_33TUM_20211010_0_L2A_NDVI.tif',
    #     tile="33TUM", product="NDVI_raw", type='raw', year=2021)
    # intersect.intersect(image)

    iaoi = IntersectAOI(io, config)

    tile_ref = TileRef(2021, '33TUM', 'NDVI_raw')
    iaoi.intersect_tile_ref('raw', tile_ref)
