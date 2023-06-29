import geopandas as gpd
import warnings
from osgeo import gdal

from src.config.config import Config
from src.config.io_config import IOConfig
from src.io_manager import IO
from src.utils import ImageRef


class IntersectAOI:
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

            warnings.warn(f'AOI in .shp format did not exist. Creating it from {filepath_gpkg}.')
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

    def intersect(self, image: ImageRef):
        """
        Intersects a local .tif file referenced by image (ImageRef) with the Area of Interest AOI.
        Saves the result locally in a new .tif file with the following name:
        - 'crop_{image.product}_{image.year}_{imgae.tile}_{image.date}.tif'

        Args:
            image: ImageRef object with the image to intersect. Needs to have the attribute type set.
        """
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
        except FileNotFoundError:
            warnings.warn(f'File {clip_filepath} already exists. Overwriting it.')

        # Open raster and clip it
        raster_to_clip = gdal.Open(filepath)
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


if __name__ == '__main__':
    io_config = IOConfig()
    config = Config()

    io = IO(io_config)
    intersect = IntersectAOI(io, config)
    image = ImageRef(
        '33_T_UN_2021_10_S2A_33TUN_20211017_0_L2A_NDVI.tif',
        tile="33TUN", product="NDVI_raw", type='raw', year=2021)
    intersect.intersect(image)
