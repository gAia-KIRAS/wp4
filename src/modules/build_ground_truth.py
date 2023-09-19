import warnings

from osgeo import gdal
import geopandas as gpd
from config.config import Config
from io_manager.io_manager import IO
from modules.abstract_module import Module
from utils import ImageRef, reference_crop_images


class BuildGroundTruth(Module):
    """
    TODO: this module is not developed yet. Current version does not work.
    This module the ground truth images, which are pixel-wise images with 0/1 values, where 1 means that the pixel
    is part of a Polygon that represents a reported landslide. The name of the image is 'ground_truth_{tile}.tif'.
    """
    def __init__(self, io: IO, config: Config):
        super().__init__(config, io)

        assert self._config.execution_where == 'local', 'This module can only be executed locally.'

        self._check_inventory()

    def _check_inventory(self):
        """
        Checks if the inventory file exists locally as a .shp file.
        If not, it is loaded as .gpkg and saved as .shp file. If .gpkg does not exist, an error is thrown.
        """
        filepath_shp = self._io.config.inventory_path['shp']
        try:
            self._io.check_existence_on_local(filepath_shp, dir=False)
        except FileNotFoundError:
            filepath_gpkg = self._io.config.inventory_path['gpkg']

            # .gpkg should exist, otherwise throw an error
            self._io.check_existence_on_local(filepath_gpkg, dir=False)

            warnings.warn(f'\nInventory in .shp format did not exist. Creating it from {filepath_gpkg}.')
            inv = gpd.read_file(filepath_gpkg)
            inv.to_file(filepath_shp, driver='ESRI Shapefile')
            self._io.check_existence_on_local(filepath_shp, dir=False)

    def run(self, on_the_server: bool = False) -> None:
        """
        This module creates one image with the ground truth for each tile. This image has the exact same size as the NCI
        images and contains 0/1 values, where 1 means that the pixel is part of a Polygon that represents a
        reported landslide. The name of the image is 'ground_truth_{tile}.tif'.
        It can only be executed locally. The on_the_server argument is ignored. The module only works for
        NDVI_reconstructed images.

        Args:
            on_the_server (bool): if True, the module is being executed on the server.
        """
        print(' -- Building ground truth -- ')

        if set(self._config.filters['product']) and set(self._config.filters['product']) != {'NDVI_reconstructed'}:
            raise ValueError('This module can only be executed for NDVI_reconstructed images.')

        tiles_set, all_tiles_set = set(self._config.filters['tile']), set(self._io.config.available_tiles)
        to_define = all_tiles_set if not tiles_set else all_tiles_set & tiles_set
        for tile in to_define:
            gt_image = self._create_ground_truth_image(tile)
            self._io.upload_file(gt_image)

    def _create_ground_truth_image(self, tile: str) -> ImageRef:
        """
        Intersect the CROP images with the polygons of the ground truth and save the result as a new .tif file.

        Args:
            tile: tile of the image
            product: product of the image
        """

        image = ImageRef(reference_crop_images[tile], 2021, tile, 'NDVI_raw', 'crop')
        self._io.download_file(image)

        image_local_dir = f'{self._io.config.base_local_dir}/{image.rel_dir()}'
        filepath = f'{self._io.config.base_local_dir}/{image.rel_filepath()}'

        self._io.check_existence_on_local(image_local_dir, dir=True)
        self._io.check_existence_on_local(filepath, dir=False)

        gt_filename = f'gt_nodates_{tile}.tif'
        gt_image = ImageRef(gt_filename, None, tile, 'NDVI_reconstructed', 'ground_truth')

        gt_local_dir = f'{self._io.config.base_local_dir}/{gt_image.rel_dir()}'
        gt_filepath = f'{gt_local_dir}/{gt_image.filename}'

        # Check directory where the image will be saved exists
        self._io.check_existence_on_local(gt_local_dir, dir=True)
        # Check if file already exists. If so, warn of overwriting
        try:
            self._io.check_existence_on_local(gt_filepath, dir=False)
            warnings.warn(f'\nFile {gt_filepath} already exists. Overwriting it.')
        except FileNotFoundError:
            pass

        # Open raster and clip it
        raster_to_clip = gdal.Open(filepath)
        gdal.Warp(
            gt_filepath,
            raster_to_clip,
            format='GTiff',
            cutlineDSName=self._io.config.inventory_path['shp'],
            cropToCutline=True,
            dstNodata=0,
            multithread=True,
            warpOptions=['NUM_THREADS=ALL_CPUS'],
            creationOptions=['COMPRESS=LZW'],
            callback=gdal.TermProgress_nocb
        )
        return gt_image
