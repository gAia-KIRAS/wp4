import geopandas as gpd
import rasterio
from rasterio.mask import mask

from src.config.config import Config
from src.config.io_config import IOConfig
from src.io_manager import IO


class IntersectAOI:
    def __init__(self, io: IO, config: Config):
        self._io = io
        self._config = config
        self._aoi = self.read_aoi()

    def read_aoi(self) -> gpd.GeoDataFrame:
        """
        Reads the Area of Interest (AOI) from the local file. File format is .gpkg and is saved as a GeoDataFrame.

        Returns:
            gdf (GeoDataFrame): GeoDataFrame with the AOI
        """
        filepath = self._io.config.aoi_path
        self._io.check_existence_on_local(filepath, dir=False)
        gdf = gpd.read_file(filepath)
        return gdf

    def intersect(self, year: int, tile: str, product: str, filename: str):
        """
        Intersects a local .tif file with the Area of Interest AOI.
        Saves the result locally in a new .tif file.

        Args:
            product: product type
            tile: tile of the Sentinel image
            year: year of the Sentinel image
            filename (str): name of the file to intersect with the AOI
        """
        dir = f'{self._io.config.base_local_dir}/raw/{year}/{tile}/{product}'
        filepath = f'{dir}/{filename}'

        # Check existance of the file and the directory
        self._io.check_existence_on_local(dir, dir=True)
        self._io.check_existence_on_local(filepath, dir=False)

        profile = {'driver': 'GTiff', 'height': 1, 'width': 10980, 'count': 1, 'dtype': rasterio.uint8}
        with rasterio.open(filepath, mode='w', **profile) as raster:
            out_image, out_transform = mask(raster, self._aoi.geometry, crop=True)
            out_meta = raster.meta.copy()

        # raster = rasterio.open(filepath, crs=)
        # out_image, out_transform = mask(raster, self._aoi.geometry, crop=True)
        # out_meta = raster.meta.copy()

        # Save the resulting raster in a new file
        # out_meta.update({"driver": "GTiff",
        #                  "height": out_image.shape[1],
        #                  "width": out_image.shape[2],
        #                  "transform": out_transform})


if __name__ == '__main__':
    io_config = IOConfig()
    config = Config()

    io = IO(io_config)
    intersect = IntersectAOI(io, config)
    intersect.intersect(2021, '33TUN', 'NDVI_raw', '33_T_UN_2021_10_S2A_33TUN_20211017_0_L2A_NDVI.tif')
