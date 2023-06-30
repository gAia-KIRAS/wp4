from dataclasses import dataclass
from datetime import datetime

RECORDS_FILE_COLUMNS = ['from', 'to', 'tile', 'year', 'product', 'timestamp', 'filename_from', 'filename_to', 'success']


@dataclass
class TileRef:
    """
    Class to store the reference to a tile. No date is stored.
    """
    year: int
    tile: str
    product: str

    def to_subpath(self):
        return f'{self.year}/{self.tile}/{self.product}'

    def __str__(self):
        return f'Year: {self.year} | Tile: {self.tile} | Product: {self.product}'


@dataclass
class ImageRef:
    """
    Class to store the reference to an image.

    Attributes:
        year: year of the data
        tile: Sentinel tile
        product: product type. Can be ['NDVI_raw', 'B02', 'B03', 'B04', 'B08', 'B11', 'SCL']
        type: (optional) type of the image. Can be ['raw', 'crop']
        tile_ref: (optional) TileRef object with the reference to the tile.
        If not set, year, tile and product must be set.
    """

    filename: str
    year: int = None
    tile: str = None
    product: str = None
    type: str = None
    tile_ref: TileRef = None

    def __post_init__(self):
        if self.tile_ref:
            if self.year or self.tile or self.product:
                raise Exception('TileRef and (year or tile or product) cannot be set at the same time.')
            self.year = self.tile_ref.year
            self.tile = self.tile_ref.tile
            self.product = self.tile_ref.product
        else:
            if not self.year or not self.tile or not self.product:
                raise Exception('year, tile and product must be set if TileRef is not set.')
            self.tile_ref = TileRef(self.year, self.tile, self.product)

    def rel_filepath(self) -> str:
        """
        Build the relative filepath of the image: {type}/{product}/{year}/{product}/{filename}.
        Examples:
        - raw/NDVI_raw/2019/33TUM/33_T_UM_2021_10_S2A_33TUM_20211010_0_L2A_NDVI.tif
        - crop/B03/2018/33TUN/33_T_UM_2021_10_S2A_33TUM_20211010_0_L2A_NDVI.tif

        Returns:
            string with the relative filepath
        """
        if not self.type:
            raise Exception('Type of image not set. Relative filepath cannot be built.')
        return f'{self.type}/{self.tile_ref.to_subpath()}/{self.filename}'

    def rel_dir(self) -> str:
        """
        Same as rel_filepath, but just build the relative directory of the image:
         {type}/{product}/{year}/{product}. Examples:
        - raw/NDVI_raw/2019/33TUM
        - crop/B03/2018/33TUN

        Returns:
            string with the relative directory
        """
        if not self.type:
            raise Exception('Type of image not set. Relative directory cannot be built.')
        return f'{self.type}/{self.tile_ref.to_subpath()}'

    def __str__(self):
        return f'( Filename: {self.filename} | Year: {self.year} | Tile: {self.tile} | ' \
               f'Product: {self.product} | Type: {self.type} )'

    def extract_date(self):
        """
        Extract date from the filepath.
        """
        if self.type is None:
            raise Exception('Type of image not set. Date cannot be extracted.')
        if self.type == 'raw':
            date = self.filename.split('_')[7]
        elif self.type == 'crop':
            date = self.filename.split('_')[-1]
        else:
            raise Exception("Type of image not in ['raw', 'crop']. Date cannot be extracted.")

        # Check it is an appropriate date
        if len(date) != 8 or not date.isdigit():
            raise Exception(f'Could not extract date. '
                            f'Filename: {self.filename}, type: {self.type}, extracted date: {date}')
        return date


def timestamp() -> str:
    """
    Return a timestamp in the format YYYYMMDD_HHMMSS.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")
