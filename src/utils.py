from dataclasses import dataclass
from datetime import datetime
from typing import Union

RECORDS_FILE_COLUMNS = ['from', 'to', 'tile', 'year', 'product', 'timestamp', 'filename_from', 'filename_to', 'success']
RECORDS_CD_FILE_COLUMNS = ['cd_id', 'threshold', 'tile', 'year', 'filename_from', 'filename_to', 'detected_events',
                           'timestamp']
RESULTS_CD_FILE_COLUMNS = ['version', 'threshold', 'tile', 'year', 'row', 'column', 'date', 'probability', 'timestamp',
                           'lat', 'lon']
IMAGE_TYPES = ['raw', 'crop', 'nci', 'testing', 'delta', 'cprob']

RAW_IMAGE_SIZES = {
    '33TUM': (10980, 10980),
    '33TUN': (10980, 10980),
    '33TVM': (10980, 10980),
    '33TVN': (10980, 10980),
}
CROP_IMAGE_SIZES = {
    '33TUM': (6139, 18280),
    '33TUN': (6139, 18280),
    '33TVM': (6139, 18280),
    '33TVN': (6139, 18280),
}
CROP_IMAGE_LIMITS = {  # (i_min, i_max, j_min, j_max)
    '33TUM': (0, 3463, 3131, 10980),
    '33TUN': (10980 - 3207, 10980, 10980 - 8755, 10980),
    '33TVM': (0, 3910, 0, 10509),
    '33TVN': (10980 - 2385, 10980, 0, 10223),
}
CROP_LIMITS_INSIDE_CROPPED = {  # (i_min, i_max, j_min, j_max)
    '33TUM': (2229, 5692, 906, 8755),
    '33TUN': (0, 3207, 0, 8755),
    '33TVM': (0, 3910, 0, 10509),
    '33TVN': (822, 3207, 7771, 17994),
}

reference_crop_images = {
    '33TVM': "crop/2021/33TVM/NDVI_raw/crop_NDVIraw_2021_33TVM_20210105.tif",
    '33TUN': "crop/2021/33TUN/NDVI_raw/crop_NDVIraw_2021_33TUN_20210103.tif",
    '33TUM': "crop/2021/33TUM/NDVI_raw/crop_NDVIraw_2021_33TUM_20210105.tif",
    '33TVN': "crop/2021/33TVN/NDVI_raw/crop_NDVIraw_2021_33TVN_20210105.tif",
}

reference_nci_images = {
    '33TUM': 'nci3_NDVIrec_2020_33TUM_20200101.tif',
    '33TUN': 'nci3_NDVIrec_2020_33TUN_20200101.tif',
    '33TVM': 'nci3_NDVIrec_2020_33TVM_20200101.tif',
    '33TVN': 'nci3_NDVIrec_2020_33TVN_20200101.tif'
}

coefficients_log_reg = {'intercept': 0.050342878586550385, 'nci_0': 0.09228724922071369, 'nci_1': 0.06389867017590802,
                        'nci_2': -6.909563461222256e-05, 'nci_3': -0.003436187944866737, 'delta_0': 0.03473757739972183,
                        'delta_1': -0.012445055651254186, 'delta_2': 0.019858978907541493,
                        'delta_3': 0.013485552057842085, 'delta_4': 0.0042371712598931474}

assert all([
    x[1] - x[0] == CROP_LIMITS_INSIDE_CROPPED[tile][1] - CROP_LIMITS_INSIDE_CROPPED[tile][0] and
    x[3] - x[2] == CROP_LIMITS_INSIDE_CROPPED[tile][3] - CROP_LIMITS_INSIDE_CROPPED[tile][2]
    for tile, x in CROP_IMAGE_LIMITS.items()]), 'CROP_LIMITS_INSIDE_CROPPED must be consistent with CROP_IMAGE_LIMITS'


@dataclass
class TileRef:
    """
    Class to store the reference to a tile. No date is stored.
    """
    year: int = None
    tile: str = None
    product: str = None

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
        type: (optional) type of the image. Can be  Also 'testing' for testing purposes.
        tile_ref: (optional) TileRef object with the reference to the tile.
        If not set, year, tile and product must be set.
    """

    filename: str
    year: Union[int, None] = None
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
            self.tile_ref = TileRef(self.year, self.tile, self.product)
        if self.type and self.type not in IMAGE_TYPES:
            raise Exception(f'Image type {self.type} not supported.')

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
         {type}/{year}/{tile}/{product}. Examples:
        - raw/2019/33TUM/NDVI_raw
        - crop/2018/33TUN/B03

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
        if self.type == 'raw' and self.product == 'NDVI_reconstructed':
            date = self.filename.split('_')[2]
        elif self.type == 'raw':
            date = self.filename.split('_')[7]
        elif self.type in ['crop', 'nci', 'delta', 'cprob']:
            date = self.filename.split('_')[-1].split('.')[0]
        else:
            raise Exception("Type of image not in ['raw', 'crop']. Date cannot be extracted.")

        # Check it is an appropriate date
        if len(date) != 8 or not date.isdigit():
            raise Exception(f'Could not extract date. '
                            f'Filename: {self.filename}, type: {self.type}, extracted date: {date}')
        return date


@dataclass
class FakeTFTypeHints:
    """
    Class to store fake type hints for tensorflow when it is not installed.
    """
    Tensor = None
    convert_to_tensor = None
    subtract = None
    multiply = None
    sqrt = None
    square = None
    divide = None
    float32 = None
    stack = None
    ones = None
    scalar_mul = None
    nn = None
    reshape = None
    squeeze = None


def timestamp() -> str:
    """
    Return a timestamp in the format YYYYMMDD_HHMMSS.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


rename_product = {
    'NDVI_raw': 'NDVIraw',
    'NDVI_reconstructed': 'NDVIrec',
}
