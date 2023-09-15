from dataclasses import dataclass
from datetime import datetime
from typing import Union

RECORDS_FILE_COLUMNS = ['from', 'to', 'tile', 'year', 'product', 'timestamp', 'filename_from', 'filename_to', 'success']
RECORDS_CD_FILE_COLUMNS = ['cd_id', 'tile', 'subtile', 'timestamp']
RESULTS_CD_FILE_COLUMNS = ['cd_id', 'tile', 'subtile', 'i', 'j', 'timestamp', 'detected_breakpoint', 'subproduct']
IMAGE_TYPES = ['raw', 'crop', 'nci', 'testing', 'ground_truth']

RAW_IMAGE_SIZES = {
    '33TUM': (10980, 10980),
    '33TUN': (10980, 10980),
    '33TVM': (10980, 10980),
    '33TVN': (10980, 10980),
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
    '33TVM': "crop_NDVIraw_2021_33TVM_20210105.tif",
    '33TUN': "crop_NDVIraw_2021_33TUN_20210103.tif",
    '33TUM': "crop_NDVIraw_2021_33TUM_20210105.tif",
    '33TVN': "crop_NDVIraw_2021_33TVN_20210105.tif",
}

reference_nci_images = {
    '33TUM': 'nci3_NDVIrec_2020_33TUM_20200101.tif',
    '33TUN': 'nci3_NDVIrec_2020_33TUN_20200101.tif',
    '33TVM': 'nci3_NDVIrec_2020_33TVM_20200101.tif',
    '33TVN': 'nci3_NDVIrec_2020_33TVN_20200101.tif'
}

subtiles = {'33TUM_0': [0, 1000, 0, 1000], '33TUM_1': [0, 1000, 1000, 2000], '33TUM_2': [0, 1000, 2000, 3000],
            '33TUM_3': [0, 1000, 3000, 4000], '33TUM_4': [0, 1000, 4000, 5000], '33TUM_5': [0, 1000, 5000, 6000],
            '33TUM_6': [0, 1000, 6000, 7000], '33TUM_7': [0, 1000, 7000, 8000], '33TUM_8': [1000, 2000, 0, 1000],
            '33TUM_9': [1000, 2000, 1000, 2000], '33TUM_10': [1000, 2000, 2000, 3000],
            '33TUM_11': [1000, 2000, 3000, 4000], '33TUM_12': [1000, 2000, 4000, 5000],
            '33TUM_13': [1000, 2000, 5000, 6000], '33TUM_14': [1000, 2000, 6000, 7000],
            '33TUM_15': [1000, 2000, 7000, 8000], '33TUM_16': [2000, 3000, 0, 1000],
            '33TUM_17': [2000, 3000, 1000, 2000], '33TUM_18': [2000, 3000, 2000, 3000],
            '33TUM_19': [2000, 3000, 3000, 4000], '33TUM_20': [2000, 3000, 4000, 5000],
            '33TUM_21': [2000, 3000, 5000, 6000], '33TUM_22': [2000, 3000, 6000, 7000],
            '33TUM_23': [2000, 3000, 7000, 8000], '33TUM_24': [3000, 4000, 0, 1000],
            '33TUM_25': [3000, 4000, 1000, 2000], '33TUM_26': [3000, 4000, 2000, 3000],
            '33TUM_27': [3000, 4000, 3000, 4000], '33TUM_28': [3000, 4000, 4000, 5000],
            '33TUM_29': [3000, 4000, 5000, 6000], '33TUM_30': [3000, 4000, 6000, 7000],
            '33TUM_31': [3000, 4000, 7000, 8000], '33TUN_0': [0, 1000, 0, 1000], '33TUN_1': [0, 1000, 1000, 2000],
            '33TUN_2': [0, 1000, 2000, 3000], '33TUN_3': [0, 1000, 3000, 4000], '33TUN_4': [0, 1000, 4000, 5000],
            '33TUN_5': [0, 1000, 5000, 6000], '33TUN_6': [0, 1000, 6000, 7000], '33TUN_7': [0, 1000, 7000, 8000],
            '33TUN_8': [0, 1000, 8000, 9000], '33TUN_9': [1000, 2000, 0, 1000], '33TUN_10': [1000, 2000, 1000, 2000],
            '33TUN_11': [1000, 2000, 2000, 3000], '33TUN_12': [1000, 2000, 3000, 4000],
            '33TUN_13': [1000, 2000, 4000, 5000], '33TUN_14': [1000, 2000, 5000, 6000],
            '33TUN_15': [1000, 2000, 6000, 7000], '33TUN_16': [1000, 2000, 7000, 8000],
            '33TUN_17': [1000, 2000, 8000, 9000], '33TUN_18': [2000, 3000, 0, 1000],
            '33TUN_19': [2000, 3000, 1000, 2000], '33TUN_20': [2000, 3000, 2000, 3000],
            '33TUN_21': [2000, 3000, 3000, 4000], '33TUN_22': [2000, 3000, 4000, 5000],
            '33TUN_23': [2000, 3000, 5000, 6000], '33TUN_24': [2000, 3000, 6000, 7000],
            '33TUN_25': [2000, 3000, 7000, 8000], '33TUN_26': [2000, 3000, 8000, 9000],
            '33TUN_27': [3000, 4000, 0, 1000], '33TUN_28': [3000, 4000, 1000, 2000],
            '33TUN_29': [3000, 4000, 2000, 3000], '33TUN_30': [3000, 4000, 3000, 4000],
            '33TUN_31': [3000, 4000, 4000, 5000], '33TUN_32': [3000, 4000, 5000, 6000],
            '33TUN_33': [3000, 4000, 6000, 7000], '33TUN_34': [3000, 4000, 7000, 8000],
            '33TUN_35': [3000, 4000, 8000, 9000], '33TVM_0': [0, 1000, 0, 1000], '33TVM_1': [0, 1000, 1000, 2000],
            '33TVM_2': [0, 1000, 2000, 3000], '33TVM_3': [0, 1000, 3000, 4000], '33TVM_4': [0, 1000, 4000, 5000],
            '33TVM_5': [0, 1000, 5000, 6000], '33TVM_6': [0, 1000, 6000, 7000], '33TVM_7': [0, 1000, 7000, 8000],
            '33TVM_8': [0, 1000, 8000, 9000], '33TVM_9': [0, 1000, 9000, 10000], '33TVM_10': [0, 1000, 10000, 11000],
            '33TVM_11': [1000, 2000, 0, 1000], '33TVM_12': [1000, 2000, 1000, 2000],
            '33TVM_13': [1000, 2000, 2000, 3000], '33TVM_14': [1000, 2000, 3000, 4000],
            '33TVM_15': [1000, 2000, 4000, 5000], '33TVM_16': [1000, 2000, 5000, 6000],
            '33TVM_17': [1000, 2000, 6000, 7000], '33TVM_18': [1000, 2000, 7000, 8000],
            '33TVM_19': [1000, 2000, 8000, 9000], '33TVM_20': [1000, 2000, 9000, 10000],
            '33TVM_21': [1000, 2000, 10000, 11000], '33TVM_22': [2000, 3000, 0, 1000],
            '33TVM_23': [2000, 3000, 1000, 2000], '33TVM_24': [2000, 3000, 2000, 3000],
            '33TVM_25': [2000, 3000, 3000, 4000], '33TVM_26': [2000, 3000, 4000, 5000],
            '33TVM_27': [2000, 3000, 5000, 6000], '33TVM_28': [2000, 3000, 6000, 7000],
            '33TVM_29': [2000, 3000, 7000, 8000], '33TVM_30': [2000, 3000, 8000, 9000],
            '33TVM_31': [2000, 3000, 9000, 10000], '33TVM_32': [2000, 3000, 10000, 11000],
            '33TVM_33': [3000, 4000, 0, 1000], '33TVM_34': [3000, 4000, 1000, 2000],
            '33TVM_35': [3000, 4000, 2000, 3000], '33TVM_36': [3000, 4000, 3000, 4000],
            '33TVM_37': [3000, 4000, 4000, 5000], '33TVM_38': [3000, 4000, 5000, 6000],
            '33TVM_39': [3000, 4000, 6000, 7000], '33TVM_40': [3000, 4000, 7000, 8000],
            '33TVM_41': [3000, 4000, 8000, 9000], '33TVM_42': [3000, 4000, 9000, 10000],
            '33TVM_43': [3000, 4000, 10000, 11000], '33TVN_0': [0, 1000, 0, 1000], '33TVN_1': [0, 1000, 1000, 2000],
            '33TVN_2': [0, 1000, 2000, 3000], '33TVN_3': [0, 1000, 3000, 4000], '33TVN_4': [0, 1000, 4000, 5000],
            '33TVN_5': [0, 1000, 5000, 6000], '33TVN_6': [0, 1000, 6000, 7000], '33TVN_7': [0, 1000, 7000, 8000],
            '33TVN_8': [0, 1000, 8000, 9000], '33TVN_9': [0, 1000, 9000, 10000], '33TVN_10': [0, 1000, 10000, 11000],
            '33TVN_11': [1000, 2000, 0, 1000], '33TVN_12': [1000, 2000, 1000, 2000],
            '33TVN_13': [1000, 2000, 2000, 3000], '33TVN_14': [1000, 2000, 3000, 4000],
            '33TVN_15': [1000, 2000, 4000, 5000], '33TVN_16': [1000, 2000, 5000, 6000],
            '33TVN_17': [1000, 2000, 6000, 7000], '33TVN_18': [1000, 2000, 7000, 8000],
            '33TVN_19': [1000, 2000, 8000, 9000], '33TVN_20': [1000, 2000, 9000, 10000],
            '33TVN_21': [1000, 2000, 10000, 11000], '33TVN_22': [2000, 3000, 0, 1000],
            '33TVN_23': [2000, 3000, 1000, 2000], '33TVN_24': [2000, 3000, 2000, 3000],
            '33TVN_25': [2000, 3000, 3000, 4000], '33TVN_26': [2000, 3000, 4000, 5000],
            '33TVN_27': [2000, 3000, 5000, 6000], '33TVN_28': [2000, 3000, 6000, 7000],
            '33TVN_29': [2000, 3000, 7000, 8000], '33TVN_30': [2000, 3000, 8000, 9000],
            '33TVN_31': [2000, 3000, 9000, 10000], '33TVN_32': [2000, 3000, 10000, 11000]}

assert all([
    x[1] - x[0] == CROP_LIMITS_INSIDE_CROPPED[tile][1] - CROP_LIMITS_INSIDE_CROPPED[tile][0] and
    x[3] - x[2] == CROP_LIMITS_INSIDE_CROPPED[tile][3] - CROP_LIMITS_INSIDE_CROPPED[tile][2]
    for tile, x in CROP_IMAGE_LIMITS.items()]), 'CROP_LIMITS_INSIDE_CROPPED must be consistent with RAW_IMAGE_SIZES'


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
        type: (optional) type of the image. Can be ['raw', 'crop']. Also 'testing' for testing purposes.
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
        elif self.type in ['crop', 'nci']:
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
