from config.config import Config
from config.io_config import IOConfig
from io_manager.io_manager import IO
from modules.nci import NCI
from utils import TileRef
import gdal
from utils import *
import numpy as np


def f(tile):
    config = Config()
    io_config = IOConfig()
    io = IO(io_config)
    nci = NCI(config, io)

    # Load the raw NDVI_rec image and its next
    tile_ref = TileRef(2020, tile, 'NDVI_reconstructed')
    image_refs, _ = io.list_files_on_server(tile_ref, image_type='raw')
    image_1 = image_refs[0]
    image_2 = image_refs[1]

    io.download_file(image_1)
    io.download_file(image_2)

    r_1 = gdal.Open(f'{io_config.base_local_dir}/{image_1.rel_filepath()}').ReadAsArray()
    original_shape = r_1.shape

    # Load the raw NDVI_raw image
    tile_ref_raw = TileRef(2021, tile, 'NDVI_raw')
    image_refs, _ = io.list_files_on_server(tile_ref_raw, image_type='raw')
    image_1_raw = image_refs[0]
    io.download_file(image_1_raw)

    r_1_raw = gdal.Open(f'{io_config.base_local_dir}/{image_1_raw.rel_filepath()}').ReadAsArray()

    # Load the cropped NDVI_raw image
    filename = {
        '33TVM': "crop_NDVIraw_2021_33TVM_20210105.tif",
        '33TUN': "crop_NDVIraw_2021_33TUN_20210103.tif",
        '33TUM': "crop_NDVIraw_2021_33TUM_20210105.tif",
        '33TVN': "crop_NDVIraw_2021_33TVN_20210105.tif",
    }
    image_1_crop = ImageRef(
        filename=filename[tile],
        tile=tile,
        year=2021,
        product='NDVI_raw',
        type='crop',
    )
    io.download_file(image_1_crop)

    r_1_crop = gdal.Open(f'{io_config.base_local_dir}/{image_1_crop.rel_filepath()}').ReadAsArray()

    assert original_shape == RAW_IMAGE_SIZES[tile], \
        f'Image shape of NDVIrec does not match expected shape {RAW_IMAGE_SIZES[tile]}'
    assert original_shape == r_1_raw.shape, \
        f'Image shape of NDVIraw does not match expected shape {r_1_raw.shape}'

    example = np.array([[np.nan, np.nan, np.nan], [np.nan, 0, np.nan], [np.nan, np.nan, np.nan]])
    j_max, j_min, i_max, i_min = get_min_max_row_col(r_1_crop)
    print(f'i_min: {i_min}, i_max: {i_max}, j_min: {j_min}, j_max: {j_max}')

    r_1_new_crop = r_1_crop[i_min:i_max, j_min:j_max]

    # Compute the NCI
    print('NCI will be computed here')
    nci_image = nci.compute_and_save_nci(image_1, image_2)

    # Load the NCI image
    r_nci = gdal.Open(f'{io_config.base_local_dir}/{nci_image.rel_filepath()}').ReadAsArray()

    n_bands = r_nci.shape[0]
    n_rows = r_nci.shape[1]
    n_cols = r_nci.shape[2]

    assert n_bands == 4 and n_rows == r_1_new_crop.shape[0] and n_cols == r_1_new_crop.shape[1], \
        f'Image shape of NCI does not match expected shape {r_1_new_crop.shape}'

    print(f'Success for tile {tile}')


def search_sequence_numpy(arr, seq):
    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na - Nseq + 1)[:, None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any() > 0:
        return np.where(np.convolve(M, np.ones((Nseq), dtype=int)) > 0)[0]
    else:
        return []  # No match found


def get_min_max_row_col(raster):
    r = raster.copy()
    # Study the r_1_crop image to identify the crop. The values outside the crop are nan.

    # Define function that returns one if all values in the array are nan, and zero otherwise
    def is_nan_array(x):
        return 1 if (abs(x) < 1e-5).all() else 0

    # Apply the function to each row and column
    row_sums = np.apply_along_axis(is_nan_array, axis=1, arr=r)
    col_sums = np.apply_along_axis(is_nan_array, axis=0, arr=r)

    # Find the first row and column that are not nan
    j_min = np.where(col_sums == 0)[0][0]
    i_min = np.where(row_sums == 0)[0][0]

    # Reverse the array and find the first row and column that are not nan
    j_max = r.shape[1] - np.where(col_sums[::-1] == 0)[0][0]
    i_max = r.shape[0] - np.where(row_sums[::-1] == 0)[0][0]

    return j_max, j_min, i_max, i_min


if __name__ == '__main__':
    """
    This is a script used for developing purposes. Won't be included in the final code and it is not relevant for the
    main pipeline
    """

    f('33TUM')
    # f('33TVM')
    # f('33TUN')
    # f('33TVN')
