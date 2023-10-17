from config.config import Config
from config.io_config import IOConfig
from io_manager.io_manager import IO
from modules.nci import NCI
from utils import TileRef
from osgeo import osr, ogr, gdal
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

    image_1_crop = ImageRef(
        filename=reference_crop_images[tile],
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


def g(tile):
    config = Config()
    io_config = IOConfig()
    io = IO(io_config)

    tile_ref = TileRef(2020, tile, 'NDVI_reconstructed')
    crop_ref = ImageRef('crop_NDVIrec_2020_33TUM_20200101.tif', tile_ref=tile_ref, type='crop')
    cprob_ref = ImageRef('cprob_NDVIrec_2020_33TUM_20200111.tif', tile_ref=tile_ref, type='cprob')

    crop = io.load_tif_as_ndarray(crop_ref)
    aux = io.load_tif_as_ndarray(cprob_ref)
    original_shape = crop.shape

    # Get dimensions from utils map
    min_i, max_i, min_j, max_j = CROP_LIMITS_INSIDE_CROPPED.get(cprob_ref.tile)

    # Create empty raster with the size of the original raster
    raster = np.empty(original_shape, dtype=np.float32)
    raster.fill(np.nan)

    # Fill image with the cprob values
    raster[min_i:max_i, min_j:max_j] = aux

    assert raster.shape == crop.shape, 'Shape of raster and crop must be the same'

    filepath_aux = r'C:\Users\jsalva\PycharmProjects\wp4\data\testing\test_NDVIrec_2020_33TUM_20200101_aux.tif'
    filepath = r'C:\Users\jsalva\PycharmProjects\wp4\data\testing\test_NDVIrec_2020_33TUM_20200101.tif'

    # Save raster with the same CRS as the crop raster using gdal
    data = gdal.Open(f'{io_config.base_local_dir}/{crop_ref.rel_filepath()}', gdal.GA_ReadOnly)
    srs = data.GetProjection()
    geoTransform = data.GetGeoTransform()
    if False:
        # Write image
        driver = gdal.GetDriverByName('GTiff')
        n_bands, rows, cols = raster.shape if raster.ndim == 3 else (1, *raster.shape)
        dataset = driver.Create(filepath_aux, cols, rows, n_bands, gdal.GDT_Float32)
        dataset.SetProjection(srs)
        dataset.SetGeoTransform(geoTransform)
        for i in range(n_bands):
            band = dataset.GetRasterBand(i + 1)
            band.SetNoDataValue(np.nan)
            band.WriteArray(raster[i]) if raster.ndim == 3 else band.WriteArray(raster)
        dataset.FlushCache()
        dataset = None

        # Use gdal translate to compress the image using LZW
        gdal.Translate(filepath, filepath_aux, options='-co COMPRESS=LZW')

    def pixel_to_world(geo_matrix, x, y):
        return x * geo_matrix[1] + geo_matrix[0], y * geo_matrix[5] + geo_matrix[3]

    def build_transform_inverse(dataset, EPSG):
        source = osr.SpatialReference(wkt=dataset.GetProjection())
        target = osr.SpatialReference()
        target.ImportFromEPSG(EPSG)
        return osr.CoordinateTransformation(source, target)

    def find_spatial_coordinate_from_pixel(dataset, transform, x, y):
        world_x, world_y = pixel_to_world(dataset.GetGeoTransform(), x, y)
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(world_x, world_y)
        point.Transform(transform)
        return point.GetX(), point.GetY()

    _t = build_transform_inverse(data, 4326)
    point = (min_j, min_i)
    # point = (0, 0)
    # point = CROP_IMAGE_SIZES.get(cprob_ref.tile)
    coordinates = find_spatial_coordinate_from_pixel(data, _t, point[0], point[1])
    print(coordinates)

    return 0


if __name__ == '__main__':
    """
    This is a script used for developing purposes. Won't be included in the final code and it is not relevant for the
    main pipeline
    """

    # f('33TUM')
    # f('33TVM')
    # f('33TUN')
    # f('33TVN')
    #
    g('33TUM')
