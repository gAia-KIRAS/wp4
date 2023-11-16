import pandas as pd

from utils import ImageRef, RAW_IMAGE_SIZES, CROP_IMAGE_LIMITS


def get_features_for_point(row, raw_images_df):
    i, j = row.i, row.j
    date = pd.to_datetime(row.detected_breakpoint, format='%Y-%m-%d')
    year = row.year
    tile = row.tile
    y = row.y

    filepath_nci = f'nci3_NDVIrec_{year}_{tile}_{date.strftime("%Y%m%d")}.tif'
    filepath_delta = f'delta_NDVIrec_{year}_{tile}_{date.strftime("%Y%m%d")}.tif'

    image_ref_nci = ImageRef(filepath_nci, year=year, tile=tile, product='NDVI_reconstructed', type='nci')
    image_ref_delta = ImageRef(filepath_delta, year=year, tile=tile, product='NDVI_reconstructed', type='delta')

    raw_image = raw_images_df.loc[
        (raw_images_df['year'] == image_ref_nci.year) &
        (raw_images_df['tile'] == image_ref_nci.tile) &
        (raw_images_df['product'] == image_ref_nci.product) &
        (raw_images_df['filename'].str.contains(image_ref_nci.extract_date()))
        ].iloc[0]
    image_ref_raw = ImageRef(raw_image.filename, raw_image.year, raw_image.tile, raw_image['product'],
                             type='raw')

    # Download images:
    if config.execution_where != 'server':
        io.download_file(image_ref_nci)
        io.download_file(image_ref_delta)
        io.download_file(image_ref_raw)

    nci = io.load_tif_as_ndarray(image_ref_nci)
    delta = io.load_tif_as_ndarray(image_ref_delta)
    raw = io.load_tif_as_ndarray(image_ref_raw)

    # Crop raw image
    raw = crop_image(raw, image_ref_raw)

    # Get features
    nci_values = nci[:, i, j]
    delta_values = delta[:, i, j]
    raw_ndvi = raw[i, j]

    # Build as list and return it
    features = []
    features.extend(nci_values)
    features.extend(delta_values)
    features += [raw_ndvi]

    return features


def crop_image(raster, image_ref: ImageRef):
    # Check dimensions
    assert raster.shape == RAW_IMAGE_SIZES.get(image_ref.tile), \
        f'Image {image_ref.filename} has wrong dimensions. Expected: {RAW_IMAGE_SIZES.get(image_ref.tile)}'

    # Get dimensions from utils map
    min_i, max_i, min_j, max_j = CROP_IMAGE_LIMITS.get(image_ref.tile)

    # Crop image
    raster = raster[min_i:max_i, min_j:max_j]

    return raster


def get_features():
    train_inv_path = f'{io.config.base_local_dir}/operation_records/train_inv.csv'
    train_inv = pd.read_csv(train_inv_path)

    raw_images_df = io.filter_all_images(image_type='raw', filters={})

    base_columns = ['i', 'j', 'year', 'tile', 'y']
    feature_names = [f'nci_{i}' for i in range(4)] + [f'delta_{i}' for i in range(4)] + ['raw']
    df = pd.DataFrame(columns=base_columns + feature_names)

    # Iterate over dataset and get features
    for index, row in enumerate(train_inv.itertuples()):
        if index > 2:
            continue
        print(f'Processing {index} of {len(train_inv)}')
        features = get_features_for_point(row, raw_images_df)
        new_row = pd.DataFrame([[row.i, row.j, row.year, row.tile, row.y] + features],
                               columns=base_columns + feature_names)
        df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(f'{io.config.base_local_dir}/operation_records/train_features.csv', index=False)


if __name__ == '__main__':
    from config.config import Config
    from io_manager.io_manager import IO
    from config.io_config import IOConfig

    config = Config()
    io_config = IOConfig()
    io = IO(io_config)

    get_features()
