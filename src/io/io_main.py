from src.config.io_config import IOConfig
from src.io.io_manager import IO
from src.utils import TileRef, ImageRef

import pandas as pd
import pytest
import shutil


def check_dates():
    """
    Check if for a particular year, the dates of the images are the same for all tiles.
    """
    io_config = IOConfig()
    io = IO(io_config)
    tiles = io_config.available_tiles
    years = io_config.available_years
    dates = {}
    for year in years:
        for tile in tiles:
            tile_ref = TileRef(year, tile, 'NDVI_raw')
            refs, df = io.list_sentinel_files(tile_ref)
            dates[(year, tile)] = set(df.date_f.dt.strftime('%Y-%m-%d'))

    for year in years:
        for tile in tiles:
            if dates[(year, tile)] != dates[(year, tiles[0])]:
                print(f'For year {year} and tile {tile}, the dates are different from the other tiles')
                print(f'Dates in {tile} not in {tiles[0]}: {sorted(dates[(year, tile)] - dates[(year, tiles[0])])}')
                print(f'Dates in {tiles[0]} not in {tile}: {sorted(dates[(year, tiles[0])] - dates[(year, tile)])}')
                print('\n')


def list_all_files_and_save():
    """
    List all files in the input directory and save them to a csv file.
    """
    io_config = IOConfig()
    io = IO(io_config)
    df = pd.DataFrame()
    for tile in io_config.available_tiles:
        for year in io_config.available_years:
            for product in io_config.available_products:
                print(f'\nListing files for {year}, {tile}, {product}')
                df = pd.concat([df, io.list_sentinel_files(TileRef(year, tile, product))[1]])
    df.to_csv('sentinel_files.csv', index=False)
    io.close_connection()


def test_file_download():
    """
    Test the download of a single file.
    """
    io_config = IOConfig()
    io = IO(io_config)
    tile_ref = TileRef(2021, '33TUN', 'NDVI_raw')
    refs, df = io.list_sentinel_files(tile_ref)
    image = refs[0]
    io.download_file(image)
    filepath = f'{io.config.base_local_dir}/{image.rel_filepath()}'
    try:
        io.check_existence_on_local(filepath)
        io.delete_local_file(image)
    except FileNotFoundError:
        pytest.fail(f'File could not be found on local machine')

    io.close_connection()


def test_extract_date():
    """
    Test the extraction of the date from the filename.
    """
    io_config = IOConfig()
    io = IO(io_config)
    tile_ref = TileRef(2021, '33TUN', 'NDVI_raw')
    refs, df = io.list_sentinel_files(tile_ref)
    image = refs[0]
    print(image.filename)
    date = image.extract_date()
    print(image.extract_date())
    assert len(date) == 8 and date.isdigit() and date == '20210103'
    io.close_connection()


def test_file_upload():
    """
    Test the upload of a single file.
    """
    io_config = IOConfig()
    io = IO(io_config)
    image = ImageRef('test_image.tif', year=2018, tile='33TUM', product='NDVI_raw', type='testing')
    io.upload_file(image)

    # Check if the file is there
    filepath = f'{io.config.base_server_dir}/wp4/{image.rel_filepath()}'
    try:
        io.check_existence_on_server(filepath)
        io.delete_remote_file(image)
    except FileNotFoundError:
        pytest.fail(f'File could not be found on server')

    io.close_connection()


def test_file_removal_on_server():
    """
    Test the removal of a single file on the server
    """
    io_config = IOConfig()
    io = IO(io_config)

    image = ImageRef('test_image.tif', year=2018, tile='33TUM', product='NDVI_raw', type='testing')
    io.upload_file(image)
    filepath = f'{io.config.base_server_dir}/wp4/{image.rel_filepath()}'
    io.check_existence_on_server(filepath)

    io.delete_remote_file(image)

    # Check if the file is there
    with pytest.raises(FileNotFoundError):
        io.check_existence_on_server(filepath)

    io.close_connection()


def test_file_removal_on_local():
    """
    Test the removal of a single file on the local machine
    """
    io_config = IOConfig()
    io = IO(io_config)

    image = ImageRef('test_image.tif', year=2018, tile='33TUM', product='NDVI_raw', type='testing')
    filepath = f'{io.config.base_local_dir}/{image.rel_filepath()}'
    image_2 = ImageRef('test_image_2.tif', year=2018, tile='33TUM', product='NDVI_raw', type='testing')
    filepath_2 = f'{io.config.base_local_dir}/{image_2.rel_filepath()}'

    # Copy the file into the same directory with test_image_to_delete.tif
    shutil.copyfile(filepath, filepath_2)

    io.delete_local_file(image_2)
    with pytest.raises(FileNotFoundError):
        io.check_existence_on_local(filepath_2)

    io.close_connection()


if __name__ == '__main__':
    list_all_files_and_save()
    check_dates()

    # test_file_download()
    # test_file_upload()
    # test_file_removal_on_server()
    # test_file_removal_on_local()
    # test_extract_date()
