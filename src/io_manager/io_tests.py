import shutil

import pandas as pd
import pytest

from config.io_config import IOConfig
from io_manager.io_manager import IO
from utils import TileRef, ImageRef, RECORDS_FILE_COLUMNS


def test_file_download():
    """
    Test the download of a single file.
    """
    io_config = IOConfig()
    io = IO(io_config)
    tile_ref = TileRef(2021, '33TUN', 'NDVI_raw')
    refs, df = io.list_files_on_server(tile_ref)
    image = refs[0]
    io.download_file(image)
    filepath = f'{io.config.base_local_dir}/{image.rel_filepath()}'
    try:
        io.check_existence_on_local(filepath)
        io.delete_local_file(image)
    except FileNotFoundError:
        pytest.fail(f'File could not be found on local machine')

    io.close_connection()


def test_extract_date_valid():
    """
    Test the extraction of the date from the filename.
    """
    io_config = IOConfig()
    io = IO(io_config)
    tile_ref = TileRef(2021, '33TUN', 'NDVI_raw')
    refs, df = io.list_files_on_server(tile_ref)
    image = refs[0]
    print(image.filename)
    date = image.extract_date()
    print(image.extract_date())
    assert len(date) == 8 and date.isdigit() and date == '20210103'
    io.close_connection()


def test_extract_date_invalid():
    """
    Test the extraction of the date from the filename.
    Date is invalid so it should raise an exception.
    """
    io_config = IOConfig()
    io = IO(io_config)
    tile_ref = TileRef(2021, '33TUN', 'NDVI_raw')
    refs, df = io.list_files_on_server(tile_ref)
    image = refs[0]
    image.filename = '_'.join(image.filename)
    with pytest.raises(Exception):
        _ = image.extract_date()
    io.close_connection()


def test_file_upload():
    """
    Test the upload of a single file.
    """
    io_config = IOConfig()
    io = IO(io_config)

    image = ImageRef('test_image.tif', year=2018, tile='33TUM', product='NDVI_raw', type='testing')
    filepath = f'{io.config.base_local_dir}/{image.rel_filepath()}'
    image_2 = ImageRef('test_image_2.tif', year=2018, tile='33TUM', product='NDVI_raw', type='testing')
    filepath_2 = f'{io.config.base_local_dir}/{image_2.rel_filepath()}'

    shutil.copyfile(filepath, filepath_2)
    io.upload_file(image_2)

    # Check if the file is there
    filepath = f'{io.config.base_server_dir}/wp4/{image_2.rel_filepath()}'
    try:
        io.check_existence_on_server(filepath)
        io.delete_remote_file(image_2)
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
    image_2 = ImageRef('test_image_2.tif', year=2018, tile='33TUM', product='NDVI_raw', type='testing')
    io.upload_file(image)
    filepath = f'{io.config.base_server_dir}/wp4/{image.rel_filepath()}'
    filepath_2 = f'{io.config.base_server_dir}/wp4/{image.rel_filepath()}'

    # Copy image
    filepath_2 = filepath_2.replace('test_image.tif', 'test_image_2.tif')
    io.run_command(f"cp {filepath} {filepath_2}")

    io.check_existence_on_server(filepath_2)
    io.delete_remote_file(image_2)

    # Check if the file is there
    with pytest.raises(FileNotFoundError):
        io.check_existence_on_server(filepath_2)

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


def test_load_of_records():
    """
    Test the loading of the log records file.
    """
    io_config = IOConfig()
    io = IO(io_config)
    records = io.get_records()
    assert len(records) > 0 and isinstance(records, pd.DataFrame) and set(records.columns) == set(RECORDS_FILE_COLUMNS)
    io.close_connection()


def test_existence_on_local_does_not_exist():
    """
    Test the check for existence of a file on the local machine.
    The file does not exist so FileNotFoundError is expected.
    """
    io_config = IOConfig()
    io = IO(io_config)
    image = ImageRef('test_image_inexistend.tif', year=2018, tile='33TUM', product='NDVI_raw', type='testing')
    filepath = f'{io.config.base_local_dir}/{image.rel_filepath()}'
    with pytest.raises(FileNotFoundError):
        io.check_existence_on_local(filepath)
    io.close_connection()


def test_existence_on_local_does_exist():
    """
    Test the check for existence of a file on the local machine.
    The file does exist so no error is expected.
    """
    io_config = IOConfig()
    io = IO(io_config)
    image = ImageRef('test_image.tif', year=2018, tile='33TUM', product='NDVI_raw', type='testing')
    filepath = f'{io.config.base_local_dir}/{image.rel_filepath()}'
    try:
        io.check_existence_on_local(filepath)
    except FileNotFoundError:
        pytest.fail(f'File could not be found on local machine')
    io.close_connection()


def test_existence_on_server_does_exist():
    """
    Test the check for existence of a file on the server.
    The file does exist so no error is expected.
    """
    io_config = IOConfig()
    io = IO(io_config)
    image = ImageRef('test_image.tif', year=2018, tile='33TUM', product='NDVI_raw', type='testing')
    filepath = f'{io.config.base_server_dir}/wp4/{image.rel_filepath()}'
    try:
        io.check_existence_on_server(filepath)
    except FileNotFoundError:
        pytest.fail(f'File could not be found on server')
    io.close_connection()


def test_existence_on_server_does_not_exist():
    """
    Test the check for existence of a file on the server.
    The file does not exist so FileNotFoundError is expected.
    """
    io_config = IOConfig()
    io = IO(io_config)
    image = ImageRef('test_image_does_not_exist.tif', year=2018, tile='33TUM', product='NDVI_raw', type='testing')
    filepath = f'{io.config.base_server_dir}/wp4/{image.rel_filepath()}'
    with pytest.raises(FileNotFoundError):
        io.check_existence_on_server(filepath)
    io.close_connection()


def test_list_files_raw():
    """
    Check listing of files of tile_ref in the server
    """
    io_config = IOConfig()
    io = IO(io_config)
    tile_ref = TileRef(2020, '33TUM', 'NDVI_raw')
    image_refs, df = io.list_files_on_server(tile_ref, 'raw')
    assert type(image_refs) == list and len(image_refs) > 0
    assert type(df) == pd.DataFrame and len(df) > 0
    assert set(df.year_f.unique()) == {'2020'}
    assert set(df.tile_f.unique()) == {'33TUM'}
    assert set(df.product_f.unique()) == {'NDVI_raw'}
    io.close_connection()


def test_list_files_crop():
    """
    Check listing of files of tile_ref in the server
    """
    io_config = IOConfig()
    io = IO(io_config)
    tile_ref = TileRef(2020, '33TUM', 'NDVI_raw')
    image_refs, df = io.list_files_on_server(tile_ref, 'crop')
    assert set(df.year_f.unique()) == {'2020'}
    assert set(df.tile_f.unique()) == {'33TUM'}
    assert set(df.product_f.unique()) == {'NDVI_raw'}
    io.close_connection()


def test_list_all_files_raw():
    """
    Check listing of all files of type 'raw' in the server
    """
    io_config = IOConfig()
    io = IO(io_config)
    df = io.list_all_files_of_type('raw')
    assert type(df) == pd.DataFrame and len(df) > 0
    assert set(df.year_f.unique()) == set(io.config.available_years)
    assert set(df.tile_f.unique()) == set(io.config.available_tiles)
    assert set(df.product_f.unique()) == set(io.config.available_products)
    io.close_connection()


def test_list_all_files_crop():
    """
    Check listing of all files of type 'crop' in the server
    """
    io_config = IOConfig()
    io = IO(io_config)
    df = io.list_all_files_of_type('crop')
    assert type(df) == pd.DataFrame and len(df) > 0
    io.close_connection()


if __name__ == '__main__':
    pytest.main([''])
