from config.io_config import IOConfig
from io_manager import IO
from utils import TileRef

import pandas as pd


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
            refs, df = io.list_files_on_server(tile_ref)
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
                df = pd.concat([df, io.list_files_on_server(TileRef(year, tile, product))[1]])
    df.to_csv('sentinel_files.csv', index=False)
    io.close_connection()


def update_broken_records_for_nci(df):
    """
    Update the records with the NCI computations that have already been done.
    """
    records = pd.read_csv(io_config.records_path)
    records = records[(records['from'] != 'raw') | (records['to'] != 'nci')]

    for index, row in df.iterrows():
        # Change format of string of the date. First convert from string to datetime
        date = pd.to_datetime(row['date_f'], format='%Y-%m-%d').strftime('%Y%m%d')
        new_record = {
            'from': ['raw'],
            'to': ['nci'],
            'tile': [row['tile']],
            'year': [row['year']],
            'product': [row['product']],
            'filename_to': [row['filename']],
            'filename_from': [f'{row["tile"]}_{row["year"]}_{date}_{row["product"]}.tif'],
            'success': [1]
        }
        records = pd.concat([records, pd.DataFrame(new_record)])
    records.to_csv(io_config.records_path, index=False)


if __name__ == '__main__':
    io_config = IOConfig()
    io = IO(io_config)

    df = io.list_all_files_of_type('nci')
    update_broken_records_for_nci(df)

    # check_dates()
