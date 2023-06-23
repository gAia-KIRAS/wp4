from typing import List, Tuple

import paramiko
import pandas as pd
import os

from src.config.io_config import IOConfig
from src.utils import ImageRef, TileRef


class IO:
    """
    Responsible for handling the input and output.

    Attributes:
        _config: IOConfig object with the configuration of the input and output
        _ssh_client: paramiko.SSHClient object with the SSH connection
    """
    def __init__(self, io_config: IOConfig):
        self._config = io_config
        self._ssh_client = None

    def open_connection(self) -> None:
        """
        Open a connection to the Kronos server through SSH.
        Raise an exception if the connection fails.
        """
        # Create an SSH client
        self._ssh_client = paramiko.SSHClient()

        # Automatically add the remote server's host key
        self._ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect to the remote server and raise exception if connection fails
        try:
            self._ssh_client.connect(self._config.server_name,
                                     username=self._config.username,
                                     password=self._config.password)
        except Exception as e:
            raise Exception(f'Error: Could not connect to remote server. '
                            f'Original error: {e}')

    def close_connection(self) -> None:
        """
        Close the SSH connection.
        """
        # Close the SSH connection
        self._ssh_client.close()

    def check_existence_on_server(self, directory: str, file: str = None) -> None:
        """
        Check if a directory exists on the remote server. Can also check if a file exists in the directory.
        If directory or file does not exist, raise an exception.

        Args:
            directory: string with the directory to check
            file: (optional) string with the file to check
        """

        # Check if directory exists. If not, raise an exception
        command = f'test -d {directory}/ && echo "True" || echo "False"'
        stdin, stdout, stderr = self._ssh_client.exec_command(command)
        if stdout.readlines()[0].strip() == 'False':
            raise Exception(f'Directory {directory} does not exist.')

        if file:
            # Check if file exists. If not, raise an exception
            command = f'test -f {directory}/{file} && echo "True" || echo "False"'
            stdin, stdout, stderr = self._ssh_client.exec_command(command)
            if stdout.readlines()[0].strip() == 'False':
                raise Exception(f'File {file} does not exist in {directory}.')

    def run_command(self, command: str):
        """
        Run a command on the remote server.

        Args:
            command: string with the command to run

        Returns:
            stdout.readlines() (list): list of strings with the output of the command
        """
        # Run the command
        stdin, stdout, stderr = self._ssh_client.exec_command(command)

        # Check for errors
        err = stderr.readlines()
        if err:
            raise Exception(f'Error: {err[0]}')

        return stdout.readlines()

    def list_sentinel_files(self, tile_ref: TileRef) -> Tuple[List[ImageRef], pd.DataFrame]:
        """
        List all files in a given directory on the remote server.

        Args:
            tile_ref (TileRef): TileRef object with the tile to list files from

        Returns:
            i_refs (List[ImageRef]): list with ImageRef objects on the given directory
            df (pd.DataFrame): dataframe with the whole information of the files in the directory
        """
        self.check_inputs_with_metadata(tile_ref)

        if tile_ref.product == 'NDVI_raw':
            dir = f'{self._config.base_server_dir}/{tile_ref.to_subpath()}'
        else:
            dir = f'{self._config.base_server_dir}/{year}/{tile}/tmp'

        # Check if directory exists
        self.check_existence_on_server(dir)

        # Run command to list all filenames and sizes
        res = self.run_command(f'cd {dir}; ls -sh')
        df = pd.DataFrame(res, columns=['filename'])

        df[['size', 'filename']] = df['filename'].str.split(expand=True)
        # Filter out files that are not .tif
        df = df[df['filename'].str.endswith('.tif')]

        # Convert size to Mb, store year, tile and product
        df['size'] = df['size'].str[:-1].astype('float')
        df['year'] = tile_ref.year
        df['tile'] = tile_ref.tile
        df['product'] = tile_ref.product

        df = df.join(df['filename'].str.split('_', expand=True))

        # We denote colname_f if the columns are reconstructed from the filename
        df['built_tile_f'] = df[0] + df[1] + df[2]
        df['year_f'] = df[3]
        df['month_f'] = df[4]
        df['x_f'] = df[5]
        df['tile_f'] = df[6]
        df['date_f'] = pd.to_datetime(df[7], format='%Y%m%d')
        df['y_f'] = df[8]
        df['base_product_f'] = df[9]

        product_map = {'NDVI': 'NDVI_raw'}
        df['product_f'] = df[10].str.split('.', expand=True)[0].map(lambda x: product_map.get(x, x))
        df['extension_f'] = df[10].str.split('.', expand=True)[1]

        df.drop(columns=list(range(11)), inplace=True)

        # Filter by product
        df = df[df['product_f'] == tile_ref.product]

        # Print summary
        print(f'Found {len(df)} files for: \n - year = {tile_ref.year}\n - tile = {tile_ref.tile}\n '
              f'- product = {tile_ref.product}.')
        print(f'Total size: {df["size"].sum()} Mb.')
        print(f'Date range: {df["date_f"].min()} - {df["date_f"].max()}')

        # Create list of ImageRef objects
        i_refs = [ImageRef(f, tile_ref=tile_ref) for f in df.filename.values.tolist()]

        return i_refs, df

    def download_file(self, image: ImageRef):
        """
        Download a file from the remote server.

        Args:
            image: reference to the image to download
        """
        self.check_inputs_with_metadata(image.tile_ref)

        if image.product == 'NDVI_raw':
            dir = f'{self._config.base_server_dir}/{image.tile_ref.to_subpath()}'
        else:
            dir = f'{self._config.base_server_dir}/{image.year}/{image.tile}/tmp'
        self.check_existence_on_server(dir, image.filename)
        local_dir = f'{self._config.base_local_dir}/{image.rel_dir()}'
        self.check_existence_on_local(local_dir, dir=True)

        try:
            self.check_existence_on_local(f'{local_dir}/{image.filename}')
            raise Warning(f'File {image.filename} already exists on local machine. Will not be downloaded')
        except FileNotFoundError as e:
            sftp = self._ssh_client.open_sftp()
            sftp.get(f'{dir}/{image.filename}', f'{self._config.base_local_dir}/{image.rel_filepath()}')

    def check_inputs_with_metadata(self, tile_ref: TileRef) -> None:
        """
        Check if the tile, year and product are available according to metadata.

        Args:
            tile_ref: reference to the tile to check
        """
        assert tile_ref.year in self._config.available_years, f'Year {tile_ref.year} not available.'
        assert tile_ref.tile in self._config.available_tiles, f'Tile {tile_ref.tile} not available.'
        assert tile_ref.product in self._config.available_products, f'Product {tile_ref.product} not available.'

    @staticmethod
    def check_existence_on_local(local_path: str, dir: bool = False):
        """
        Check if a directory or file exists on the local machine.
        If it is a directory and does not exist, create it.
        If it is a file and does not exist, raise an error.

        Args:
            local_path: string with the directory to check
            dir: boolean indicating if the path is a directory or not
        """
        if dir:
            if not os.path.isdir(local_path):
                os.makedirs(local_path)
        else:
            if not os.path.isfile(local_path):
                raise FileNotFoundError(f'File {local_path} does not exist.')

    def upload_file(self, image: ImageRef):
        """
        Upload a file to the remote server.

        Args:
            image: ImageRef object with the image to upload
        """
        self.check_inputs_with_metadata(image.tile_ref)

        if image.product == 'NDVI_raw':
            dir = f'{self._config.base_server_dir}/{image.year}/{image.tile}/{image.product}'
        else:
            dir = f'{self._config.base_server_dir}/{image.year}/{image.tile}/tmp'
        self.check_existence_on_server(dir, image.filename)

        local_dir = f'{self._config.base_local_dir}/raw/{image.year}/{image.tile}/{image.product}'
        self.check_existence_on_local(local_dir, dir=True)



        sftp = self._ssh_client.open_sftp()
        sftp.put(f'{local_dir}/{image.filename}', f'{dir}/{image.filename}')

    @property
    def config(self):
        return self._config


if __name__ == '__main__':
    """
    List all files in the input directory and save them to a csv file.
    This code is just for testing purposes, will be removed later.
    """
    io_config = IOConfig()
    io = IO(io_config)
    io.open_connection()
    # df = pd.DataFrame()
    # for tile in io_config.available_tiles:
    #     for year in io_config.available_years:
    #         for product in io_config.available_products:
    #             print(f'\nListing files for {year}, {tile}, {product}')
    #             df = pd.concat([df, io.list_sentinel_files(year, tile, product)[1]])
    # df.to_csv('sentinel_files.csv', index=False)

    tile_ref = TileRef(2021, '33TUN', 'NDVI_raw')
    refs, df = io.list_sentinel_files(tile_ref)
    # filename = df.filename.iloc[0]
    image = refs[0]
    image.type = 'raw'

    io.download_file(image)

    io.close_connection()

