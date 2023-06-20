import paramiko
import pandas as pd
import os

from src.config.io_config import IOConfig


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

    def list_sentinel_files(self, year: int, tile: str, product: str) -> pd.DataFrame:
        """
        List all files in a given directory on the remote server.

        Args:
            year: year of the data
            tile: Sentinel tile
            product: product type. Can be ['NDVI_raw', 'B02', 'B03', 'B04', 'B08', 'B11', 'SCL']

        Returns:
            df (pd.DataFrame): dataframe with the following columns
        """
        self.check_inputs_with_metadata(year, tile, product)

        if product == 'NDVI_raw':
            dir = f'{self._config.base_server_dir}/{year}/{tile}/{product}'
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
        df['year'] = year
        df['tile'] = tile
        df['product'] = product

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
        df = df[df['product_f'] == product]

        # Print summary
        print(f'Found {len(df)} files for: \n - year = {year}\n - tile = {tile}\n - product = {product}.')
        print(f'Total size: {df["size"].sum()} Mb.')
        print(f'Date range: {df["date_f"].min()} - {df["date_f"].max()}')

        return df

    def download_file(self, year: int, tile: str, product: str, filename: str):
        """
        Download a file from the remote server.

        Args:
            year: year of the data
            tile: Sentinel tile
            product: product type
            filename: name of the file to download
        """
        self.check_inputs_with_metadata(year, tile, product)

        if product == 'NDVI_raw':
            dir = f'{self._config.base_server_dir}/{year}/{tile}/{product}'
        else:
            dir = f'{self._config.base_server_dir}/{year}/{tile}/tmp'
        self.check_existence_on_server(dir, filename)
        local_dir = f'{self._config.base_local_dir}/{year}/{tile}/{product}'
        self.check_existence_on_local(local_dir)

        sftp = self._ssh_client.open_sftp()
        sftp.get(f'{dir}/{filename}', f'{local_dir}/{filename}')

    def check_inputs_with_metadata(self, year: int, tile: str, product: str) -> None:
        """
        Check if the tile, year and product are available according to metadata.

        Args:
            product: product type
            tile: Sentinel tile
            year: year of the data
        """
        assert year in self._config.available_years, f'Year {year} not available.'
        assert tile in self._config.available_tiles, f'Tile {tile} not available.'
        assert product in self._config.available_products, f'Product {product} not available.'

    def check_existence_on_local(self, local_dir):
        """
        Check if a directory exists on the local machine. If not, create it.

        Args:
            local_dir: string with the directory to check
        """
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)


if __name__ == '__main__':
    """
    List all files in the input directory and save them to a csv file.
    This code is just for testing purposes, will be removed later.
    """
    io_config = IOConfig()
    io = IO(io_config)
    io.open_connection()
    df = pd.DataFrame()
    for tile in io_config.available_tiles:
        for year in io_config.available_years:
            for product in io_config.available_products:
                print(f'\nListing files for {year}, {tile}, {product}')
                df = pd.concat([df, io.list_sentinel_files(year, tile, product)])
    df.to_csv('sentinel_files.csv', index=False)

    year, tile, prod = 2021, '33TUM', 'NDVI_raw'
    df = io.list_sentinel_files(year, tile, prod)
    filename = df.filename.iloc[0]
    #
    # io.download_file(year, tile, prod, filename)

    io.close_connection()

