from typing import List, Tuple, Any

import numpy as np
import paramiko
import pandas as pd
import os
import warnings
import pickle

from config.io_config import IOConfig
from utils import ImageRef, TileRef, RECORDS_FILE_COLUMNS


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
        self._open_connection()

    def _open_connection(self) -> None:
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

    def check_existence_on_server(self, remote_path: str, dir: bool = False) -> None:
        """
        Check if a directory exists on the remote server. If it is a directory and does not exist, create it.
        If it is a file and does not exist, raise an error.

        Args:
            remote_path: string with the path to check
            dir: boolean indicating whether the path is a directory. Otherwise, it is a file.
        """

        if dir:
            # Check if directory exists. If not, create it
            command = f'test -d {remote_path}/ && echo "True" || echo "False"'
            stdin, stdout, stderr = self._ssh_client.exec_command(command)
            if stdout.readlines()[0].strip() == 'False':
                # Create the directory
                command = f'mkdir -p {remote_path}'
                stdin, stdout, stderr = self._ssh_client.exec_command(command)

        else:
            # Check if file exists. If not, raise an exception
            command = f'test -f {remote_path} && echo "True" || echo "False"'
            stdin, stdout, stderr = self._ssh_client.exec_command(command)
            if stdout.readlines()[0].strip() == 'False':
                raise FileNotFoundError(f'File {remote_path} does not exist.')

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
            err_message = "\n----- ERROR MESSAGE FROM SERVER CONSOLE -----\n" + \
                          '\n'.join(err) + "\n----- END OF ERROR MESSAGE -----\n\n"
            raise Exception(f'Error: {err_message}')

        return stdout.readlines()

    def list_files_on_server(self, tile_ref: TileRef, image_type='raw') -> Tuple[List[ImageRef], pd.DataFrame]:
        """
        List all Sentinel files (type='raw') for a particular tile reference (tile + product + year).

        Args:
            tile_ref (TileRef): TileRef object with the tile to list files from
            image_type (str): type of the files. Must be in IMAGE_TYPES (src.utils.py)

        Returns:
            i_refs (List[ImageRef]): list with ImageRef objects on the given directory
            df (pd.DataFrame): dataframe with the whole information of the files in the directory
        """
        self.check_inputs_with_metadata(tile_ref)

        dir = self.build_remote_dir_for_tile(tile_ref, image_type=image_type)

        # Check if directory exists. Will create it if it does not exist
        self.check_existence_on_server(dir, dir=True)

        # Run command to list all filenames and sizes
        res = self.run_command(f'cd {dir}; ls -sh')
        df = pd.DataFrame(res, columns=['filename'])

        df[['size', 'filename']] = df['filename'].str.split(expand=True)
        # Filter out files that are not .tif
        df = df[df['filename'].str.endswith('.tif')]

        if df.empty:
            return [], df

        # Convert size to Mb, store year, tile and product
        df['size'] = df['size'].str[:-1].astype('float')
        df['year'] = str(tile_ref.year)
        df['tile'] = tile_ref.tile
        df['product'] = tile_ref.product

        if image_type == 'raw' and tile_ref.product == 'NDVI_reconstructed':
            df = df.join(df['filename'].str.split('_', expand=True))
            df['built_tile_f'] = df[0]
            df['year_f'] = df[1]
            df['date_f'] = pd.to_datetime(df[2], format='%Y%m%d')
            df['product_f'] = df[3] + '_' + df[4].str.split('.', expand=True)[0]
            df['extension_f'] = df[4].str.split('.', expand=True)[1]
            df['month_f'] = df['date_f'].dt.month
            df['x_f'] = np.NAN
            df['y_f'] = np.NAN
            df['base_product_f'] = np.NAN

            df.drop(columns=list(range(5)), inplace=True)

        elif image_type == 'raw':
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
        else:
            df = df.join(df['filename'].str.split('_', expand=True))
            # We denote colname_f if the columns are reconstructed from the filename
            df['built_tile_f'] = df[3]
            df['year_f'] = df[2]
            df['tile_f'] = df[3]
            df['date_f'] = df[4].str.split('.', expand=True)[0]
            df['date_f'] = pd.to_datetime(df['date_f'], format='%Y%m%d')
            df['extension_f'] = df[4].str.split('.', expand=True)[1]
            df['month_f'] = df['date_f'].dt.month
            df['x_f'] = np.NAN
            df['y_f'] = np.NAN
            df['base_product_f'] = np.NAN

            product_map = {'NDVIraw': 'NDVI_raw'}
            df['product_f'] = df[1].map(lambda x: product_map.get(x, x))

            df.drop(columns=list(range(5)), inplace=True)

        # Filter by product
        df = df[df['product_f'] == tile_ref.product]

        # Print summary
        print(f'Found {len(df)} files for: \n - year = {tile_ref.year}\n - tile = {tile_ref.tile}\n '
              f'- product = {tile_ref.product}\n - type = {image_type}')
        print(f'Total size: {df["size"].sum()} Mb.')
        print(f'Date range: {df["date_f"].min()} - {df["date_f"].max()}')

        # Sort by date
        df.sort_values(by='date_f', inplace=True)

        # Create list of ImageRef objects
        i_refs = [ImageRef(f, tile_ref=tile_ref, type=image_type) for f in df.filename.values.tolist()]

        return i_refs, df

    def list_all_files_of_type(self, image_type) -> pd.DataFrame:
        """
        List all raw files on the server.

        Returns:
            df (pd.DataFrame): dataframe with the whole information of the files in the directory
        """
        try:
            df = pd.read_csv(self._config.all_images_path[image_type])
            return df
        except FileNotFoundError:
            df = pd.DataFrame()
            for tile in self._config.available_tiles:
                for year in self._config.available_years:
                    for product in self._config.available_products:
                        print(f'\nListing files for {year}, {tile}, {product}')
                        df = pd.concat([df, self.list_files_on_server(TileRef(year, tile, product), image_type)[1]])
            df.to_csv(self._config.all_images_path[image_type], index=False)
            return df

    def build_remote_dir_for_image(self, image: ImageRef) -> str:
        """
        Build the remote directory where the image is located. For type = "raw", the remote base directory is
        wp3/sentinel2_L2A. For any other type, the remote base directory is wp4/{type}.

        Args:
            image: reference to the image to download. Must have type attribute.
        """
        if image.type is None:
            raise Exception('Cannot build remote directory. Image type must be specified.')

        return self.build_remote_dir_for_tile(image.tile_ref, image.type)

    def build_remote_dir_for_tile(self, tile_ref: TileRef, image_type: str):
        """
        Build the remote directory where the correspondant images are located.
        For type = "raw", the remote base directory is wp3/sentinel2_L2A.
        For any other type, the remote base directory is wp4/{type}.

        Args:
            tile_ref: reference to the image to download
            image_type: string in IMAGE_TYPES
        """
        rel_dir = ("wp3/sentinel2_L2A" if image_type == 'raw' else f"wp4/{image_type}") + \
                  f"/{tile_ref.year}/{tile_ref.tile}"
        if tile_ref.product in ['NDVI_raw', 'NDVI_reconstructed'] or image_type != 'raw':
            rel_dir += f'/{tile_ref.product}'
        else:
            # Only raw non-NDVI files are stored in a tmp folder
            rel_dir += f"/tmp"
        return f"{self._config.base_server_dir}/{rel_dir}"

    def download_file(self, image: ImageRef):
        """
        Download a file from the remote server. If the file already exists locally, it will not be downloaded.

        Args:
            image: reference to the image to download
        """
        print(f'Downloading image {image} from server')
        self.check_inputs_with_metadata(image.tile_ref)

        dir = self.build_remote_dir_for_image(image)
        filepath = f'{dir}/{image.filename}'
        self.check_existence_on_server(dir, dir=True)
        self.check_existence_on_server(filepath, dir=False)

        local_dir = f'{self._config.base_local_dir}/{image.rel_dir()}'
        local_filepath = f'{self._config.base_local_dir}/{image.rel_filepath()}'
        self.check_existence_on_local(local_dir, dir=True)

        try:
            self.check_existence_on_local(local_filepath)
            warnings.warn(f'\nFile {image.filename} already exists on local machine. Will not be downloaded')
        except FileNotFoundError as e:
            sftp = self._ssh_client.open_sftp()
            sftp.get(f'{dir}/{image.filename}', local_filepath)
            sftp.close()

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
        print(f'Uploading image {image} to server.')
        self.check_inputs_with_metadata(image.tile_ref)

        dir = self.build_remote_dir_for_image(image)
        filepath = f'{dir}/{image.filename}'
        self.check_existence_on_server(dir, dir=True)

        try:
            self.check_existence_on_server(filepath, dir=False)
            warnings.warn(f'\nFile {image.filename} already exists on server. Overwritting it.')
        except FileNotFoundError:
            pass

        local_dir = f'{self._config.base_local_dir}/{image.rel_dir()}'
        self.check_existence_on_local(local_dir, dir=True)
        local_filepath = f'{self._config.base_local_dir}/{image.rel_filepath()}'
        self.check_existence_on_local(local_filepath)

        sftp = self._ssh_client.open_sftp()
        sftp.put(local_filepath, filepath)
        sftp.close()

    def delete_local_file(self, image: ImageRef):
        """
        Delete a file from the local machine.

        Args:
            image: ImageRef object with the image to delete
        """
        print(f'Deleting image {image} from local machine.')

        self.check_inputs_with_metadata(image.tile_ref)

        local_dir = f'{self._config.base_local_dir}/{image.rel_dir()}'
        self.check_existence_on_local(local_dir, dir=True)
        try:
            self.check_existence_on_local(f'{local_dir}/{image.filename}', dir=False)
        except FileNotFoundError as e:
            warnings.warn(f'\nFile {image.filename} does not exist on local machine. Will not be deleted')
            return

        os.remove(f'{local_dir}/{image.filename}')

    def delete_remote_file(self, image: ImageRef):
        print(f'Deleting image {image} from server.')
        self.check_inputs_with_metadata(image.tile_ref)

        dir = self.build_remote_dir_for_image(image)
        filepath = f'{dir}/{image.filename}'
        self.check_existence_on_server(dir, dir=True)
        self.check_existence_on_server(filepath, dir=False)

        # Make it impossible to delete files from the raw folder
        if image.type == 'raw':
            raise Exception('Cannot delete images from the raw folder in the server!')

        sftp = self._ssh_client.open_sftp()
        sftp.remove(filepath)
        sftp.close()

    def get_records(self) -> pd.DataFrame:
        """
        Get the records .csv file from the local machine. It is a table with the following columns:
        - 'from': str with the type of the image before the operation
        - 'to': str with the type of the image after the operation
        - 'year': int with the year of the image
        - 'tile': str with the tile of the image
        - 'product': str with the product of the image
        - 'timestamp': datetime with the timestamp of the operation
        - 'filename_from': str with the filename of the image before the operation
        - 'filename_to': str with the filename of the image after the operation
        - 'success': str with the status of the operation. Can be 1 (success) or 0 (failure)
        If it is empty, return an empty DataFrame with the correspondent columns.

        Returns:
            records: pandas DataFrame with the records
        """
        try:
            records = pd.read_csv(self._config.records_path)
            return records
        except FileNotFoundError:
            return pd.DataFrame(columns=RECORDS_FILE_COLUMNS)

    def save_records(self, records: pd.DataFrame):
        """
        Save the records .csv file to the local machine.

        Args:
            records: pandas DataFrame with the records
        """
        assert set(records.columns) == set(RECORDS_FILE_COLUMNS), \
            f'Columns of records file must be {RECORDS_FILE_COLUMNS}'
        records.to_csv(self._config.records_path, index=False)

    @staticmethod
    def save_pickle(object: Any, filepath: str):
        """
        Save an object to a pickle file.

        Args:
            object: object to save
            filepath: path to the pickle file
        """
        with open(filepath, 'wb') as f:
            pickle.dump(object, f)

    @staticmethod
    def load_pickle(filepath: str) -> Any:
        """
        Load an object from a pickle file.

        Args:
            filepath: path to the pickle file

        Returns:
            object: object loaded from the pickle file
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def filter_all_images(self, image_type: str, filters: dict) -> pd.DataFrame:
        """
        Filter all raw
        """
        images_df = self.list_all_files_of_type(image_type)[['year', 'tile', 'product', 'filename']]
        if filters['product']:
            images_df = images_df.loc[images_df['product'].isin(filters['product'])]
        if filters['year']:
            images_df = images_df.loc[images_df['year'].isin(filters['year'])]
        if filters['tile']:
            images_df = images_df.loc[images_df['tile'].isin(filters['tile'])]
        return images_df

    def upload_config(self) -> None:
        """
        Upload the config.yaml and io_config.yaml files to the server. Always replace the existing ones.
        """
        print('Uploading config.yaml and io_config.yaml to server.')

        sftp = self._ssh_client.open_sftp()

        # 1. copy io_config.yaml to server
        local_io_config_path = f'{self._config.config_path}'
        server_io_config_path = f'{self._config.server_repo_root}/io_config.yaml'
        self.check_existence_on_local(local_io_config_path)
        sftp.put(local_io_config_path, server_io_config_path)

        # 2. copy config.yaml to server
        local_config_path = local_io_config_path.replace('io_config.yaml', 'config.yaml')
        server_io_config_path = f'{self._config.server_repo_root}/config.yaml'
        self.check_existence_on_local(local_config_path)
        sftp.put(local_config_path, server_io_config_path)

        sftp.close()

    def upload_inventory(self):
        """
        Upload the inventory and aoi files to the server if they are not there. On the server, they are located in
        newstorage2/wp4/inventory. We do not copy the whole inventory folder, only the three needed files.
        """
        print('Uploading inventory and aoi files to server.')

        sftp = self._ssh_client.open_sftp()

        self.check_existence_on_server(f'{self._config.base_server_dir}/wp4/inventory', dir=True)
        server_path_inventory = f'{self._config.base_server_dir}/wp4/{self._config.inventory_rel_path}'
        server_path_aoi_shp = f'{self._config.base_server_dir}/wp4/{self._config.aoi_rel_path["shp"]}'
        server_path_aoi_gpkg = f'{self._config.base_server_dir}/wp4/{self._config.aoi_rel_path["gpkg"]}'

        try:
            self.check_existence_on_server(server_path_inventory, dir=False)
        except FileNotFoundError:
            sftp.put(self._config.inventory_path, server_path_inventory)

        try:
            self.check_existence_on_server(server_path_aoi_shp, dir=False)
        except FileNotFoundError:
            sftp.put(self._config.aoi_path['shp'], server_path_aoi_shp)

        try:
            self.check_existence_on_server(server_path_aoi_gpkg, dir=False)
        except FileNotFoundError:
            sftp.put(self._config.aoi_path['gpkg'], server_path_aoi_gpkg)

        sftp.close()

    def upload_operations(self):
        """
        Overwrite the operations folder in the server with the local one. The operations folder is located in
        newstorage2/wp4/operation_records. We copy the whole folder.
        """
        print('Uploading operations folder to server.')

        sftp = self._ssh_client.open_sftp()

        local_operations_path = f'{self._config.base_local_dir}/operation_records'
        server_operations_path = f'{self._config.base_server_dir}/wp4/operation_records'
        self.check_existence_on_server(server_operations_path, dir=True)

        # List all files in the local operations folder using os.listdir
        local_operations_files = os.listdir(local_operations_path)

        # For each of the files, do sftp.put to the server with the same file name
        for file in local_operations_files:
            sftp.put(f'{local_operations_path}/{file}', f'{server_operations_path}/{file}')

        sftp.close()

    def update_records_on_local(self):
        """
        Update the records file on the local machine with the one on the server.
        """
        print('Updating records file on local machine.')

        sftp = self._ssh_client.open_sftp()

        local_records_path = f'{self._config.records_path}'
        server_records_path = f'{self._config.base_server_dir}/wp4/operation_records/records.csv'
        self.check_existence_on_server(server_records_path, dir=False)
        sftp.get(server_records_path, local_records_path)

        sftp.close()

    @property
    def config(self):
        return self._config

