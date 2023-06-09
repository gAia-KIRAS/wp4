import paramiko
import pandas as pd

from src.config.io_config import IOConfig


class IO:
    # Responsible for handling input and output
    def __init__(self, io_config: IOConfig):
        self._config = io_config
        self._ssh_client = None

    def open_connection(self):
        # Open connection to the Kronos server through SSH
        # Create an SSH client
        self._ssh_client = paramiko.SSHClient()

        # Automatically add the remote server's host key
        self._ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect to the remote server
        self._ssh_client.connect(self._config.server_name,
                           username=self._config.username,
                           password=self._config.password)

    def close_connection(self):
        # Close the SSH connection
        self._ssh_client.close()

    def check_existance_dir(self, directory):
        # Check if directory exists. If not, raise an exception
        command = f'test -d {directory}/ && echo "True" || echo "False"'
        stdin, stdout, stderr = self._ssh_client.exec_command(command)
        if stdout.readlines()[0].strip() == 'False':
            raise Exception(f'Directory {directory} does not exist.')

    def run_command(self, command):
        # Run a command on the remote server
        stdin, stdout, stderr = self._ssh_client.exec_command(command)

        # Check for errors
        err = stderr.readlines()
        if err:
            raise Exception(f'Error: {err[0]}')

        return stdout.readlines()

    def list_sentinel_files(self, year, tile, product):
        # TODO: checks on the year, title, and product

        dir = f'{self._config.base_input_dir}/{year}/{tile}/{product}'
        self.check_existance_dir(dir)

        res = self.run_command(f'cd {dir}; ls -sh')

        df = pd.DataFrame(res, columns=['filename'])
        df[['size', 'filename']] = df['filename'].str.split(' ', expand=True)
        # Filter out column with the total
        df = df[df['size'] != 'total']
        df['size'] = df['size'].str[:-1].astype('int64')
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
        df['product_f'] = df[10].str.split('.', expand=True)[0]
        df['extension_f'] = df[10].str.split('.', expand=True)[1].str[:-1]

        df.drop(columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'filename'], inplace=True)
        print(df
              )


if __name__ == '__main__':
    io_config = IOConfig()
    io = IO(io_config)
    io.open_connection()
    io.list_sentinel_files('2018', '33TUM', 'NDVI_raw')
    io.close_connection()
