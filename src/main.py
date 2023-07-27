from pyinstrument import Profiler
import argparse

from config.config import Config
from modules.nci import NCI
from config.io_config import IOConfig
from io.io_manager import IO
from modules.intersect_with_aoi import IntersectAOI


def parse_arguments():
    """
    Parses the arguments passed to the script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_execution', action='store_true',
                        help='If this flag is set to true, the script is being executed on the server')
    return parser.parse_args()


if __name__ == "__main__":
    # We might reach this point in three different ways:
    # 1. Locally, and we want to execute locally
    # 2. Locally, but we want to execute on the server
    # 3. On the server, and we want to execute on the server

    print('heyyyyyyy')

    args = parse_arguments()

    config = Config()
    io_config = IOConfig()
    io_manager = IO(io_config)

    profiler = Profiler()
    if config.profiling_active:
        profiler.start()

    module = {
        'crop': IntersectAOI(config=config, io=io_manager),
        'nci': NCI(config=config, io=io_manager)
    }.get(config.execute, KeyError(f'{config.execute} is not a valid module.'))

    if args.server_execution:
        # This means that we are on the server, and we want to execute on the server
        module.run_on_server()

        # Copy the records from server to local
        io_manager.update_records_on_local()

    else:

        if config.execution_where == 'local':
            # This means that we are on the local machine, and we want to execute locally

            module.run()
            io_manager.close_connection()

        elif config.execution_where == 'server':
            # This means that we are on the local machine, and we want to execute on the server

            # 1. Upload the config.yaml file to the server
            io_manager.upload_config()
            # 2. Upload the inventory data to the server, in case it is not there
            io_manager.upload_inventory()
            # 3. Upload the operations data to the server
            io_manager.upload_operations()

            # 4. Execute the module
            # TODO: use the 'nice' command to manage cpu usage and priority

            command = f'nice -n 10 python {io_manager.config.server_repo_root}/src/main.py --server_execution'
            io_manager.run_command(command)

    if config.profiling_active:
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
        if config.profiling_browser:
            profiler.open_in_browser()
