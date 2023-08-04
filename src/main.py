from pyinstrument import Profiler
from datetime import datetime, timedelta
import argparse

from config.config import Config
from modules.nci import NCI
from config.io_config import IOConfig
from io_manager.io_manager import IO
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

    args = parse_arguments()

    config = Config()
    io_config = IOConfig()
    io_manager = IO(io_config)

    if args.server_execution:
        # Modify paths in io_config to match the server paths. Now local paths are server paths
        io_config.modify_paths_for_server()

    profiler = Profiler()
    if config.profiling_active:
        profiler.start()

    module = {
        'crop': IntersectAOI,
        'nci': NCI
    }.get(config.execute, KeyError(f'{config.execute} is not a valid module.'))(config=config, io=io_manager)

    if args.server_execution:
        # This means that we are on the server, and we want to execute on the server
        print(' -- Server execution -- ')

        # Execute module
        module.run(on_the_server=True)

    else:

        if config.execution_where == 'local':
            # This means that we are on the local machine, and we want to execute locally
            print(' -- Local execution -- ')

            module.run()
            io_manager.close_connection()

        elif config.execution_where == 'server':
            # This means that we are on the local machine, and we want to execute on the server
            print(' -- Preparing the server execution -- ')

            # 1. Upload the config.yaml file to the server
            io_manager.upload_config()
            # 2. Upload the inventory data to the server, in case it is not there
            io_manager.upload_inventory()
            # 3. Upload the operations data to the server
            io_manager.upload_operations()

            print(' -- Server execution information -- ')
            print(f'Executing module: {config.execute}')
            print(f'Filters: {config.filters}')
            print(f'Runtime limit: {config.time_limit} minutes')
            print(f'Execution started at: {datetime.now()}')
            print(f'Expected to finish before: {datetime.now() + timedelta(minutes=config.time_limit)}')

            # 4. Execute the module
            command = f"""
            cd {io_manager.config.server_repo_root};
            nice -n 10 {io_manager.config.server_python_executable} {io_manager.config.server_repo_root}/src/main.py --server_execution;
            """
            print(' -- Server execution started -- ')
            out = io_manager.run_command(command, raise_exception=False)
            print(*out, sep='\n')
            print(' -- Server execution finished -- ')

            # Copy the records from server to local
            io_manager.update_records_on_local()

    if config.profiling_active:
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
        if config.profiling_browser:
            profiler.open_in_browser()
