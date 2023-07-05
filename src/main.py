from pyinstrument import Profiler
from src.config.config import Config

from src.modules.nci import NCI
from src.config.io_config import IOConfig
from src.io.io_manager import IO
from src.modules.intersect_with_aoi import IntersectAOI


if __name__ == "__main__":
    config = Config()
    io_config = IOConfig()
    io_manager = IO(io_config)

    profiler = Profiler()
    if config.profiling_active:
        profiler.start()

    module = {
        'crop': IntersectAOI(config=config, io=io_manager),
        'modules': NCI(config=config, io=io_manager)
    }.get(config.execute, KeyError(f'{config.execute} is not a valid module.'))

    module.run()
    io_manager.close_connection()

    if config.profiling_active:
        profiler.stop()
        print(profiler.output_text(unicode=True, color=True))
        if config.profiling_browser:
            profiler.open_in_browser()
