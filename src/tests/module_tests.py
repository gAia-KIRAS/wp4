import pytest

from config.config import Config
from config.io_config import IOConfig
from io_manager.io_manager import IO
from modules.intersect_with_aoi import IntersectAOI
from modules.nci import NCI
from utils import TileRef


def test_nci_for_NDVI_raw():
    """
    Test NCI computation for NDVI_raw
    """
    config = Config()
    io_config = IOConfig()
    io = IO(io_config)
    nci = NCI(config, io)

    tile_ref = TileRef(2020, '33TUM', 'NDVI_raw')
    image_refs, _ = io.list_files_on_server(tile_ref, image_type='raw')
    image_1 = image_refs[0]
    image_2 = image_refs[1]

    io.download_file(image_1)
    io.download_file(image_2)

    try:
        _ = nci.compute_and_save_nci(image_1, image_2)
    except Exception:
        pytest.fail(f'NCI computation failed')


def test_nci_for_NDVI_reconstructed():
    """
    Test NCI computation for NDVI_reconstructed
    """
    config = Config()
    io_config = IOConfig()
    io = IO(io_config)
    nci = NCI(config, io)

    tile_ref = TileRef(2020, '33TUM', 'NDVI_reconstructed')
    image_refs, _ = io.list_files_on_server(tile_ref, image_type='raw')
    image_1 = image_refs[0]
    image_2 = image_refs[1]

    io.download_file(image_1)
    io.download_file(image_2)

    try:
        _ = nci.compute_and_save_nci(image_1, image_2)
    except Exception:
        pytest.fail(f'NCI computation failed')


def test_intersect_with_aoi():
    """
    Test intersection with AOI for NDVI_reconstructed
    """
    io_config = IOConfig()
    config = Config()

    io = IO(io_config)
    tile_ref = TileRef(2020, '33TUM', 'NDVI_reconstructed')
    image_refs, _ = io.list_files_on_server(tile_ref, image_type='raw')
    image = image_refs[0]
    io.download_file(image)

    module = IntersectAOI(io, config)

    try:
        intersected = module.intersect(image)
        io.check_existence_on_local(f'{io_config.base_local_dir}/{intersected.rel_filepath()}', dir=False)
    except Exception:
        pytest.fail(f'Intersection with AOI failed')


if __name__ == '__main__':
    pytest.main([''])
