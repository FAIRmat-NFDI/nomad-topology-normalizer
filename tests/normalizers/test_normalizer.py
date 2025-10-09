import pytest
from nomad.client import normalize_all
from nomad.datamodel import EntryArchive, EntryMetadata
from nomad.datamodel.metainfo.workflow import Workflow

# from nomad.normalizing.topology import TopologyNormalizer
from nomad.utils import get_logger

from nomad_topology_normalizer.normalizers.normalizer import TopologyNormalizer2

LOGGER = get_logger(__name__)


@pytest.fixture
def mock_normalizer(
    entry_archive_with_simulation, mock_repr_system, mock_repr_symmetry, mock_logger
):
    """Create a TopologyNormalizer."""

    normalizer = TopologyNormalizer2(
        entry_archive=entry_archive_with_simulation,
        repr_system=mock_repr_system,
        repr_symmetry=mock_repr_symmetry,
        conv_atoms=None,
        logger=mock_logger,
    )

    return normalizer


def test_topology_calculation_2():
    archive = EntryArchive(metadata=EntryMetadata())

    class MockSystem:
        atoms = None

    normalizer = TopologyNormalizer2(
        entry_archive=archive,
        repr_system=MockSystem(),
        repr_symmetry=None,
        conv_atoms=None,
        logger=LOGGER,
    )

    result = normalizer.topology_calculation_2()
    assert result is not None


def test_normalizer():
    entry_archive = EntryArchive(
        metadata=EntryMetadata(), workflow2=Workflow(name='test')
    )
    normalize_all(entry_archive)
    assert entry_archive.workflow2.name == 'test'
