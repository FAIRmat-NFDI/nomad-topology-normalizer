from nomad.client import normalize_all
from nomad.datamodel import EntryArchive, EntryMetadata
from nomad.datamodel.metainfo.workflow import Workflow

# from nomad.normalizing.topology import TopologyNormalizer
from nomad.utils import get_logger

from nomad_topology_normalizer.normalizers.normalizer import TopologyNormalizer

LOGGER = get_logger(__name__)


def test_topology_calculation_2():
    """Test topology_calculation_2 with minimal new schema data."""
    from nomad.datamodel.results import Material, Results
    from nomad_simulations.schema_packages.general import Simulation
    from nomad_simulations.schema_packages.model_system import ModelSystem

    archive = EntryArchive(metadata=EntryMetadata())

    simulation = Simulation()
    model_system = ModelSystem(name='test_system')

    # No sub_systems, topology_calculation_2 should return None
    simulation.model_system.append(model_system)
    archive.data = simulation

    # Initialize results (needed for topology storage)
    archive.results = Results()
    archive.results.material = Material()

    normalizer = TopologyNormalizer()

    # Call normalize to set up entry_archive and other attributes
    normalizer.normalize(archive, LOGGER)

    result = normalizer.topology_calculation_2()
    assert result is None


def test_normalizer():
    entry_archive = EntryArchive(
        metadata=EntryMetadata(), workflow2=Workflow(name='test')
    )
    normalize_all(entry_archive)
    assert entry_archive.workflow2.name == 'test'
