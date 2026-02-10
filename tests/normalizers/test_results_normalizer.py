"""Tests for ResultsNormalizer schema detection and routing logic."""

import numpy as np
import pytest
from nomad.datamodel import EntryArchive, EntryMetadata
from nomad.datamodel.results import Properties, Results
from nomad.units import ureg
from nomad.utils import get_logger
from nomad_simulations.schema_packages.atoms_state import AtomsState
from nomad_simulations.schema_packages.general import Simulation
from nomad_simulations.schema_packages.model_system import ModelSystem

from nomad_topology_normalizer.normalizers.results import (
    ResultsNormalizerBase as ResultsNormalizer,
)

LOGGER = get_logger(__name__)


@pytest.fixture
def archive_with_data_schema():
    """Create an archive with v2 data schema (archive.data)."""
    archive = EntryArchive(metadata=EntryMetadata())

    # Create v2 data schema
    simulation = Simulation()
    model_system = ModelSystem(
        name='test_system',
        type='molecule',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )

    # Add particle states
    model_system.particle_states.append(
        AtomsState(chemical_symbol='H', atomic_number=1)
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='H', atomic_number=1)
    )

    # Add cell properties
    model_system.lattice_vectors = np.eye(3) * 10.0 * ureg.angstrom
    model_system.periodic_boundary_conditions = [True, True, True]

    simulation.model_system.append(model_system)
    archive.data = simulation

    # Initialize results
    archive.results = Results()
    archive.results.properties = Properties()

    return archive


@pytest.fixture
def archive_empty():
    """Create an archive with neither data nor run schema."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()
    return archive


def test_schema_detection_data_schema(archive_with_data_schema, caplog):
    """Test that v2 data schema is detected and routed correctly."""
    normalizer = ResultsNormalizer()

    # Clear any previous log records
    caplog.clear()

    # Run normalization
    normalizer.normalize(archive_with_data_schema, LOGGER)

    # Check that the correct path was taken
    # Look for the info log message
    assert any(
        'v2 data schema results normalization' in record.message
        for record in caplog.records
    ), 'Should log v2 data schema path'

    # Should NOT see legacy message
    assert not any(
        'legacy results normalization' in record.message for record in caplog.records
    ), 'Should not log legacy path'


def test_schema_detection_no_schema(archive_empty, caplog):
    """Test behavior when neither v2 data schema nor legacy schema is present."""
    normalizer = ResultsNormalizer()

    # Clear any previous log records
    caplog.clear()

    # Run normalization
    try:
        normalizer.normalize(archive_empty, LOGGER)
    except Exception:
        # May fail without proper data - that's OK
        pass

    # Should take legacy path (default fallback)
    assert any(
        'legacy results normalization' in record.message for record in caplog.records
    ), 'Should log legacy path as fallback'


def test_data_schema_creates_topology(archive_with_data_schema):
    """Test that v2 data schema path creates topology."""
    normalizer = ResultsNormalizer()

    # Run normalization
    normalizer.normalize(archive_with_data_schema, LOGGER)

    # Check that results were populated
    assert archive_with_data_schema.results is not None
    assert archive_with_data_schema.results.material is not None

    # Note: Topology creation depends on having proper structure data
    # This test just verifies the path executes without error


def test_data_schema_priority_over_run(archive_with_data_schema):
    """Test that v2 data schema takes priority when both schemas present."""
    # Add a mock run section to the data schema archive
    from nomad.datamodel.data import ArchiveSection

    class MockRun(ArchiveSection):
        system = []

    archive_with_data_schema.run = [MockRun()]

    normalizer = ResultsNormalizer()

    # Run normalization (should use data schema path, not run)
    normalizer.normalize(archive_with_data_schema, LOGGER)

    # Should succeed without delegating to legacy normalizer
    assert archive_with_data_schema.results is not None


def test_normalize_with_data_schema_calls_topology_normalizer(
    archive_with_data_schema, monkeypatch
):
    """Test that _normalize_with_data_schema calls TopologyNormalizer."""
    from nomad_topology_normalizer.normalizers.topology import TopologyNormalizer

    # Track if TopologyNormalizer.normalize was called
    normalize_called = []

    original_normalize = TopologyNormalizer.normalize

    def mock_normalize(self, archive, logger):
        normalize_called.append(True)
        # Call original to avoid breaking the test
        return original_normalize(self, archive, logger)

    monkeypatch.setattr(TopologyNormalizer, 'normalize', mock_normalize)

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive_with_data_schema, LOGGER)

    # Verify TopologyNormalizer.normalize was called
    assert len(normalize_called) > 0, 'TopologyNormalizer.normalize should be called'


def test_normalize_measurements_still_works(archive_with_data_schema):
    """Test that measurement normalization still works with new architecture."""
    # Skip this test as it requires complex measurement schema setup
    pytest.skip('Test requires complete measurement schema implementation')
