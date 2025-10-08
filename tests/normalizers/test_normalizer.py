import pytest
from nomad.datamodel import EntryArchive, EntryMetadata
from nomad.datamodel.results import Relation, System

# from nomad.normalizing.topology import TopologyNormalizer
from nomad.utils import get_logger

from nomad_topology_normalizer.normalizers.normalizer import TopologyNormalizer
from src.nomad_topology_normalizer.normalizers.normalizer import (
    add_system,
    add_system_info,
    get_topology_id,
    get_topology_original,
)

LOGGER = get_logger(__name__)


@pytest.fixture
def mock_normalizer(
    entry_archive_with_simulation, mock_repr_system, mock_repr_symmetry, mock_logger
):
    """Create a TopologyNormalizer."""

    normalizer = TopologyNormalizer(
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

    normalizer = TopologyNormalizer(
        entry_archive=archive,
        repr_system=MockSystem(),
        repr_symmetry=None,
        conv_atoms=None,
        logger=LOGGER,
    )

    result = normalizer.topology_calculation_2()
    assert result is not None


def test_get_topology_id_formats_correctly():
    assert get_topology_id(0) == 'results/material/topology/0'
    assert get_topology_id(17) == 'results/material/topology/17'


def test_get_topology_original_minimal_archive():
    archive = EntryArchive()
    original = get_topology_original(atoms=None, archive=archive)

    assert isinstance(original, System)
    assert original.method == 'parser'
    assert original.label == 'original'
    assert original.system_relation == Relation(type='root')
    assert original.dimensionality is None


def test_add_system_creates_parent_child_links_and_ids():
    topo = {}

    root = System(method='parser', label='root', system_relation=Relation(type='root'))
    add_system(root, topo)
    assert root.system_id == 'results/material/topology/0'
    assert topo[root.system_id] is root
    assert root.child_systems in (None, [])

    child1 = System(
        method='parser', label='child1', system_relation=Relation(type='subsystem')
    )
    add_system(child1, topo, parent=root)
    assert child1.parent_system == root.system_id
    assert root.child_systems and child1.system_id in root.child_systems

    child2 = System(
        method='parser', label='child2', system_relation=Relation(type='subsystem')
    )
    add_system(child2, topo, parent=root)
    assert len(root.child_systems) == 2
    assert {child1.system_id, child2.system_id} == set(root.child_systems)


def test_add_system_info_is_safe_without_atoms():
    topo = {}
    sys = System(method='parser', label='no_atoms')
    add_system(sys, topo)
    # Should be a no-op, but must not crash
    add_system_info(sys, topo, masses=None)
