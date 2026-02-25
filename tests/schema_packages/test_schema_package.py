import os.path

from nomad.client import normalize_all, parse


def test_schema_package():
    test_file = os.path.join('tests', 'data', 'test.archive.yaml')
    entry_archive = parse(test_file)[0]
    normalize_all(entry_archive)

    assert entry_archive.data.message == 'Hello Markus!'


def test_second_archive_populates_results_material_topology():
    """Direct v2 System fixture should populate topology with mapped fractions."""
    test_file = os.path.join('tests', 'data', 'second.archive.yaml')
    entry_archive = parse(test_file)[0]
    normalize_all(entry_archive)

    topology = entry_archive.results.material.topology
    assert topology is not None
    assert len(topology) == 5

    root = topology[0]
    imported = topology[1]
    subsystems = topology[2:]

    assert root.label == 'original'
    assert root.system_relation.type == 'root'
    assert root.child_systems == [imported.system_id]

    # The direct SystemV2 entry is imported as a parser node under the root.
    assert imported.label == 'second'
    assert imported.parent_system == root.system_id
    assert len(imported.child_systems) == 3

    assert [node.label for node in subsystems] == [
        'subsystem',
        'subsystem',
        'subsystem',
    ]
    assert [node.parent_system for node in subsystems] == [imported.system_id] * 3
    assert [node.atomic_fraction for node in subsystems] == [0.5, 0.25, 0.25]
