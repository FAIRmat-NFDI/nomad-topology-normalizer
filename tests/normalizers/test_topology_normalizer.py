# from nomad.normalizing.topology import TopologyNormalizer
from types import SimpleNamespace

import numpy as np
from nomad.client import normalize_all
from nomad.datamodel import EntryArchive, EntryMetadata
from nomad.datamodel.metainfo.workflow import Workflow
from nomad.datamodel.results import Material, Relation, Results, System
from nomad.units import ureg
from nomad.utils import get_logger
from nomad_simulations.schema_packages.atoms_state import AtomsState
from nomad_simulations.schema_packages.general import Simulation
from nomad_simulations.schema_packages.model_system import ModelSystem

from nomad_topology_normalizer.normalizers.topology import (
    TopologyNormalizer,
    add_system,
)

LOGGER = get_logger(__name__)


def _make_bulk_model_system() -> ModelSystem:
    system = ModelSystem(is_representative=True, type='bulk')
    system.positions = np.array([[0.0, 0.0, 0.0], [1.35, 1.35, 1.35]]) * ureg.angstrom
    system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    system.periodic_boundary_conditions = [True, True, True]
    system.particle_states.append(AtomsState(chemical_symbol='Si', atomic_number=14))
    system.particle_states.append(AtomsState(chemical_symbol='Si', atomic_number=14))
    return system


def test_topology_calculation():
    """Test topology_calculation with minimal new schema data."""

    archive = EntryArchive(metadata=EntryMetadata())

    simulation = Simulation()
    model_system = ModelSystem(name='test_system')

    # No sub_systems, topology_calculation should return None
    simulation.model_system.append(model_system)
    archive.data = simulation

    # Initialize results (needed for topology storage)
    archive.results = Results()
    archive.results.material = Material()

    normalizer = TopologyNormalizer()

    # Call normalize to set up entry_archive and other attributes
    normalizer.normalize(archive, LOGGER)

    result = normalizer.topology_calculation()
    assert result is None


def test_topology_calculation_with_subsystem():
    """Test topology_calculation with subsystems."""

    # Create archive
    archive = EntryArchive(metadata=EntryMetadata())

    # Create root ModelSystem with subsystems
    root = ModelSystem(
        name='test_system',
        type='molecule',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        * ureg.angstrom,
        n_particles=3,
    )

    # Add particle states
    root.particle_states.append(AtomsState(chemical_symbol='O', atomic_number=8))
    root.particle_states.append(AtomsState(chemical_symbol='H', atomic_number=1))
    root.particle_states.append(AtomsState(chemical_symbol='H', atomic_number=1))

    # Add cell properties directly to ModelSystem (v2 schema)
    root.lattice_vectors = np.eye(3) * 10.0 * ureg.angstrom
    root.periodic_boundary_conditions = [True, True, True]

    # Add a subsystem
    subsystem = ModelSystem(
        name='molecule',
        branch_label='molecule',
        particle_indices=np.array([0, 1, 2], dtype=np.int32),
    )
    root.sub_systems.append(subsystem)

    # Add to simulation
    simulation = Simulation()
    simulation.model_system.append(root)
    archive.data = simulation

    # Initialize results
    archive.results = Results()
    archive.results.material = Material()

    # Create normalizer and normalize
    normalizer = TopologyNormalizer()
    normalizer.normalize(archive, LOGGER)

    result = normalizer.topology_calculation()

    # Should return a list of System objects
    assert result is not None
    assert isinstance(result, list)
    assert len(result) > 0


def test_topology_root_populates_atoms_when_ase_conversion_fails(monkeypatch):
    archive = EntryArchive(metadata=EntryMetadata())

    root = ModelSystem(
        name='test_system',
        type='molecule',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    root.periodic_boundary_conditions = [True, True, True]
    root.particle_states.append(AtomsState(chemical_symbol='H', atomic_number=1))
    root.particle_states.append(AtomsState(chemical_symbol='H', atomic_number=1))
    subsystem = ModelSystem(
        name='molecule',
        branch_label='molecule',
        particle_indices=np.array([0, 1], dtype=np.int32),
    )
    root.sub_systems.append(subsystem)

    simulation = Simulation()
    simulation.model_system.append(root)
    archive.data = simulation
    archive.results = Results()
    archive.results.material = Material()

    def _raise_to_ase_atoms(self):
        raise ValueError('forced failure for fallback path test')

    monkeypatch.setattr(ModelSystem, 'to_ase_atoms', _raise_to_ase_atoms)

    normalizer = TopologyNormalizer()
    normalizer.normalize(archive, LOGGER)

    topology = archive.results.material.topology
    assert topology is not None
    assert len(topology) > 0
    original = topology[0]
    assert original.label == 'original'
    assert original.atoms is not None
    expected_atoms_cls = System.m_def.all_sub_sections['atoms'].sub_section.section_cls
    assert isinstance(original.atoms, expected_atoms_cls)
    assert original.atoms.positions is not None
    assert len(original.atoms.positions) == 2


def test_topology_calculation_prefers_representative_system():
    archive = EntryArchive(metadata=EntryMetadata())

    non_rep = ModelSystem(name='non_rep', is_representative=False)

    representative = ModelSystem(
        name='rep_system',
        type='molecule',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    representative.particle_states.append(
        AtomsState(chemical_symbol='H', atomic_number=1)
    )
    representative.particle_states.append(
        AtomsState(chemical_symbol='H', atomic_number=1)
    )
    representative.lattice_vectors = np.eye(3) * 10.0 * ureg.angstrom
    representative.periodic_boundary_conditions = [True, True, True]
    representative.sub_systems.append(
        ModelSystem(
            name='molecule',
            branch_label='molecule',
            particle_indices=np.array([0, 1], dtype=np.int32),
        )
    )

    simulation = Simulation()
    simulation.model_system.append(non_rep)
    simulation.model_system.append(representative)
    archive.data = simulation
    archive.results = Results()
    archive.results.material = Material()

    normalizer = TopologyNormalizer()
    normalizer.normalize(archive, LOGGER)

    topology = archive.results.material.topology
    assert topology is not None
    assert len(topology) > 0
    original = topology[0]
    assert original.label == 'original'
    assert original.n_atoms == 2


def test_topology_calculation_falls_back_to_topology_bearing_model_system():
    archive = EntryArchive(metadata=EntryMetadata())

    topology_system = ModelSystem(
        name='topology_system',
        type='molecule',
        is_representative=False,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    topology_system.particle_states.append(
        AtomsState(chemical_symbol='H', atomic_number=1)
    )
    topology_system.particle_states.append(
        AtomsState(chemical_symbol='H', atomic_number=1)
    )
    topology_system.sub_systems.append(
        ModelSystem(
            name='molecule',
            branch_label='molecule',
            particle_indices=np.array([0, 1], dtype=np.int32),
        )
    )

    representative_frame = ModelSystem(
        name='trajectory_frame',
        type='molecule',
        is_representative=True,
        positions=np.array([[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    representative_frame.particle_states.append(
        AtomsState(chemical_symbol='H', atomic_number=1)
    )
    representative_frame.particle_states.append(
        AtomsState(chemical_symbol='H', atomic_number=1)
    )

    simulation = Simulation()
    simulation.model_system.append(topology_system)
    simulation.model_system.append(representative_frame)
    simulation.representative_system_index = 1
    archive.data = simulation
    archive.results = Results()
    archive.results.material = Material()

    normalizer = TopologyNormalizer()
    normalizer.normalize(archive, LOGGER)

    topology = archive.results.material.topology
    assert topology is not None
    assert len(topology) > 1
    original = topology[0]
    assert original.label == 'original'
    assert original.n_atoms == 2
    assert any(section.label == 'molecule' for section in topology)


def test_topology_root_normalizes_atomic_numbers_from_symbol_and_nat():
    archive = EntryArchive(metadata=EntryMetadata())

    root = ModelSystem(
        name='test_system',
        type='molecule',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=1,
    )
    root.periodic_boundary_conditions = [False, False, False]
    root.particle_states.append(AtomsState(chemical_symbol='Sr', atomic_number=238))
    root.sub_systems.append(
        ModelSystem(
            name='atom',
            branch_label='molecule',
            particle_indices=np.array([0], dtype=np.int32),
        )
    )

    simulation = Simulation()
    simulation.model_system.append(root)
    archive.data = simulation
    archive.results = Results()
    archive.results.material = Material()

    normalizer = TopologyNormalizer()
    normalizer.normalize(archive, LOGGER)

    topology = archive.results.material.topology
    assert topology is not None
    assert len(topology) > 0
    original = topology[0]
    assert original.atoms is not None
    assert original.atoms.labels[0] == 'Sr'
    assert original.atoms.atomic_numbers[0] == 38


def test_topology_root_converts_unitless_geometry_to_meter_storage():
    archive = EntryArchive(metadata=EntryMetadata())

    root = ModelSystem(
        name='test_system',
        type='bulk',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]),
        n_particles=2,
    )
    root.lattice_vectors = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    root.periodic_boundary_conditions = [True, True, True]
    root.particle_states.append(AtomsState(chemical_symbol='Si', atomic_number=14))
    root.particle_states.append(AtomsState(chemical_symbol='Si', atomic_number=14))
    root.sub_systems.append(
        ModelSystem(
            name='subsystem',
            branch_label='molecule',
            particle_indices=np.array([0, 1], dtype=np.int32),
        )
    )

    simulation = Simulation()
    simulation.model_system.append(root)
    archive.data = simulation
    archive.results = Results()
    archive.results.material = Material()

    normalizer = TopologyNormalizer()
    normalizer.normalize(archive, LOGGER)

    serialized = archive.m_to_dict()
    top0 = serialized['results']['material']['topology'][0]
    positions = np.array(top0['atoms']['positions'])
    lattice_vectors = np.array(top0['atoms']['lattice_vectors'])

    # Unitless payload is interpreted as angstrom-like and stored in meters.
    assert positions[1, 0] == np.float64(1e-10)
    assert positions[1, 1] == np.float64(2e-10)
    assert positions[1, 2] == np.float64(3e-10)
    assert lattice_vectors[0, 0] == np.float64(5e-10)


def test_topology_calculation_nested_subsystems():
    """Test nested hierarchy: root -> molecule_group -> molecule."""

    archive = EntryArchive(metadata=EntryMetadata())

    # Create root system with 6 atoms (2 water molecules)
    root = ModelSystem(
        name='test_system',
        type='molecule',
        is_representative=True,
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],  # H2O molecule 1
                [5.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [5.0, 1.0, 0.0],  # H2O molecule 2
            ]
        )
        * ureg.angstrom,
        n_particles=6,
    )

    # Add particle states
    particle_data = [('O', 8), ('H', 1), ('H', 1), ('O', 8), ('H', 1), ('H', 1)]
    for symbol, atomic_num in particle_data:
        root.particle_states.append(
            AtomsState(chemical_symbol=symbol, atomic_number=atomic_num)
        )

    # Add cell properties directly to ModelSystem (v2 schema)
    root.lattice_vectors = np.eye(3) * 10.0 * ureg.angstrom
    root.periodic_boundary_conditions = [True, True, True]

    # Add molecule_group containing two molecules
    molecule_group = ModelSystem(
        name='water_group',
        branch_label='molecule_group',
        particle_indices=np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
    )

    # Add nested molecules within the group
    mol1 = ModelSystem(
        name='water0',
        branch_label='molecule',
        particle_indices=np.array([0, 1, 2], dtype=np.int32),
    )
    mol2 = ModelSystem(
        name='water',
        branch_label='molecule',
        particle_indices=np.array([3, 4, 5], dtype=np.int32),
    )
    molecule_group.sub_systems.append(mol1)
    molecule_group.sub_systems.append(mol2)

    root.sub_systems.append(molecule_group)

    # Add to simulation
    simulation = Simulation()
    simulation.model_system.append(root)
    archive.data = simulation

    # Initialize results
    archive.results = Results()
    archive.results.material = Material()

    # Normalize and run topology calculation
    normalizer = TopologyNormalizer()
    normalizer.normalize(archive, LOGGER)
    result = normalizer.topology_calculation()

    # Verify nested structure created
    assert result is not None
    assert isinstance(result, list)
    n_systems = 3
    assert len(result) >= n_systems  # original + molecule_group + molecules


def test_topology_calculation_multiple_same_label():
    """Test multiple subsystems with same label (e.g., multiple H2O molecules)."""

    archive = EntryArchive(metadata=EntryMetadata())

    # Create root with 6 atoms
    root = ModelSystem(
        name='test_system',
        type='molecule',
        is_representative=True,
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [5.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [5.0, 1.0, 0.0],
            ]
        )
        * ureg.angstrom,
        n_particles=6,
    )

    # Add particle states
    particle_data = [('O', 8), ('H', 1), ('H', 1), ('O', 8), ('H', 1), ('H', 1)]
    for symbol, atomic_num in particle_data:
        root.particle_states.append(
            AtomsState(chemical_symbol=symbol, atomic_number=atomic_num)
        )

    # Add two molecules with identical labels
    mol1 = ModelSystem(
        name='water',
        branch_label='molecule',
        particle_indices=np.array([0, 1, 2], dtype=np.int32),
    )
    mol2 = ModelSystem(
        name='water',
        branch_label='molecule',
        particle_indices=np.array([3, 4, 5], dtype=np.int32),
    )
    root.sub_systems.append(mol1)
    root.sub_systems.append(mol2)

    simulation = Simulation()
    simulation.model_system.append(root)
    archive.data = simulation

    archive.results = Results()
    archive.results.material = Material()

    normalizer = TopologyNormalizer()
    normalizer.normalize(archive, LOGGER)
    result = normalizer.topology_calculation()

    assert result is not None
    # Should have original + one system for label 'water' with multiple instances
    systems_dict = {s.label: s for s in result}
    assert 'water' in systems_dict
    water_system = systems_dict['water']
    # Should have 2 instances (indices arrays)
    assert water_system.indices is not None
    n_indices = 2
    assert len(water_system.indices) == n_indices


def test_topology_calculation_branch_label_types():
    """Test different branch_label types: monomer, monomer_group."""

    archive = EntryArchive(metadata=EntryMetadata())

    root = ModelSystem(
        name='test_system',
        type='molecule',
        is_representative=True,
        positions=np.array(
            [
                # First ethylene molecule (C2H4)
                [0.0, 0.0, 0.0],  # C1
                [1.34, 0.0, 0.0],  # C2 (C=C bond ~1.34 Å)
                [-0.51, 0.93, 0.0],  # H1
                [-0.51, -0.93, 0.0],  # H2
                [1.85, 0.93, 0.0],  # H3
                [1.85, -0.93, 0.0],  # H4
                # Second ethylene molecule (C2H4), shifted 5 Å along x
                [5.0, 0.0, 0.0],
                [6.34, 0.0, 0.0],
                [4.49, 0.93, 0.0],
                [4.49, -0.93, 0.0],
                [6.85, 0.93, 0.0],
                [6.85, -0.93, 0.0],
            ]
        )
        * ureg.angstrom,
        n_particles=12,
    )

    # Add particle states
    particle_data = [
        ('C', 6),
        ('C', 6),
        ('H', 1),
        ('H', 1),
        ('H', 1),
        ('H', 1),
        ('C', 6),
        ('C', 6),
        ('H', 1),
        ('H', 1),
        ('H', 1),
        ('H', 1),
    ]
    for symbol, atomic_num in particle_data:
        root.particle_states.append(
            AtomsState(chemical_symbol=symbol, atomic_number=atomic_num)
        )

    # Add monomer_group containing both ethylene molecules
    monomer_group = ModelSystem(
        name='ethylene_group',
        branch_label='monomer_group',
        particle_indices=np.array(list(range(12)), dtype=np.int32),
    )

    # Add monomers within group
    monomer1 = ModelSystem(
        name='ethylene',
        branch_label='monomer',
        particle_indices=np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
    )
    monomer2 = ModelSystem(
        name='ethylene',
        branch_label='monomer',
        particle_indices=np.array([6, 7, 8, 9, 10, 11], dtype=np.int32),
    )
    monomer_group.sub_systems.append(monomer1)
    monomer_group.sub_systems.append(monomer2)

    root.sub_systems.append(monomer_group)

    simulation = Simulation()
    simulation.model_system.append(root)
    archive.data = simulation

    archive.results = Results()
    archive.results.material = Material()

    normalizer = TopologyNormalizer()
    normalizer.normalize(archive, LOGGER)
    result = normalizer.topology_calculation()

    assert result is not None
    systems_dict = {s.label: s for s in result}

    # Verify monomer_group created
    assert 'ethylene_group' in systems_dict
    assert systems_dict['ethylene_group'].structural_type == 'group'

    # Verify monomers created
    assert 'ethylene' in systems_dict
    assert systems_dict['ethylene'].building_block == 'monomer'


def test_topology_calculation_no_positions():
    """Test system with particle_states but no positions - should return None."""

    archive = EntryArchive(metadata=EntryMetadata())

    root = ModelSystem(
        name='test_system',
        type='molecule',
        is_representative=True,
        n_particles=3,
    )

    # Add particle states but NO positions
    root.particle_states.append(AtomsState(chemical_symbol='O', atomic_number=8))
    root.particle_states.append(AtomsState(chemical_symbol='H', atomic_number=1))
    root.particle_states.append(AtomsState(chemical_symbol='H', atomic_number=1))

    subsystem = ModelSystem(
        name='molecule',
        branch_label='molecule',
        particle_indices=np.array([0, 1, 2], dtype=np.int32),
    )
    root.sub_systems.append(subsystem)

    simulation = Simulation()
    simulation.model_system.append(root)
    archive.data = simulation

    archive.results = Results()
    archive.results.material = Material()

    normalizer = TopologyNormalizer()
    normalizer.normalize(archive, LOGGER)
    result = normalizer.topology_calculation()

    # Should return None due to missing positions
    assert result is None


def test_topology_calculation_no_particle_states():
    """Test system with positions but no particle_states - should return None."""

    archive = EntryArchive(metadata=EntryMetadata())

    root = ModelSystem(
        name='test_system',
        type='molecule',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        * ureg.angstrom,
        n_particles=3,
    )
    # NO particle_states added

    subsystem = ModelSystem(
        name='molecule',
        branch_label='molecule',
        particle_indices=np.array([0, 1, 2], dtype=np.int32),
    )
    root.sub_systems.append(subsystem)

    simulation = Simulation()
    simulation.model_system.append(root)
    archive.data = simulation

    archive.results = Results()
    archive.results.material = Material()

    normalizer = TopologyNormalizer()
    normalizer.normalize(archive, LOGGER)
    result = normalizer.topology_calculation()

    # Should return None due to missing particle_states
    assert result is None


def test_topology_calculation_mismatched_label_atom_counts():
    """Test same label but different atom counts - should log warning."""

    archive = EntryArchive(metadata=EntryMetadata())

    root = ModelSystem(
        name='test_system',
        type='molecule',
        is_representative=True,
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [5.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
            ]
        )
        * ureg.angstrom,
        n_particles=5,
    )

    for _ in range(5):
        root.particle_states.append(AtomsState(chemical_symbol='H', atomic_number=1))

    # First molecule with 3 atoms
    mol1 = ModelSystem(
        name='fragment',
        branch_label='molecule',
        particle_indices=np.array([0, 1, 2], dtype=np.int32),
    )
    # Second molecule with 2 atoms but same label - should trigger warning
    mol2 = ModelSystem(
        name='fragment',
        branch_label='molecule',
        particle_indices=np.array([3, 4], dtype=np.int32),
    )
    root.sub_systems.append(mol1)
    root.sub_systems.append(mol2)

    simulation = Simulation()
    simulation.model_system.append(root)
    archive.data = simulation

    archive.results = Results()
    archive.results.material = Material()

    normalizer = TopologyNormalizer()
    normalizer.normalize(archive, LOGGER)

    # Should not crash, may log warning
    result = normalizer.topology_calculation()
    assert result is not None

    # First instance should be stored, second should be rejected
    systems_dict = {s.label: s for s in result}
    assert 'fragment' in systems_dict
    fragment_system = systems_dict['fragment']
    # Should only have first instance (3 atoms)
    n_groups, n_indices = 1, 3
    assert len(fragment_system.indices) == n_groups
    assert len(fragment_system.indices[0]) == n_indices


def test_topology_calculation_cgbead_system():
    """Test coarse-grained system with CGBeadState particles and mass processing."""
    from nomad_simulations.schema_packages.atoms_state import CGBeadState

    archive = EntryArchive(metadata=EntryMetadata())

    root = ModelSystem(
        name='test_system',
        type='molecule',
        is_representative=True,
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ]
        )
        * ureg.angstrom,
        n_particles=3,
    )

    # Add coarse-grained beads with explicit masses
    bead1 = CGBeadState(bead_name='CG_A', mass=100.0 * ureg.amu)
    bead2 = CGBeadState(bead_name='CG_B', mass=150.0 * ureg.amu)
    bead3 = CGBeadState(bead_name='CG_C', mass=250.0 * ureg.amu)
    root.particle_states.append(bead1)
    root.particle_states.append(bead2)
    root.particle_states.append(bead3)

    # Add subsystem with first two beads (total mass: 250 amu)
    subsystem = ModelSystem(
        name='cg_molecule',
        branch_label='molecule',
        particle_indices=np.array([0, 1], dtype=np.int32),
    )
    root.sub_systems.append(subsystem)

    simulation = Simulation()
    simulation.model_system.append(root)
    archive.data = simulation

    archive.results = Results()
    archive.results.material = Material()

    normalize_all(archive)

    # Then run topology normalizer
    normalizer = TopologyNormalizer()
    normalizer.normalize(archive, LOGGER)
    result = normalizer.topology_calculation()

    # Should handle CG systems
    assert result is not None
    assert isinstance(result, list)
    systems_dict = {s.label: s for s in result}
    assert 'cg_molecule' in systems_dict

    # Check mass-related properties
    cg_mol = systems_dict['cg_molecule']

    # Verify n_atoms is set correctly (2 beads)
    n_beads = 2
    assert cg_mol.n_atoms == n_beads

    # Verify atomic_fraction is calculated (2 out of 3 particles)
    expected_atomic_fraction = 2.0 / 3.0
    diff_threshold = 1e-6
    assert cg_mol.atomic_fraction is not None
    assert abs(cg_mol.atomic_fraction - expected_atomic_fraction) < diff_threshold

    # Check if mass_fraction is populated from upstream v2 normalizers
    # Total mass: 100 + 150 + 250 = 500 amu
    # Subsystem mass: 100 + 150 = 250 amu
    # Expected mass_fraction: 250 / 500 = 0.5
    if cg_mol.mass_fraction is not None:
        expected_mass_fraction = 0.5
        assert abs(cg_mol.mass_fraction - expected_mass_fraction) < diff_threshold

    # Verify original system also has correct total particles
    original = systems_dict.get('original')
    if original:
        assert original.n_atoms == n_beads + 1  # 3 beads


def test_normalizer():
    entry_archive = EntryArchive(
        metadata=EntryMetadata(), workflow2=Workflow(name='test')
    )
    normalize_all(entry_archive)
    assert entry_archive.workflow2.name == 'test'


def test_topology_bulk_prefers_v2_symmetry_no_matid_recompute(monkeypatch):
    normalizer = TopologyNormalizer()
    normalizer.logger = LOGGER
    normalizer.masses = None
    normalizer.conv_atoms = _make_bulk_model_system().to_ase_atoms()
    normalizer.repr_system = _make_bulk_model_system()
    normalizer.repr_system.type = 'bulk'
    normalizer.repr_system.symmetry = SimpleNamespace(
        hall_number=523,
        hall_symbol='-F 4 2 3',
        bravais_lattice='cF',
        crystal_system='cubic',
        space_group_number=225,
        space_group_symbol='Fm-3m',
        point_group_symbol='m-3m',
    )
    normalizer.repr_system.local_symmetry = None
    normalizer.repr_symmetry = SimpleNamespace(
        m_cache={
            'symmetry_analyzer': SimpleNamespace(
                get_space_group_number=lambda: 229,
                get_wyckoff_sets_conventional=lambda: [],
            )
        }
    )

    def fail_create_symmetry(_):
        raise AssertionError('MatID symmetry recomputation should not be used')

    monkeypatch.setattr(normalizer, '_create_symmetry', fail_create_symmetry)

    topology = {}
    original = System(
        label='original',
        system_relation=Relation(type='root'),
        n_atoms=len(normalizer.repr_system.particle_states),
    )
    add_system(original, topology)
    material = Material(material_id='v2-material-id')

    normalizer._topology_bulk(original, topology, material)

    conv_system = next(
        item for item in topology.values() if item.label == 'conventional cell'
    )
    assert conv_system.symmetry.space_group_number == 225
    assert conv_system.symmetry.space_group_symbol == 'Fm-3m'
    assert conv_system.material_id == 'v2-material-id'


def test_topology_bulk_uses_matid_for_material_id_fallback(monkeypatch):
    normalizer = TopologyNormalizer()
    normalizer.logger = LOGGER
    normalizer.masses = None
    normalizer.conv_atoms = _make_bulk_model_system().to_ase_atoms()
    normalizer.repr_system = _make_bulk_model_system()
    normalizer.repr_system.type = 'bulk'
    normalizer.repr_system.symmetry = SimpleNamespace(
        hall_number=523,
        hall_symbol='-F 4 2 3',
        bravais_lattice='cF',
        crystal_system='cubic',
        space_group_number=225,
        space_group_symbol='Fm-3m',
        point_group_symbol=None,
    )
    normalizer.repr_system.local_symmetry = None

    analyzer = SimpleNamespace(
        get_space_group_number=lambda: 229,
        get_wyckoff_sets_conventional=lambda: [],
    )
    normalizer.repr_symmetry = SimpleNamespace(
        point_group='m-3m',
        m_cache={'symmetry_analyzer': analyzer},
    )
    called = {'value': False}

    def create_symmetry(_):
        called['value'] = True

    monkeypatch.setattr(normalizer, '_create_symmetry', create_symmetry)

    topology = {}
    original = System(
        label='original',
        system_relation=Relation(type='root'),
        n_atoms=len(normalizer.repr_system.particle_states),
    )
    add_system(original, topology)

    normalizer._topology_bulk(original, topology, Material())

    assert called['value'] is False
    conv_system = next(
        item for item in topology.values() if item.label == 'conventional cell'
    )
    assert conv_system.material_id is not None


def test_topology_runs_matid_for_bulk_with_complete_v2_symmetry(monkeypatch):
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results(material=Material(structural_type='bulk'))
    system = _make_bulk_model_system()
    system.symmetry = SimpleNamespace(
        hall_number=523,
        hall_symbol='-F 4 2 3',
        bravais_lattice='cF',
        crystal_system='cubic',
        space_group_number=225,
        space_group_symbol='Fm-3m',
        point_group_symbol='m-3m',
    )
    simulation = Simulation()
    simulation.model_system.append(system)
    archive.data = simulation

    normalizer = TopologyNormalizer()
    normalizer.entry_archive = archive
    normalizer.repr_system = system
    normalizer.logger = LOGGER

    monkeypatch.setattr(normalizer, 'topology_calculation', lambda: None)

    called = {'value': False}

    def run_matid(_):
        called['value'] = True
        return ['topology-matid']

    monkeypatch.setattr(normalizer, 'topology_matid', run_matid)
    monkeypatch.setattr(normalizer, 'topology_data', lambda *_: ['topology-data'])

    result = normalizer.topology(archive.results.material, system_v2=system)

    assert called['value'] is True
    assert result == ['topology-matid']


def test_complete_v2_bulk_symmetry_gets_material_id():
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results(material=Material(structural_type='bulk'))
    system = _make_bulk_model_system()
    system.symmetry = SimpleNamespace(
        hall_number=523,
        hall_symbol='-F 4 2 3',
        bravais_lattice='cF',
        crystal_system='cubic',
        space_group_number=225,
        space_group_symbol='Fm-3m',
        point_group_symbol='m-3m',
    )
    archive.data = Simulation(model_system=[system])

    TopologyNormalizer().normalize(archive, LOGGER)

    assert archive.results.material.material_id is not None


def test_topology_runs_matid_for_bulk_with_incomplete_v2_symmetry(monkeypatch):
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results(material=Material(structural_type='bulk'))
    system = _make_bulk_model_system()
    system.symmetry = SimpleNamespace(
        hall_number=523,
        hall_symbol='-F 4 2 3',
        bravais_lattice='cF',
        crystal_system='cubic',
        space_group_number=225,
        space_group_symbol='Fm-3m',
        point_group_symbol=None,
    )
    simulation = Simulation()
    simulation.model_system.append(system)
    archive.data = simulation

    normalizer = TopologyNormalizer()
    normalizer.entry_archive = archive
    normalizer.repr_system = system
    normalizer.logger = LOGGER

    called = {'value': False}
    monkeypatch.setattr(normalizer, 'topology_calculation', lambda: None)

    def run_matid(_):
        called['value'] = True
        return ['topology-matid']

    monkeypatch.setattr(normalizer, 'topology_matid', run_matid)
    monkeypatch.setattr(normalizer, 'topology_data', lambda *_: ['topology-data'])

    result = normalizer.topology(archive.results.material, system_v2=system)

    assert called['value'] is True
    assert result == ['topology-matid']
