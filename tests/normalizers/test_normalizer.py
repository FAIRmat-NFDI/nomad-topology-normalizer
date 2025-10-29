# from nomad.normalizing.topology import TopologyNormalizer
import numpy as np
from nomad.client import normalize_all
from nomad.datamodel import EntryArchive, EntryMetadata
from nomad.datamodel.metainfo.workflow import Workflow
from nomad.datamodel.results import Material, Results
from nomad.units import ureg
from nomad.utils import get_logger
from nomad_simulations.schema_packages.atoms_state import AtomsState
from nomad_simulations.schema_packages.general import Simulation
from nomad_simulations.schema_packages.model_system import AtomicCell, ModelSystem

from nomad_topology_normalizer.normalizers.normalizer import TopologyNormalizer

LOGGER = get_logger(__name__)


def test_topology_calculation_2():
    """Test topology_calculation_2 with minimal new schema data."""

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


def test_topology_calculation_2_with_subsystem():
    """Test topology_calculation_2 with subsystems."""

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

    # Add cell
    cell = AtomicCell()
    cell.lattice_vectors = np.eye(3) * 10.0 * ureg.angstrom
    cell.periodic_boundary_conditions = [True, True, True]
    root.cell.append(cell)

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

    result = normalizer.topology_calculation_2()

    # Should return a list of System objects
    assert result is not None
    assert isinstance(result, list)
    assert len(result) > 0


def test_topology_calculation_2_nested_subsystems():
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

    # Add cell
    cell = AtomicCell()
    cell.lattice_vectors = np.eye(3) * 10.0 * ureg.angstrom
    cell.periodic_boundary_conditions = [True, True, True]
    root.cell.append(cell)

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
    result = normalizer.topology_calculation_2()

    # Verify nested structure created
    assert result is not None
    assert isinstance(result, list)
    n_systems = 3
    assert len(result) >= n_systems  # original + molecule_group + molecules


def test_topology_calculation_2_multiple_same_label():
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
    result = normalizer.topology_calculation_2()

    assert result is not None
    # Should have original + one system for label 'water' with multiple instances
    systems_dict = {s.label: s for s in result}
    assert 'water' in systems_dict
    water_system = systems_dict['water']
    # Should have 2 instances (indices arrays)
    assert water_system.indices is not None
    n_indices = 2
    assert len(water_system.indices) == n_indices


def test_topology_calculation_2_branch_label_types():
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
    result = normalizer.topology_calculation_2()

    assert result is not None
    systems_dict = {s.label: s for s in result}

    # Verify monomer_group created
    assert 'ethylene_group' in systems_dict
    assert systems_dict['ethylene_group'].structural_type == 'group'

    # Verify monomers created
    assert 'ethylene' in systems_dict
    assert systems_dict['ethylene'].building_block == 'monomer'


def test_topology_calculation_2_no_positions():
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
    result = normalizer.topology_calculation_2()

    # Should return None due to missing positions
    assert result is None


def test_topology_calculation_2_no_particle_states():
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
    result = normalizer.topology_calculation_2()

    # Should return None due to missing particle_states
    assert result is None


def test_topology_calculation_2_mismatched_label_atom_counts():
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
    result = normalizer.topology_calculation_2()
    assert result is not None

    # First instance should be stored, second should be rejected
    systems_dict = {s.label: s for s in result}
    assert 'fragment' in systems_dict
    fragment_system = systems_dict['fragment']
    # Should only have first instance (3 atoms)
    n_groups, n_indices = 1, 3
    assert len(fragment_system.indices) == n_groups
    assert len(fragment_system.indices[0]) == n_indices


def test_topology_calculation_2_cgbead_system():
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
    result = normalizer.topology_calculation_2()

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
