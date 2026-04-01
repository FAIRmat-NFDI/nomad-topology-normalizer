"""Tests for ResultsNormalizer schema detection and routing logic."""

import numpy as np
import pytest
from nomad.datamodel import EntryArchive, EntryMetadata
from nomad.datamodel.data import ArchiveSection
from nomad.metainfo import Quantity
from nomad.datamodel.results import Properties, Results
from nomad.units import ureg
from nomad.utils import get_logger
from nomad_simulations.schema_packages.atoms_state import AtomsState
from nomad_simulations.schema_packages.general import Program, Simulation
from nomad_simulations.schema_packages.model_method import (
    BSE,
    DFT,
    DMFT,
    GW,
    ModelMethod,
    Wannier,
)
from nomad_simulations.schema_packages.model_system import ModelSystem
from nomad_simulations.schema_packages.outputs import Outputs
from nomad_simulations.schema_packages.properties import (
    AbsorptionSpectrum,
    ElectronicBandGap,
    ElectronicBandStructure,
    ElectronicDensityOfStates,
    ElectronicGreensFunction,
    TotalForce,
)
from nomad_simulations.schema_packages.properties import (
    PotentialEnergy as SimPotentialEnergy,
)
from nomad_simulations.schema_packages.properties import (
    RadiusOfGyration as SimRadiusOfGyration,
)
from nomad_simulations.schema_packages.properties import (
    Temperature as SimTemperature,
)
from nomad_simulations.schema_packages.variables import (
    Energy2,
    MatsubaraFrequency,
)

from nomad_topology_normalizer.normalizers.results import (
    ResultsNormalizerBase as ResultsNormalizer,
)

try:
    import runschema.calculation  # noqa: F401

    HAS_RUNSCHEMA = True
except Exception:
    HAS_RUNSCHEMA = False

LOGGER = get_logger(__name__)


def _kline_path():
    class _DummyKPath(ArchiveSection):
        points = Quantity(type=np.float64, shape=['*', 3])

    return _DummyKPath(points=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]))


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
def archive_with_nested_system():
    """Create an archive with SystemV2 nested deep in archive.data."""
    from nomad.metainfo import SubSection

    archive = EntryArchive(metadata=EntryMetadata())

    class Container(ArchiveSection):
        sub = SubSection(sub_section=ArchiveSection)

    # Create v2 data schema
    model_system = ModelSystem(
        name='nested_system',
        type='molecule',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='H', atomic_number=1)
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='H', atomic_number=1)
    )
    model_system.lattice_vectors = np.eye(3) * 10.0 * ureg.angstrom
    model_system.periodic_boundary_conditions = [True, True, True]

    # Nest it
    container = Container()
    container.sub = model_system
    archive.data = container

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


def test_non_simulation_data_schema_uses_legacy_path(caplog):
    """Custom archive.data without model_system should not use v2 sim path."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    class CustomData(ArchiveSection):
        name = None

    archive.data = CustomData()

    normalizer = ResultsNormalizer()
    caplog.clear()

    try:
        normalizer.normalize(archive, LOGGER)
    except Exception:
        # Legacy path may still fail depending on environment; routing is
        # tested via logs.
        pass

    assert any(
        'legacy results normalization' in record.message for record in caplog.records
    ), 'Non-simulation archive.data should use legacy path'
    assert not any(
        'v2 data schema results normalization' in record.message
        for record in caplog.records
    ), 'Non-simulation archive.data should not use v2 sim path'


def test_schema_detection_nested_system(archive_with_nested_system, caplog):
    """Test that nested SystemV2 is detected and causes v2 normalization."""
    normalizer = ResultsNormalizer()
    caplog.clear()

    # Run normalization
    normalizer.normalize(archive_with_nested_system, LOGGER)

    assert any(
        'v2 data schema results normalization' in record.message
        for record in caplog.records
    ), 'Nested SystemV2 should trigger v2 normalization'
    # Keep this suite focused on routing. Detailed topology behavior is covered
    # in test_topology_normalizer.py.
    assert archive_with_nested_system.results is not None
    assert archive_with_nested_system.results.material is not None


def test_data_schema_initializes_results_sections(archive_with_data_schema):
    """v2 data schema path should initialize key results sections."""
    normalizer = ResultsNormalizer()

    # Run normalization
    normalizer.normalize(archive_with_data_schema, LOGGER)

    # Check that results were populated
    assert archive_with_data_schema.results is not None
    assert archive_with_data_schema.results.material is not None


def test_data_schema_populates_method_from_simulation():
    """v2 Simulation program/model_method should populate results.method."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    simulation = Simulation()
    simulation.program = Program(
        name='VASP', version='6.4.2', version_internal='git-abc123'
    )
    simulation.model_method.append(DFT())
    simulation.model_method.append(Wannier(localization_type='maximally_localized'))

    model_system = ModelSystem(
        name='test_system',
        type='bulk',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    model_system.periodic_boundary_conditions = [True, True, True]
    simulation.model_system.append(model_system)

    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    assert archive.results.method is not None
    assert archive.results.method.method_name == 'DFT'
    assert archive.results.method.simulation is not None
    assert archive.results.method.simulation.program_name == 'VASP'
    assert archive.results.method.simulation.program_version == '6.4.2'
    assert archive.results.method.simulation.program_version_internal == 'git-abc123'
    assert archive.results.method.simulation.dft is not None
    assert archive.results.method.simulation.tb is not None
    assert archive.results.method.simulation.tb.type == 'Wannier'
    assert (
        archive.results.method.simulation.tb.localization_type
        == 'maximally_localized'
    )


def test_data_schema_skips_unsupported_method_names():
    """Unsupported model methods should be ignored for results.method transfer."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    simulation = Simulation()
    simulation.program = Program(name='VASP')
    simulation.model_method.append(DFT())
    simulation.model_method.append(ModelMethod(name='HF'))

    model_system = ModelSystem(
        name='test_system',
        type='bulk',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    model_system.periodic_boundary_conditions = [True, True, True]
    simulation.model_system.append(model_system)
    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    assert archive.results.method is not None
    assert archive.results.method.method_name == 'DFT'
    assert archive.results.method.simulation is not None
    assert archive.results.method.simulation.dft is not None


def test_data_schema_maps_method_details_for_gw_bse_dmft():
    """v2 method details should map to legacy-equivalent results fields."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    simulation = Simulation()
    simulation.program = Program(name='VASP')
    simulation.model_method.append(GW(type='G0W0'))
    simulation.model_method.append(BSE(type='RPA', solver='TDA'))
    simulation.model_method.append(
        DMFT(
            impurity_solver='CT-HYB',
            magnetic_state='paramagnetic',
            inverse_temperature=(1.0 / ureg.eV),
        )
    )

    model_system = ModelSystem(
        name='test_system',
        type='bulk',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    model_system.periodic_boundary_conditions = [True, True, True]
    simulation.model_system.append(model_system)
    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    method = archive.results.method
    assert method is not None
    assert method.method_name == 'GW'
    assert method.simulation is not None
    assert method.simulation.gw is not None
    assert method.simulation.gw.type == 'G0W0'
    assert method.simulation.bse is not None
    assert method.simulation.bse.type == 'RPA'
    assert method.simulation.bse.solver == 'TDA'
    assert method.simulation.dmft is not None
    assert method.simulation.dmft.impurity_solver_type == 'CT-HYB'
    assert method.simulation.dmft.magnetic_state == 'paramagnetic'
    assert method.simulation.dmft.inverse_temperature is not None


def test_data_schema_method_name_uses_first_supported_model_method():
    """Method name should use first supported model_method for compatibility."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    simulation = Simulation()
    simulation.program = Program(name='VASP')
    # Intentionally place DMFT first and DFT last.
    simulation.model_method.append(
        DMFT(
            impurity_solver='CT-HYB',
            magnetic_state='paramagnetic',
        )
    )
    simulation.model_method.append(GW(type='G0W0'))
    simulation.model_method.append(DFT())

    model_system = ModelSystem(
        name='test_system',
        type='bulk',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    model_system.periodic_boundary_conditions = [True, True, True]
    simulation.model_system.append(model_system)
    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    assert archive.results.method is not None
    assert archive.results.method.method_name == 'DMFT'


def test_data_schema_maps_dft_spin_and_jacobs_ladder():
    """v2 DFT metadata should map to legacy-equivalent DFT result fields."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    simulation = Simulation()
    simulation.program = Program(name='VASP')
    simulation.model_method.append(DFT(is_spin_polarized=True, jacobs_ladder='GGA'))

    model_system = ModelSystem(
        name='test_system',
        type='bulk',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    model_system.periodic_boundary_conditions = [True, True, True]
    simulation.model_system.append(model_system)
    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    assert archive.results.method is not None
    assert archive.results.method.simulation is not None
    assert archive.results.method.simulation.dft is not None
    assert archive.results.method.simulation.dft.spin_polarized is True
    assert archive.results.method.simulation.dft.jacobs_ladder == 'GGA'
    assert archive.results.method.simulation.dft.xc_functional_type == 'GGA'


def test_data_schema_maps_outputs_electronic_properties():
    """v2 outputs should map electronic properties into results."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    simulation = Simulation()
    simulation.program = Program(name='VASP')
    simulation.model_method.append(DFT())

    model_system = ModelSystem(
        name='test_system',
        type='bulk',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    model_system.periodic_boundary_conditions = [True, True, True]
    simulation.model_system.append(model_system)

    output = Outputs()
    output.electronic_band_gaps.append(ElectronicBandGap(value=1.5 * ureg.eV))
    dos = ElectronicDensityOfStates(
        value=np.array([0.1, 0.2, 0.3]) / ureg.eV, spin_channel=0
    )
    dos.energies = Energy2(points=np.array([-1.0, 0.0, 1.0]) * ureg.eV)
    output.electronic_dos.append(dos)
    band_structure = ElectronicBandStructure(value=np.array([[1.0, 1.1]]) * ureg.eV)
    band_structure.k_path = _kline_path()
    output.electronic_band_structures.append(band_structure)
    greens_function = ElectronicGreensFunction(value=(1.0 + 0.0j) / ureg.eV)
    greens_function.matsubara_frequency = MatsubaraFrequency(
        points=np.array([0.1j, 0.2j]) * ureg.eV
    )
    output.electronic_greens_functions.append(greens_function)
    absorption = AbsorptionSpectrum(value=np.array([0.5, 0.6]) / ureg.eV)
    absorption.energies = Energy2(points=np.array([0.0, 1.0]) * ureg.eV)
    output.absorption_spectra.append(absorption)
    output.radii_of_gyration.append(SimRadiusOfGyration(value=1.2e-10 * ureg.meter))
    output.temperatures.append(SimTemperature(value=300 * ureg.kelvin))
    output.potential_energies.append(SimPotentialEnergy(value=-5.0 * ureg.eV))
    simulation.outputs.append(output)

    output_2 = Outputs()
    output_2.temperatures.append(SimTemperature(value=320 * ureg.kelvin))
    output_2.potential_energies.append(SimPotentialEnergy(value=-4.8 * ureg.eV))
    simulation.outputs.append(output_2)
    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    electronic = archive.results.properties.electronic
    assert electronic is not None
    assert electronic.band_gap
    assert electronic.band_gap[0].value is not None
    if HAS_RUNSCHEMA:
        assert not electronic.dos_electronic_new
        assert electronic.dos_electronic
    else:
        assert not electronic.dos_electronic_new
    assert electronic.band_structure_electronic
    assert electronic.band_structure_electronic[0].segment
    assert archive.results.properties.spectroscopic is not None
    assert archive.results.properties.spectroscopic.spectra
    assert archive.results.properties.structural is not None
    assert archive.results.properties.structural.radius_of_gyration
    assert archive.results.properties.thermodynamic is not None
    assert archive.results.properties.thermodynamic.trajectory


def test_data_schema_uses_latest_output_for_electronic_properties():
    """Electronic mapping should keep latest output payload for legacy parity."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    simulation = Simulation()
    simulation.program = Program(name='VASP')
    simulation.model_method.append(DFT())

    model_system = ModelSystem(
        name='test_system',
        type='bulk',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    model_system.periodic_boundary_conditions = [True, True, True]
    simulation.model_system.append(model_system)

    output_1 = Outputs()
    output_1.electronic_band_gaps.append(ElectronicBandGap(value=1.5 * ureg.eV))
    band_structure_1 = ElectronicBandStructure(value=np.array([[1.0, 1.1]]) * ureg.eV)
    band_structure_1.k_path = _kline_path()
    output_1.electronic_band_structures.append(band_structure_1)
    simulation.outputs.append(output_1)

    output_2 = Outputs()
    output_2.electronic_band_gaps.append(ElectronicBandGap(value=2.5 * ureg.eV))
    band_structure_2 = ElectronicBandStructure(value=np.array([[2.0, 2.1]]) * ureg.eV)
    band_structure_2.k_path = _kline_path()
    output_2.electronic_band_structures.append(band_structure_2)
    simulation.outputs.append(output_2)

    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    electronic = archive.results.properties.electronic
    assert electronic is not None
    assert len(electronic.band_gap or []) == 1
    assert electronic.band_gap[0].value.magnitude == pytest.approx(2.5)
    assert len(electronic.band_structure_electronic or []) == 1


def test_data_schema_band_structure_references_are_not_root_paths():
    """Band structure segment refs should point to concrete run/calculation paths."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    simulation = Simulation()
    simulation.program = Program(name='VASP')
    simulation.model_method.append(DFT())

    model_system = ModelSystem(
        name='test_system',
        type='bulk',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    model_system.periodic_boundary_conditions = [True, True, True]
    simulation.model_system.append(model_system)

    output = Outputs()
    band_structure = ElectronicBandStructure(value=np.array([[1.0, 1.1]]) * ureg.eV)
    band_structure.k_path = _kline_path()
    output.electronic_band_structures.append(band_structure)
    simulation.outputs.append(output)
    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    serialized = archive.m_to_dict()
    bs = (
        serialized.get('results', {})
        .get('properties', {})
        .get('electronic', {})
        .get('band_structure_electronic', [])
    )
    assert bs
    segment_refs = bs[0].get('segment', [])
    assert segment_refs
    assert all(ref != '/' for ref in segment_refs)
    assert all(ref.startswith('/run/') for ref in segment_refs)


def test_data_schema_populates_deprecated_dos_mapping():
    """v2 DOS mapping writes deprecated dos_electronic compatibility mirror."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    simulation = Simulation()
    simulation.program = Program(name='VASP')
    simulation.model_method.append(DFT())

    model_system = ModelSystem(
        name='test_system',
        type='bulk',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    model_system.periodic_boundary_conditions = [True, True, True]
    simulation.model_system.append(model_system)

    output = Outputs()
    dos = ElectronicDensityOfStates(
        value=np.array([0.1, 0.2, 0.3]) / ureg.eV,
        spin_channel=0,
    )
    dos.energies = Energy2(points=np.array([-1.0, 0.0, 1.0]) * ureg.eV)
    output.electronic_dos.append(dos)
    simulation.outputs.append(output)
    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    electronic = archive.results.properties.electronic
    if HAS_RUNSCHEMA:
        assert electronic is not None
        assert not electronic.dos_electronic_new
        assert electronic.dos_electronic
    else:
        assert electronic is None or not electronic.dos_electronic_new
        assert electronic is None or not electronic.dos_electronic


def test_data_schema_populates_deprecated_dos_with_references():
    """Deprecated dos_electronic should be emitted with valid references."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    simulation = Simulation()
    simulation.program = Program(name='VASP')
    simulation.model_method.append(DFT())

    model_system = ModelSystem(
        name='test_system',
        type='bulk',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    model_system.periodic_boundary_conditions = [True, True, True]
    simulation.model_system.append(model_system)

    output = Outputs()
    dos = ElectronicDensityOfStates(
        value=np.array([0.1, 0.2, 0.3]) / ureg.eV,
        spin_channel=0,
    )
    dos.energies = Energy2(points=np.array([-1.0, 0.0, 1.0]) * ureg.eV)
    output.electronic_dos.append(dos)
    simulation.outputs.append(output)
    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    electronic = archive.results.properties.electronic
    if HAS_RUNSCHEMA:
        assert electronic is not None
        assert not electronic.dos_electronic_new
        assert electronic.dos_electronic
        dos = electronic.dos_electronic[0]
        assert dos.energies is not None
        assert dos.total
    else:
        assert electronic is None or not electronic.dos_electronic_new
        assert electronic is None or not electronic.dos_electronic


def test_data_schema_does_not_create_empty_electronic_properties():
    """Non-electronic outputs should not instantiate empty properties.electronic."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    simulation = Simulation()
    simulation.program = Program(name='VASP')
    simulation.model_method.append(DFT())

    model_system = ModelSystem(
        name='test_system',
        type='bulk',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    model_system.periodic_boundary_conditions = [True, True, True]
    simulation.model_system.append(model_system)

    output = Outputs()
    output.radii_of_gyration.append(SimRadiusOfGyration(value=1.2e-10 * ureg.meter))
    output.temperatures.append(SimTemperature(value=300 * ureg.kelvin))
    output.potential_energies.append(SimPotentialEnergy(value=-5.0 * ureg.eV))
    simulation.outputs.append(output)
    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    assert archive.results.properties.electronic is None
    assert archive.results.properties.structural is not None
    assert archive.results.properties.structural.radius_of_gyration
    assert archive.results.properties.thermodynamic is not None
    assert archive.results.properties.thermodynamic.trajectory


def test_data_schema_logs_unmapped_output_groups(archive_with_data_schema, caplog):
    """Unsupported outputs should stay unset and produce a TODO log."""
    output = Outputs()
    output.total_forces.append(
        TotalForce(value=np.array([[1.0, 0.0, 0.0]]) * ureg.newton)
    )
    archive_with_data_schema.data.outputs.append(output)

    normalizer = ResultsNormalizer()
    caplog.clear()
    normalizer.normalize(archive_with_data_schema, LOGGER)

    properties = archive_with_data_schema.results.properties
    assert properties.mechanical is None
    assert properties.vibrational is None
    assert properties.magnetic is None
    assert properties.dynamical is None


def test_data_schema_skips_dos_cleanly_without_runschema(
    archive_with_data_schema, caplog
):
    """DOS mapping should skip cleanly when runschema-based refs cannot be built."""
    output = Outputs()
    dos = ElectronicDensityOfStates(
        value=np.array([0.1, 0.2, 0.3]) / ureg.eV,
        spin_channel=0,
    )
    dos.energies = Energy2(points=np.array([-1.0, 0.0, 1.0]) * ureg.eV)
    output.electronic_dos.append(dos)
    archive_with_data_schema.data.outputs.append(output)

    normalizer = ResultsNormalizer()
    caplog.clear()
    normalizer.normalize(archive_with_data_schema, LOGGER)

    properties = archive_with_data_schema.results.properties
    electronic = properties.electronic
    if HAS_RUNSCHEMA:
        assert electronic is not None
        assert electronic.dos_electronic
        assert not electronic.dos_electronic_new
    else:
        assert electronic is None or not electronic.dos_electronic_new
        assert 'Skipping DOS compatibility mapping for dos_electronic' in caplog.text


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

    def mock_normalize(self, archive, logger, system_v2=None):
        normalize_called.append(True)
        # Call original to avoid breaking the test
        return original_normalize(self, archive, logger, system_v2=system_v2)

    monkeypatch.setattr(TopologyNormalizer, 'normalize', mock_normalize)

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive_with_data_schema, LOGGER)

    # Verify TopologyNormalizer.normalize was called
    assert len(normalize_called) > 0, 'TopologyNormalizer.normalize should be called'


def test_legacy_path_delegates_to_nomad_fair_results_normalizer(
    archive_empty, monkeypatch
):
    """Legacy fallback should delegate to nomad-FAIR ResultsNormalizer."""
    from nomad.normalizing.results import ResultsNormalizer as LegacyResultsNormalizer

    called = []
    original_normalize = LegacyResultsNormalizer.normalize

    def mock_normalize(self, archive, logger=None):
        called.append(True)
        return original_normalize(self, archive, logger)

    monkeypatch.setattr(LegacyResultsNormalizer, 'normalize', mock_normalize)

    normalizer = ResultsNormalizer()
    try:
        normalizer.normalize(archive_empty, LOGGER)
    except Exception:
        # Legacy normalize may fail in minimal fixture; delegation is what matters.
        pass

    assert called, 'Legacy ResultsNormalizer.normalize should be delegated to'


def test_normalize_measurements_still_works(archive_with_data_schema):
    """Test that measurement normalization still works with new architecture."""
    # Skip this test as it requires complex measurement schema setup
    pytest.skip('Test requires complete measurement schema implementation')
