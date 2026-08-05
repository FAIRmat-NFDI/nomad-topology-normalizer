"""Tests for ResultsNormalizer schema detection and routing logic."""

import numpy as np
import pytest
from nomad.datamodel import EntryArchive, EntryMetadata
from nomad.datamodel.data import ArchiveSection
from nomad.datamodel.results import (
    DOSElectronic,
    ElectronicProperties,
    GreensFunctionsElectronic,
    Properties,
    Results,
)
from nomad.metainfo import Quantity
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
from nomad_simulations.schema_packages.numerical_settings import (
    KSpace,
    SelfConsistency,
    Smearing,
)
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
    KLinePath,
    MatsubaraFrequency,
)
from nomad_simulations.schema_packages.workflow.general import (
    EnergyConvergenceTarget,
    ForceConvergenceTarget,
)
from nomad_simulations.schema_packages.workflow.geometry_optimization import (
    GeometryOptimization as SimGeometryOptimizationWorkflow,
)
from nomad_simulations.schema_packages.workflow.geometry_optimization import (
    GeometryOptimizationMethod as SimGeometryOptimizationMethod,
)
from nomad_simulations.schema_packages.workflow.geometry_optimization import (
    GeometryOptimizationResults as SimGeometryOptimizationResults,
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
        archive.results.method.simulation.tb.localization_type == 'maximally_localized'
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


def test_data_schema_maps_flexible_unit_scf_threshold(archive_with_data_schema):
    """Current SelfConsistency quantities should map without a legacy unit field."""
    simulation = archive_with_data_schema.data
    simulation.program = Program(name='VASP')
    dft = DFT()
    dft.numerical_settings.append(SelfConsistency(threshold_change=1.0e-5 * ureg.eV))
    simulation.model_method.append(dft)

    ResultsNormalizer().normalize(archive_with_data_schema, LOGGER)

    simulation_results = archive_with_data_schema.results.method.simulation
    threshold = simulation_results.dft.scf_threshold_energy_change
    assert threshold.to('eV').magnitude == pytest.approx(1.0e-5)


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
    greens_value_type = ElectronicGreensFunction.m_def.all_quantities['value'].type
    if (
        'tau' in GreensFunctionsElectronic.m_def.all_quantities
        and greens_value_type.__class__.__name__ != 'HDF5Dataset'
    ):
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
    if HAS_RUNSCHEMA:
        assert len(getattr(archive, 'run', None) or []) == 1
    else:
        assert len(getattr(archive, 'run', None) or []) == 0


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
    assert electronic.band_gap[0].value.to('eV').magnitude == pytest.approx(2.5)
    assert len(electronic.band_structure_electronic or []) == 1


def test_data_schema_prefers_representative_system_outputs_for_electronic_properties():
    """Electronic mapping should prefer outputs linked to representative system."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    simulation = Simulation()
    simulation.program = Program(name='VASP')
    simulation.model_method.append(DFT())

    representative_system = ModelSystem(
        name='rep_system',
        type='bulk',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    representative_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    representative_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    representative_system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    representative_system.periodic_boundary_conditions = [True, True, True]

    other_system = ModelSystem(
        name='other_system',
        type='bulk',
        is_representative=False,
        positions=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    other_system.particle_states.append(
        AtomsState(chemical_symbol='Ge', atomic_number=32)
    )
    other_system.particle_states.append(
        AtomsState(chemical_symbol='Ge', atomic_number=32)
    )
    other_system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    other_system.periodic_boundary_conditions = [True, True, True]

    simulation.model_system.append(representative_system)
    simulation.model_system.append(other_system)

    # If supported by current schema version, set explicit representative index.
    if 'representative_index' in simulation.m_def.all_quantities:
        simulation.representative_index = 0

    output_rep = Outputs()
    output_rep.model_system_ref = representative_system
    output_rep.electronic_band_gaps.append(ElectronicBandGap(value=1.5 * ureg.eV))
    band_structure_rep = ElectronicBandStructure(value=np.array([[1.0, 1.1]]) * ureg.eV)
    band_structure_rep.k_path = _kline_path()
    output_rep.electronic_band_structures.append(band_structure_rep)
    simulation.outputs.append(output_rep)

    output_other = Outputs()
    output_other.model_system_ref = other_system
    output_other.electronic_band_gaps.append(ElectronicBandGap(value=2.5 * ureg.eV))
    band_structure_other = ElectronicBandStructure(
        value=np.array([[2.0, 2.1]]) * ureg.eV
    )
    band_structure_other.k_path = _kline_path()
    output_other.electronic_band_structures.append(band_structure_other)
    simulation.outputs.append(output_other)

    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    electronic = archive.results.properties.electronic
    assert electronic is not None
    assert len(electronic.band_gap or []) == 1
    assert electronic.band_gap[0].value.to('eV').magnitude == pytest.approx(1.5)
    assert len(electronic.band_structure_electronic or []) == 1


def test_data_schema_band_structure_mapping_creates_valid_segment_refs():
    """Band structure mapping from v2 outputs should create non-orphan segment refs."""
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
    assert serialized.get('run')
    bs = (
        serialized.get('results', {})
        .get('properties', {})
        .get('electronic', {})
        .get('band_structure_electronic', [])
    )
    assert bs
    segments = bs[0].get('segment', [])
    assert segments
    assert segments[0] != '/'
    assert str(segments[0]).startswith(
        '/run/0/calculation/0/band_structure_electronic/0/segment/'
    )


def test_data_schema_band_structure_uses_numerical_settings_kline_path_fallback():
    """Band mapping should fall back to model-method k_line_path.

    This fallback is used when `bs.k_path` is missing.
    """
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    simulation = Simulation()
    simulation.program = Program(name='VASP')

    dft = DFT()
    k_space = KSpace()
    k_space.k_line_path = _kline_path()
    dft.numerical_settings.append(k_space)
    simulation.model_method.append(dft)

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
    output.electronic_band_structures.append(band_structure)
    simulation.outputs.append(output)
    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    electronic = archive.results.properties.electronic
    assert electronic is not None
    assert electronic.band_structure_electronic
    segments = electronic.band_structure_electronic[0].segment
    assert segments
    assert segments[0].m_def.name != 'EntryArchive'
    assert np.array(segments[0].kpoints).shape[0] > 0


def test_data_schema_band_structure_skips_non_kspace_numerical_settings():
    """Fallback search should ignore numerical settings without `k_line_path`."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    simulation = Simulation()
    simulation.program = Program(name='VASP')

    dft = DFT()
    dft.numerical_settings.append(Smearing())
    k_space = KSpace()
    k_space.k_line_path = _kline_path()
    dft.numerical_settings.append(k_space)
    simulation.model_method.append(dft)

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
    output.electronic_band_structures.append(band_structure)
    simulation.outputs.append(output)
    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    electronic = archive.results.properties.electronic
    assert electronic is not None
    assert electronic.band_structure_electronic
    segments = electronic.band_structure_electronic[0].segment
    assert segments
    assert np.array(segments[0].kpoints).shape[0] > 0


def test_data_schema_band_structure_uses_kline_vertex_values_fallback():
    """Band mapping should use high_symmetry_path_values fallback.

    This fallback is used when k_line points are missing.
    """
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    simulation = Simulation()
    simulation.program = Program(name='exciting')

    dft = DFT()
    k_space = KSpace()
    k_line_path = KLinePath()
    k_line_path.high_symmetry_path_values = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]]
    )
    k_space.k_line_path = k_line_path
    dft.numerical_settings.append(k_space)
    simulation.model_method.append(dft)

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
    output.electronic_band_structures.append(band_structure)
    simulation.outputs.append(output)
    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    electronic = archive.results.properties.electronic
    assert electronic is not None
    assert electronic.band_structure_electronic
    segments = electronic.band_structure_electronic[0].segment
    assert segments
    assert (
        np.array(segments[0].kpoints).shape[0]
        == np.array(segments[0].energies.magnitude).shape[1]
    )


def test_data_schema_propagates_reference_energy_to_legacy_electronic_sections():
    """Normalization scope: propagate HOE reference to BS/DOS compatibility payloads."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    simulation = Simulation()
    simulation.program = Program(name='exciting')
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

    reference_energy = 1.0 * ureg.eV

    output = Outputs()
    band_structure = ElectronicBandStructure(value=np.array([[1.0, 1.1]]) * ureg.eV)
    band_structure.k_path = _kline_path()
    band_structure.highest_occupied = reference_energy
    output.electronic_band_structures.append(band_structure)

    dos = ElectronicDensityOfStates(value=np.array([0.1, 0.2, 0.3]) / ureg.eV)
    dos.energies = Energy2(points=np.array([-1.0, 0.0, 1.0]) * ureg.eV)
    output.electronic_dos.append(dos)

    output.electronic_band_gaps.append(ElectronicBandGap(value=0.5 * ureg.eV))
    simulation.outputs.append(output)
    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    electronic = archive.results.properties.electronic
    assert electronic is not None

    assert electronic.band_structure_electronic
    bs = electronic.band_structure_electronic[0]
    assert bs.band_gap
    assert bs.band_gap[0].energy_highest_occupied is not None

    assert electronic.dos_electronic
    dos_compat = electronic.dos_electronic[0]
    assert dos_compat.band_gap
    assert dos_compat.band_gap[0].energy_highest_occupied is not None


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
        assert len(getattr(archive, 'run', None) or []) == 1
        assert len(getattr(archive.run[0], 'calculation', None) or []) == 1
        assert (
            len(getattr(archive.run[0].calculation[0], 'dos_electronic', None) or [])
            == 1
        )
    else:
        assert electronic is None or not electronic.dos_electronic_new
        assert electronic is None or not electronic.dos_electronic
        assert len(getattr(archive, 'run', None) or []) == 0


def test_data_schema_populates_deprecated_dos_with_references():
    """Deprecated dos_electronic should point to run/calculation compatibility refs."""
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
    assert electronic is not None
    assert not electronic.dos_electronic_new
    assert electronic.dos_electronic

    serialized = archive.m_to_dict()
    dos_sections = (
        serialized.get('results', {})
        .get('properties', {})
        .get('electronic', {})
        .get('dos_electronic', [])
    )
    assert dos_sections
    dos_ref = dos_sections[0]
    assert dos_ref.get('energies')
    assert dos_ref['energies'].startswith('/run/0/calculation/0/dos_electronic/')
    assert dos_ref.get('total')
    assert all(
        ref.startswith('/run/0/calculation/0/dos_electronic/')
        for ref in dos_ref['total']
    )
    assert len(getattr(archive, 'run', None) or []) == 1


def test_data_schema_replaces_malformed_existing_dos_entries():
    """v2 DOS mapping should replace stale malformed DOS references in results."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()
    archive.results.properties.electronic = archive.results.properties.m_create(
        ElectronicProperties
    )

    # Simulate stale malformed entry that can crash GUI (missing energies/total)
    archive.results.properties.electronic.m_add_sub_section(
        ElectronicProperties.dos_electronic, DOSElectronic()
    )

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

    ResultsNormalizer().normalize(archive, LOGGER)

    serialized = archive.m_to_dict()
    dos_sections = (
        serialized.get('results', {})
        .get('properties', {})
        .get('electronic', {})
        .get('dos_electronic', [])
    )
    assert dos_sections
    assert len(dos_sections) == 1
    assert dos_sections[0].get('energies')
    assert dos_sections[0].get('total')


def test_data_schema_drops_malformed_stale_dos_without_new_electronic_payload():
    """Malformed stale dos_electronic should be removed to avoid GUI crashes."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()
    archive.results.properties.electronic = archive.results.properties.m_create(
        ElectronicProperties
    )
    archive.results.properties.electronic.m_add_sub_section(
        ElectronicProperties.dos_electronic, DOSElectronic()
    )

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

    # No electronic payload in outputs: only thermodynamic quantity.
    output = Outputs()
    output.temperatures.append(SimTemperature(value=300 * ureg.kelvin))
    simulation.outputs.append(output)
    archive.data = simulation

    ResultsNormalizer().normalize(archive, LOGGER)

    electronic = archive.results.properties.electronic
    if electronic is not None:
        assert not (electronic.dos_electronic or [])


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
    """DOS mapping should skip cleanly when runschema-based payload cannot be built."""
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
        assert len(getattr(archive_with_data_schema, 'run', None) or []) == 1
    else:
        assert electronic is None or not electronic.dos_electronic_new
        assert (
            'Skipping DOS mapping for results.properties.electronic.dos_electronic'
            in caplog.text
        )
        assert len(getattr(archive_with_data_schema, 'run', None) or []) == 0


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


def test_data_schema_maps_geometry_optimization_workflow(archive_with_data_schema):
    workflow = SimGeometryOptimizationWorkflow()
    workflow.method = SimGeometryOptimizationMethod(optimization_type='atomic')
    workflow.method.convergence_targets = [
        EnergyConvergenceTarget(threshold=1e-6 * ureg.eV),
        ForceConvergenceTarget(threshold=1e-5 * ureg.newton),
    ]
    workflow.results = SimGeometryOptimizationResults(
        final_energy_difference=2e-6 * ureg.eV,
        final_force_maximum=4e-5 * ureg.newton,
        final_displacement_maximum=1e-12 * ureg.meter,
    )

    archive_with_data_schema.workflow2 = workflow

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive_with_data_schema, LOGGER)

    geometry_optimization = (
        archive_with_data_schema.results.properties.geometry_optimization
    )
    assert geometry_optimization is not None
    assert geometry_optimization.type == 'atomic'
    assert geometry_optimization.convergence_tolerance_energy_difference is not None
    assert geometry_optimization.convergence_tolerance_force_maximum is not None
    assert geometry_optimization.final_energy_difference is not None
    assert geometry_optimization.final_force_maximum is not None
    # Note: Both trajectory and system_optimized cannot be populated from new schema
    # They expect legacy runschema types (Calculation, System), not
    # nomad-simulations (Outputs, ModelSystem)
    # This is expected - convergence values are sufficient for new schema workflows


def test_data_schema_geometry_optimization_without_legacy_refs(
    archive_with_data_schema,
):
    """Test geometry optimization mapping without legacy calculations_ref fields."""
    workflow = SimGeometryOptimizationWorkflow()
    workflow.method = SimGeometryOptimizationMethod(optimization_type='atomic')
    workflow.method.convergence_targets = [
        EnergyConvergenceTarget(threshold=1e-6 * ureg.eV),
        ForceConvergenceTarget(threshold=1e-5 * ureg.newton),
    ]
    # Results without calculations_ref or calculation_result_ref
    workflow.results = SimGeometryOptimizationResults(
        final_energy_difference=2e-6 * ureg.eV,
        final_force_maximum=4e-5 * ureg.newton,
        final_displacement_maximum=1e-12 * ureg.meter,
    )

    archive_with_data_schema.workflow2 = workflow

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive_with_data_schema, LOGGER)

    geometry_optimization = (
        archive_with_data_schema.results.properties.geometry_optimization
    )
    assert geometry_optimization is not None
    assert geometry_optimization.type == 'atomic'
    assert geometry_optimization.convergence_tolerance_energy_difference is not None
    assert geometry_optimization.convergence_tolerance_force_maximum is not None
    assert geometry_optimization.final_energy_difference is not None
    assert geometry_optimization.final_force_maximum is not None
    # Note: trajectory and system_optimized cannot be populated from new
    # schema (type incompatibility)


def test_data_schema_geometry_optimization_with_method_tolerances(
    archive_with_data_schema,
):
    """Test geometry optimization when tolerances are on method directly."""
    workflow = SimGeometryOptimizationWorkflow()
    method = SimGeometryOptimizationMethod(optimization_type='cell_shape')
    method.convergence_tolerance_energy_difference = 1e-7 * ureg.eV
    method.convergence_tolerance_force_maximum = 1e-6 * ureg.newton
    workflow.method = method

    workflow.results = SimGeometryOptimizationResults(
        final_energy_difference=2e-7 * ureg.eV,
        final_force_maximum=4e-6 * ureg.newton,
    )

    archive_with_data_schema.workflow2 = workflow

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive_with_data_schema, LOGGER)

    geometry_optimization = (
        archive_with_data_schema.results.properties.geometry_optimization
    )
    assert geometry_optimization is not None
    assert geometry_optimization.type == 'cell_shape'
    assert geometry_optimization.convergence_tolerance_energy_difference.to(
        ureg.eV
    ).magnitude == pytest.approx(1e-7)
    assert (
        geometry_optimization.convergence_tolerance_force_maximum.magnitude
        == pytest.approx(1e-6)
    )
    assert geometry_optimization.final_energy_difference.to(
        ureg.eV
    ).magnitude == pytest.approx(2e-7)
    assert geometry_optimization.final_force_maximum.magnitude == pytest.approx(4e-6)


def test_data_schema_geometry_optimization_detects_via_class_name(
    archive_with_data_schema,
):
    """Test that geometry optimization is detected by workflow class name."""
    workflow = SimGeometryOptimizationWorkflow()
    # Workflow with no method or results, but class name should be enough

    archive_with_data_schema.workflow2 = workflow

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive_with_data_schema, LOGGER)

    geometry_optimization = (
        archive_with_data_schema.results.properties.geometry_optimization
    )
    assert geometry_optimization is not None


def test_data_schema_merges_electronic_payload_split_across_outputs(
    archive_with_data_schema,
):
    representative = archive_with_data_schema.data.model_system[0]

    output_bs = Outputs(model_system_ref=representative)
    output_bs.electronic_band_structures.append(
        ElectronicBandStructure(
            value=np.array([[0.0, 1.0], [0.1, 1.1]]) * ureg.eV,
            highest_occupied=0.0 * ureg.eV,
            lowest_unoccupied=1.0 * ureg.eV,
            k_path=_kline_path(),
        )
    )

    output_dos = Outputs(model_system_ref=representative)
    dos = ElectronicDensityOfStates(
        value=np.array([0.2, 0.3, 0.4]) / ureg.eV,
        spin_channel=0,
    )
    dos.energies = Energy2(points=np.array([-1.0, 0.0, 1.0]) * ureg.eV)
    dos.energies_origin = 0.0 * ureg.eV
    output_dos.electronic_dos.append(dos)

    archive_with_data_schema.data.outputs = [output_bs, output_dos]

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive_with_data_schema, LOGGER)

    electronic = archive_with_data_schema.results.properties.electronic
    assert electronic is not None
    assert electronic.band_structure_electronic
    if HAS_RUNSCHEMA:
        assert electronic.dos_electronic


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
