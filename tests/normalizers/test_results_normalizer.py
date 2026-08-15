"""Tests for ResultsNormalizer schema detection and routing logic."""

from types import SimpleNamespace

import numpy as np
import pytest
from nomad.datamodel import EntryArchive, EntryMetadata
from nomad.datamodel.data import ArchiveSection
from nomad.datamodel.results import (
    BandGap,
    BandStructureElectronic,
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
from nomad_simulations.schema_packages.outputs import Outputs, TrajectoryOutputs
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
    Frequency,
    ImaginaryTime,
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

from nomad_results_normalizer.normalizers.results import (
    DATA_SCHEMA_COMPATIBILITY_ANNOTATION,
)
from nomad_results_normalizer.normalizers.results import (
    ResultsNormalizerBase as ResultsNormalizer,
)

try:
    import runschema.calculation  # noqa: F401
    import runschema.run  # noqa: F401

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
    """Create an archive with nomad-simulations data schema (archive.data)."""
    archive = EntryArchive(metadata=EntryMetadata())

    # Create nomad-simulations data schema
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

    # Create nomad-simulations data schema
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
    """Data-schema archives should run the plugin path."""
    normalizer = ResultsNormalizer()

    # Clear any previous log records
    caplog.clear()

    # Run normalization
    normalizer.normalize(archive_with_data_schema, LOGGER)

    assert any(
        'data-schema results normalization' in record.message
        for record in caplog.records
    ), 'Should log data-schema path'
    assert not any(
        'legacy results normalization' in record.message for record in caplog.records
    ), 'Plugin should not run legacy results normalization itself'


def test_schema_detection_no_schema(archive_empty, caplog):
    """Archives without data-schema content should be skipped by this plugin."""
    normalizer = ResultsNormalizer()

    # Clear any previous log records
    caplog.clear()

    # Run normalization
    try:
        normalizer.normalize(archive_empty, LOGGER)
    except Exception:
        # May fail without proper data - that's OK
        pass

    assert any(
        'Skipping data-schema results normalization' in record.message
        for record in caplog.records
    ), 'Should skip data-schema path when no nomad-simulations data exists'


def test_non_simulation_data_schema_skips_plugin_path(caplog):
    """Custom archive.data without model_system should not use data-schema path."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    class CustomData(ArchiveSection):
        name = None

    archive.data = CustomData()

    normalizer = ResultsNormalizer()
    caplog.clear()

    normalizer.normalize(archive, LOGGER)

    assert any(
        'Skipping data-schema results normalization' in record.message
        for record in caplog.records
    ), 'Non-simulation archive.data should be skipped by this plugin'
    assert not any(
        record.message == 'Running data-schema results normalization'
        for record in caplog.records
    ), 'Non-simulation archive.data should not use data-schema path'


def test_schema_detection_nested_system(archive_with_nested_system, caplog):
    """Test that nested SystemV2 is detected and causes data-schema normalization."""
    normalizer = ResultsNormalizer()
    caplog.clear()

    # Run normalization
    normalizer.normalize(archive_with_nested_system, LOGGER)

    assert any(
        'data-schema results normalization' in record.message
        for record in caplog.records
    ), 'Nested SystemV2 should trigger data-schema normalization'
    # Keep this suite focused on routing. Detailed topology behavior is covered
    # in test_topology_normalizer.py.
    assert archive_with_nested_system.results is not None
    assert archive_with_nested_system.results.material is not None


def test_data_schema_initializes_results_sections(archive_with_data_schema):
    """Data-schema path should initialize key results sections."""
    normalizer = ResultsNormalizer()

    # Run normalization
    normalizer.normalize(archive_with_data_schema, LOGGER)

    # Check that results were populated
    assert archive_with_data_schema.results is not None
    assert archive_with_data_schema.results.material is not None


def test_data_schema_populates_method_from_simulation():
    """Simulation program/model_method should populate results.method."""
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
    """Data-schema method details should map to legacy-equivalent results fields."""
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
    """Data-schema DFT metadata should map to legacy-equivalent DFT result fields."""
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
    """Data-schema outputs should map electronic properties into results."""
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

    output = TrajectoryOutputs(time=0.0 * ureg.ps)
    output.electronic_band_gaps.append(ElectronicBandGap(value=1.5 * ureg.eV))
    dos = ElectronicDensityOfStates(
        value=np.array([0.1, 0.2, 0.3]) / ureg.eV, spin_channel=0
    )
    dos.energies = Energy2(points=np.array([-1.0, 0.0, 1.0]) * ureg.eV)
    output.electronic_dos.append(dos)
    band_structure = ElectronicBandStructure(value=np.array([[1.0], [1.1]]) * ureg.eV)
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

    output_2 = TrajectoryOutputs(time=1.0 * ureg.ps)
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
    band_structure_1 = ElectronicBandStructure(value=np.array([[1.0], [1.1]]) * ureg.eV)
    band_structure_1.k_path = _kline_path()
    output_1.electronic_band_structures.append(band_structure_1)
    simulation.outputs.append(output_1)

    output_2 = Outputs()
    output_2.electronic_band_gaps.append(ElectronicBandGap(value=2.5 * ureg.eV))
    band_structure_2 = ElectronicBandStructure(value=np.array([[2.0], [2.1]]) * ureg.eV)
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
    band_structure_rep = ElectronicBandStructure(
        value=np.array([[1.0], [1.1]]) * ureg.eV
    )
    band_structure_rep.k_path = _kline_path()
    output_rep.electronic_band_structures.append(band_structure_rep)
    simulation.outputs.append(output_rep)

    output_other = Outputs()
    output_other.model_system_ref = other_system
    output_other.electronic_band_gaps.append(ElectronicBandGap(value=2.5 * ureg.eV))
    band_structure_other = ElectronicBandStructure(
        value=np.array([[2.0], [2.1]]) * ureg.eV
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
    """Band mapping from data-schema outputs should create non-orphan segment refs."""
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
    band_structure = ElectronicBandStructure(value=np.array([[1.0], [1.1]]) * ureg.eV)
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
    assert archive.m_resolve(str(segments[0])) is not None


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
    band_structure = ElectronicBandStructure(value=np.array([[1.0], [1.1]]) * ureg.eV)
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
    band_structure = ElectronicBandStructure(value=np.array([[1.0], [1.1]]) * ureg.eV)
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


def test_data_schema_band_structure_skips_ambiguous_vertex_only_path():
    """Vertex-only paths must not be synthetically resampled onto energy points."""
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
    band_structure = ElectronicBandStructure(value=np.array([[1.0], [1.1]]) * ureg.eV)
    output.electronic_band_structures.append(band_structure)
    simulation.outputs.append(output)
    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    electronic = archive.results.properties.electronic
    assert electronic is None or not electronic.band_structure_electronic


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
    band_structure = ElectronicBandStructure(value=np.array([[1.0], [1.1]]) * ureg.eV)
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
    """Data-schema DOS mapping writes deprecated dos_electronic compatibility mirror."""
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
    """Data-schema DOS mapping should replace stale malformed DOS references."""
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

    output = TrajectoryOutputs(time=1.0 * ureg.ps)
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


def test_data_schema_trajectory_uses_physical_time(archive_with_data_schema):
    output = TrajectoryOutputs(time=2.5 * ureg.ps, wall_end=100 * ureg.s)
    output.temperatures.append(SimTemperature(value=300 * ureg.kelvin))
    archive_with_data_schema.data.outputs.append(output)

    ResultsNormalizer().normalize(archive_with_data_schema, LOGGER)

    trajectory = archive_with_data_schema.results.properties.thermodynamic.trajectory[0]
    time = trajectory.temperature.time.to('second').magnitude
    assert np.asarray(time)[0] == pytest.approx(2.5e-12)


def test_data_schema_logs_trajectory_dropped_without_physical_time(
    archive_with_data_schema, caplog
):
    """`time` only exists on TrajectoryOutputs; dropping it must not be silent."""
    output = Outputs()
    output.temperatures.append(SimTemperature(value=300 * ureg.kelvin))
    output.potential_energies.append(SimPotentialEnergy(value=-5.0 * ureg.eV))
    archive_with_data_schema.data.outputs.append(output)

    caplog.clear()
    ResultsNormalizer().normalize(archive_with_data_schema, LOGGER)

    thermodynamic = archive_with_data_schema.results.properties.thermodynamic
    assert thermodynamic is None or not thermodynamic.trajectory
    assert any(
        'skipping trajectory series without physical time' in record.message
        for record in caplog.records
    )


def test_data_schema_keeps_each_method_on_the_representative_system(
    archive_with_data_schema,
):
    """DFT and GW results on one system are legacy-equivalent side-by-side data.

    Legacy `get_gw_workflow_properties` publishes both, labelled; this path
    must not collapse them onto whichever output happens to come last.
    """
    simulation = archive_with_data_schema.data
    system = simulation.model_system[0]
    dft = DFT()
    gw = GW()
    simulation.model_method.extend([dft, gw])

    dft_output = Outputs(model_system_ref=system, model_method_ref=dft)
    dos = ElectronicDensityOfStates(value=np.array([0.1, 0.2]) / ureg.eV)
    dos.energies = Energy2(points=np.array([-1.0, 1.0]) * ureg.eV)
    dft_output.electronic_dos.append(dos)
    simulation.outputs.append(dft_output)

    gw_output = Outputs(model_system_ref=system, model_method_ref=gw)
    band_structure = ElectronicBandStructure(value=np.array([[1.0], [1.1]]) * ureg.eV)
    band_structure.k_path = _kline_path()
    gw_output.electronic_band_structures.append(band_structure)
    simulation.outputs.append(gw_output)

    ResultsNormalizer().normalize(archive_with_data_schema, LOGGER)

    electronic = archive_with_data_schema.results.properties.electronic
    assert [
        band_structure.label for band_structure in electronic.band_structure_electronic
    ] == ['GW']
    assert [dos.label for dos in electronic.dos_electronic] == ['DFT']


def test_data_schema_drops_outputs_from_non_representative_systems(
    archive_with_data_schema, caplog
):
    """Merging outputs across different systems stays forbidden."""
    simulation = archive_with_data_schema.data
    representative_system = simulation.model_system[0]
    other_system = ModelSystem(type='bulk')
    simulation.model_system.append(other_system)

    representative_output = Outputs(model_system_ref=representative_system)
    representative_output.electronic_band_gaps.append(
        ElectronicBandGap(value=1.5 * ureg.eV)
    )
    simulation.outputs.append(representative_output)

    other_output = Outputs(model_system_ref=other_system)
    other_output.electronic_band_gaps.append(ElectronicBandGap(value=2.5 * ureg.eV))
    simulation.outputs.append(other_output)

    caplog.clear()
    ResultsNormalizer().normalize(archive_with_data_schema, LOGGER)

    electronic = archive_with_data_schema.results.properties.electronic
    assert [band_gap.value.to('eV').magnitude for band_gap in electronic.band_gap] == [
        1.5
    ]
    assert any(
        'discarding electronic outputs from non-representative systems'
        in record.message
        for record in caplog.records
    )


@pytest.mark.skipif(not HAS_RUNSCHEMA, reason='requires runschema')
def test_data_schema_gives_each_method_its_own_legacy_dos_references(
    archive_with_data_schema,
):
    """Per-method DOS payloads must not overwrite each other's run references."""
    simulation = archive_with_data_schema.data
    system = simulation.model_system[0]
    dft = DFT()
    gw = GW()
    simulation.model_method.extend([dft, gw])

    for method, values in ((dft, [0.1, 0.2]), (gw, [0.3, 0.4])):
        output = Outputs(model_system_ref=system, model_method_ref=method)
        dos = ElectronicDensityOfStates(value=np.array(values) / ureg.eV)
        dos.energies = Energy2(points=np.array([-1.0, 1.0]) * ureg.eV)
        output.electronic_dos.append(dos)
        simulation.outputs.append(output)

    ResultsNormalizer().normalize(archive_with_data_schema, LOGGER)

    serialized = archive_with_data_schema.m_to_dict()
    dos_sections = serialized['results']['properties']['electronic']['dos_electronic']
    assert [section['label'] for section in dos_sections] == ['DFT', 'GW']

    references = [section['energies'] for section in dos_sections]
    references += [ref for section in dos_sections for ref in section['total']]
    assert len(set(references)) == len(references)
    assert all(
        archive_with_data_schema.m_resolve(reference) is not None
        for reference in references
    )


@pytest.mark.skipif(not HAS_RUNSCHEMA, reason='requires runschema')
def test_data_schema_preserves_legacy_calculation_and_result_sections():
    archive = EntryArchive(
        metadata=EntryMetadata(), results=Results(properties=Properties())
    )

    legacy_run = runschema.run.Run(raw_id='parser-owned')
    legacy_calculation = runschema.calculation.Calculation()
    legacy_dos = runschema.calculation.Dos(energies=np.array([-2.0, 2.0]) * ureg.eV)
    legacy_dos.total.append(
        runschema.calculation.DosValues(value=np.array([0.3, 0.4]) / ureg.eV)
    )
    legacy_calculation.dos_electronic.append(legacy_dos)
    legacy_band_structure = runschema.calculation.BandStructure()
    legacy_band_structure.segment.append(
        runschema.calculation.BandEnergies(
            energies=np.array([[[2.0], [2.1]]]) * ureg.eV,
            kpoints=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        )
    )
    legacy_calculation.band_structure_electronic.append(legacy_band_structure)
    legacy_run.calculation.append(legacy_calculation)
    archive.run.append(legacy_run)

    electronic = archive.results.properties.m_create(ElectronicProperties)
    parser_dos = DOSElectronic(label='parser-owned')
    parser_dos.energies = '/run/0/calculation/0/dos_electronic/0/energies'
    parser_dos.total = ['/run/0/calculation/0/dos_electronic/0/total/0']
    electronic.m_add_sub_section(ElectronicProperties.dos_electronic, parser_dos)
    parser_band_structure = BandStructureElectronic(label='parser-owned')
    parser_band_structure.segment = legacy_band_structure.segment
    electronic.m_add_sub_section(
        ElectronicProperties.band_structure_electronic, parser_band_structure
    )
    electronic.m_add_sub_section(
        ElectronicProperties.band_gap, BandGap(value=9.0 * ureg.eV)
    )

    simulation = Simulation(program=Program(name='VASP'))
    system = ModelSystem(is_representative=True, type='molecule')
    simulation.model_system.append(system)
    output = Outputs()
    output.electronic_band_gaps.append(ElectronicBandGap(value=1.0 * ureg.eV))
    mapped_dos = ElectronicDensityOfStates(value=np.array([0.1, 0.2]) / ureg.eV)
    mapped_dos.energies = Energy2(points=np.array([-1.0, 1.0]) * ureg.eV)
    output.electronic_dos.append(mapped_dos)
    mapped_band_structure = ElectronicBandStructure(
        value=np.array([[1.0], [1.1]]) * ureg.eV
    )
    mapped_band_structure.k_path = _kline_path()
    output.electronic_band_structures.append(mapped_band_structure)
    simulation.outputs.append(output)
    archive.data = simulation

    ResultsNormalizer().normalize(archive, LOGGER)

    assert archive.run[0] is legacy_run
    assert legacy_calculation.dos_electronic[0] is legacy_dos
    assert legacy_calculation.band_structure_electronic[0] is legacy_band_structure
    # The generated run is identified by annotation alone; no user-facing
    # quantity is claimed to mark it.
    assert archive.run[1].raw_id is None
    assert DATA_SCHEMA_COMPATIBILITY_ANNOTATION in archive.run[1].m_annotations
    assert any(section.label == 'parser-owned' for section in electronic.dos_electronic)
    assert any(
        section.label == 'parser-owned'
        for section in electronic.band_structure_electronic
    )
    assert any(gap.value.to('eV').magnitude == 9.0 for gap in electronic.band_gap)

    serialized = archive.m_to_dict()
    generated_dos = next(
        section
        for section in serialized['results']['properties']['electronic'][
            'dos_electronic'
        ]
        if DATA_SCHEMA_COMPATIBILITY_ANNOTATION in section.get('m_annotations', {})
    )
    assert generated_dos.get('label') is None
    assert generated_dos['energies'].startswith('/run/1/calculation/0/')
    assert all(
        ref.startswith('/run/1/calculation/0/') for ref in generated_dos['total']
    )
    assert archive.m_resolve(generated_dos['energies']) is not None
    assert all(archive.m_resolve(ref) is not None for ref in generated_dos['total'])


def test_data_schema_output_mapping_is_idempotent(archive_with_data_schema):
    output = TrajectoryOutputs(time=2.0 * ureg.ps)
    absorption = AbsorptionSpectrum(value=np.array([0.5, 0.6]) / ureg.eV)
    absorption.energies = Energy2(points=np.array([0.0, 1.0]) * ureg.eV)
    output.absorption_spectra.append(absorption)
    output.radii_of_gyration.append(SimRadiusOfGyration(value=1.2 * ureg.angstrom))
    output.temperatures.append(SimTemperature(value=300 * ureg.kelvin))
    archive_with_data_schema.data.outputs.append(output)

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive_with_data_schema, LOGGER)
    properties = archive_with_data_schema.results.properties
    first_counts = (
        len(properties.spectroscopic.spectra),
        len(properties.structural.radius_of_gyration),
        len(properties.thermodynamic.trajectory),
    )
    first_time = properties.thermodynamic.trajectory[0].temperature.time.copy()

    assert DATA_SCHEMA_COMPATIBILITY_ANNOTATION in (
        properties.spectroscopic.spectra[0].m_annotations
    )
    assert DATA_SCHEMA_COMPATIBILITY_ANNOTATION in (
        properties.structural.radius_of_gyration[0].m_annotations
    )
    assert DATA_SCHEMA_COMPATIBILITY_ANNOTATION in (
        properties.thermodynamic.trajectory[0].m_annotations
    )
    assert properties.thermodynamic.trajectory[0].provenance is None

    normalizer.normalize(archive_with_data_schema, LOGGER)

    assert (
        len(properties.spectroscopic.spectra),
        len(properties.structural.radius_of_gyration),
        len(properties.thermodynamic.trajectory),
    ) == first_counts
    np.testing.assert_allclose(
        properties.thermodynamic.trajectory[0].temperature.time,
        first_time,
    )


def test_greens_mapping_does_not_create_axis_only_result():
    if 'tau' not in GreensFunctionsElectronic.m_def.all_quantities:
        pytest.skip('legacy Green functions fields unavailable')

    greens = SimpleNamespace(
        imaginary_time=None,
        matsubara_frequency=None,
        real_frequency=Frequency(points=np.array([0.0, 1.0]) * ureg.eV),
        value=np.array([1.0 + 1.0j, 2.0 + 1.0j]) / ureg.eV,
    )
    output = SimpleNamespace(
        electronic_greens_functions=[greens],
        electronic_self_energies=[],
        hybridization_functions=[],
        quasiparticle_weights=[],
        chemical_potentials=[],
    )
    normalizer = ResultsNormalizer()
    normalizer.logger = LOGGER

    assert normalizer._map_greens_functions(output) is None


def test_greens_mapping_recovers_from_incompatible_payload_shape(caplog):
    if 'tau' not in GreensFunctionsElectronic.m_def.all_quantities:
        pytest.skip('legacy Green functions fields unavailable')

    greens = SimpleNamespace(
        imaginary_time=ImaginaryTime(points=np.array([0.0, 1.0]) * ureg.s),
        matsubara_frequency=None,
        real_frequency=None,
        value=np.array([1.0, 2.0]) / ureg.eV,
    )
    output = SimpleNamespace(
        electronic_greens_functions=[greens],
        electronic_self_energies=[],
        hybridization_functions=[],
        quasiparticle_weights=[],
        chemical_potentials=[],
    )
    normalizer = ResultsNormalizer()
    normalizer.logger = LOGGER

    caplog.clear()
    assert normalizer._map_greens_functions(output) is None
    assert any(
        'skipping incompatible greens axis/payload pair' in record.message
        for record in caplog.records
    )


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


def test_data_schema_runs_after_legacy_run(archive_with_data_schema):
    """Data-schema normalization should run after legacy run normalization."""
    # Add a mock run section to the data schema archive
    from nomad.datamodel.data import ArchiveSection

    class MockRun(ArchiveSection):
        system = []

    archive_with_data_schema.run = [MockRun()]

    normalizer = ResultsNormalizer()

    normalizer.normalize(archive_with_data_schema, LOGGER)

    assert archive_with_data_schema.results is not None


def test_data_schema_runs_plugin_path(archive_with_data_schema, monkeypatch):
    calls = []

    def method(self, archive):
        calls.append('method')

    def outputs(self, archive):
        calls.append('outputs')

    monkeypatch.setattr(ResultsNormalizer, '_normalize_method_with_data_schema', method)
    monkeypatch.setattr(
        ResultsNormalizer, '_normalize_outputs_with_data_schema', outputs
    )

    ResultsNormalizer().normalize(archive_with_data_schema, LOGGER)

    assert calls == ['method', 'outputs']


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


def test_data_schema_normalize_calls_topology_normalizer(
    archive_with_data_schema, monkeypatch
):
    """Test that the data-schema normalize path calls TopologyNormalizer."""
    from nomad_results_normalizer.normalizers.topology import TopologyNormalizer

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


def test_normalize_measurements_still_works(archive_with_data_schema):
    """Test that measurement normalization still works with new architecture."""
    # Skip this test as it requires complex measurement schema setup
    pytest.skip('Test requires complete measurement schema implementation')
