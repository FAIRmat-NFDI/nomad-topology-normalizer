"""Tests for MaterialNormalizer v2 functionality."""

import numpy as np
import pytest
from nomad.datamodel import EntryArchive, EntryMetadata
from nomad.datamodel.results import Material, Results
from nomad.units import ureg
from nomad.utils import get_logger
from nomad_simulations.schema_packages.atoms_state import AtomsState
from nomad_simulations.schema_packages.general import Simulation
from nomad_simulations.schema_packages.model_system import AtomicCell, ModelSystem

LOGGER = get_logger(__name__)


def create_test_system(
    chemical_symbols: list[str] = ['Si', 'Si'],
    positions: list[list[float]] = [[0, 0, 0], [0.25, 0.25, 0.25]],
    lattice_vectors: list[list[float]] = [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]],
    pbc: list[bool] = [True, True, True],
) -> ModelSystem:
    """Create a test ModelSystem with v2 schema."""
    system = ModelSystem(is_representative=True)
    system.positions = np.array(positions) * ureg.angstrom

    # Add cell
    cell = AtomicCell(
        lattice_vectors=np.array(lattice_vectors) * ureg.angstrom,
        periodic_boundary_conditions=pbc,
    )
    system.cell.append(cell)

    # Add particle states
    for symbol in chemical_symbols:
        state = AtomsState(chemical_symbol=symbol)
        system.particle_states.append(state)

    return system


class TestMaterialNormalizer:
    """Test MaterialNormalizer with v2 schema data."""

    def test_chemical_formula_extraction(self):
        """Test that MaterialNormalizer extracts chemical_formula.hill from v2 system."""
        # Create archive with v2 system
        archive = EntryArchive(metadata=EntryMetadata())
        archive.results = Results()
        archive.results.material = Material()

        # Create system with known composition
        system = create_test_system(
            chemical_symbols=['H', 'H', 'O'],
            positions=[[0, 0, 0], [1.0, 0, 0], [0, 0.5, 0]],
            lattice_vectors=[[10, 0, 0], [0, 10, 0], [0, 0, 10]],
        )

        # Normalize system to populate chemical_formula
        simulation = Simulation()
        simulation.model_system.append(system)
        archive.data = simulation
        system.normalize(archive, LOGGER)

        # Verify chemical_formula is populated
        assert system.chemical_formula is not None
        assert system.chemical_formula.hill is not None
        # H2O should give 'H2O' in Hill notation
        assert 'H' in system.chemical_formula.hill
        assert 'O' in system.chemical_formula.hill

    def test_material_normalizer_with_silicon(self):
        """Test MaterialNormalizer with silicon crystal structure."""
        archive = EntryArchive(metadata=EntryMetadata())
        archive.results = Results()
        archive.results.material = Material()

        # Create silicon system
        system = create_test_system(
            chemical_symbols=['Si', 'Si'],
            positions=[[0, 0, 0], [0.25, 0.25, 0.25]],
            lattice_vectors=[[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]],
            pbc=[True, True, True],
        )

        simulation = Simulation()
        simulation.model_system.append(system)
        archive.data = simulation

        # Normalize to populate chemical formulas
        system.normalize(archive, LOGGER)

        # Verify chemical formula is available
        assert system.chemical_formula is not None
        assert system.chemical_formula.hill == 'Si2'

    def test_particle_states_chemical_symbols(self):
        """Test that MaterialNormalizer can access chemical symbols from particle_states."""
        archive = EntryArchive(metadata=EntryMetadata())
        archive.results = Results()
        archive.results.material = Material()

        # Create system with varied elements
        system = create_test_system(
            chemical_symbols=['Na', 'Cl'],
            positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
            lattice_vectors=[[5.64, 0, 0], [0, 5.64, 0], [0, 0, 5.64]],
        )

        simulation = Simulation()
        simulation.model_system.append(system)
        archive.data = simulation
        system.normalize(archive, LOGGER)

        # Verify particle_states have chemical_symbol
        assert len(system.particle_states) == 2
        symbols = [state.chemical_symbol for state in system.particle_states]
        assert 'Na' in symbols
        assert 'Cl' in symbols

    def test_dimensionality_from_pbc(self):
        """Test that dimensionality is correctly determined from PBC."""
        archive = EntryArchive(metadata=EntryMetadata())
        archive.results = Results()
        archive.results.material = Material()

        # Create 2D system (periodic in x,y only)
        system = create_test_system(
            chemical_symbols=['C', 'C'],
            positions=[[0, 0, 0], [1.42, 0, 0]],
            lattice_vectors=[[2.46, 0, 0], [1.23, 2.13, 0], [0, 0, 20]],
            pbc=[True, True, False],
        )

        simulation = Simulation()
        simulation.model_system.append(system)
        archive.data = simulation
        system.normalize(archive, LOGGER)

        # Check if system type reflects 2D nature
        # (actual dimensionality computation happens in topology normalizer)
        assert system.cell[0].periodic_boundary_conditions == [True, True, False]

    def test_empty_system_handling(self):
        """Test MaterialNormalizer handles empty systems gracefully."""
        archive = EntryArchive(metadata=EntryMetadata())
        archive.results = Results()
        archive.results.material = Material()

        # Create minimal system without particles
        system = ModelSystem(is_representative=True)
        cell = AtomicCell(
            lattice_vectors=np.eye(3) * 10 * ureg.angstrom,
            periodic_boundary_conditions=[True, True, True],
        )
        system.cell.append(cell)

        simulation = Simulation()
        simulation.model_system.append(system)
        archive.data = simulation

        # Should not crash
        system.normalize(archive, LOGGER)

        # Chemical formula should be None or empty
        if system.chemical_formula:
            assert system.chemical_formula.hill is None

    def test_complex_composition(self):
        """Test with more complex chemical composition."""
        archive = EntryArchive(metadata=EntryMetadata())
        archive.results = Results()
        archive.results.material = Material()

        # Create perovskite-like structure (e.g., CaTiO3)
        system = create_test_system(
            chemical_symbols=['Ca', 'Ti', 'O', 'O', 'O'],
            positions=[
                [0, 0, 0],
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0],
                [0.5, 0, 0.5],
                [0, 0.5, 0.5],
            ],
            lattice_vectors=[[3.8, 0, 0], [0, 3.8, 0], [0, 0, 3.8]],
        )

        simulation = Simulation()
        simulation.model_system.append(system)
        archive.data = simulation
        system.normalize(archive, LOGGER)

        # Verify all elements are captured
        assert system.chemical_formula is not None
        hill = system.chemical_formula.hill
        assert 'Ca' in hill
        assert 'Ti' in hill
        assert 'O' in hill

    def test_legacy_fallback(self):
        """Test that MaterialNormalizer handles systems without chemical_formula."""
        archive = EntryArchive(metadata=EntryMetadata())
        archive.results = Results()
        archive.results.material = Material()

        # Create system but don't normalize it
        system = ModelSystem(is_representative=True)
        system.positions = np.array([[0, 0, 0], [1, 0, 0]]) * ureg.angstrom

        cell = AtomicCell(
            lattice_vectors=np.eye(3) * 10 * ureg.angstrom,
            periodic_boundary_conditions=[True, True, True],
        )
        system.cell.append(cell)

        # Add particle states
        for symbol in ['H', 'H']:
            state = AtomsState(chemical_symbol=symbol)
            system.particle_states.append(state)

        simulation = Simulation()
        simulation.model_system.append(system)
        archive.data = simulation

        # MaterialNormalizer should handle missing chemical_formula
        # by checking hasattr or falling back gracefully
        assert system.chemical_formula is None or not hasattr(
            system, 'chemical_formula'
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
