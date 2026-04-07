"""
Topology Normalizer for v2 Data Schema
=======================================

This module provides topology normalization for entries using v2 data schema
(archive.data with basesections.v2.System). It is called by ResultsNormalizer
after schema version detection.

Topology Creation Strategy (Waterfall)
---------------------------------------

1. **topology_calculation()** - Extract from parser-defined sub_systems
   └─ Uses data.model_system[].sub_systems if available

2. **topology_matid()** - Algorithmic detection using MatID
   └─ Symmetry analysis and clustering for automatic topology detection

3. **topology_data()** - Fallback for SystemV2 entries
   └─ Direct conversion from SystemV2 structure

Note: Schema version detection and routing logic is in ResultsNormalizer.
See results.py for the complete normalization cascade architecture.
"""

from typing import (
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    # from nomad.datamodel.datamodel import (
    #     EntryArchive,
    # )
    from numpy.typing import NDArray

import json
import pathlib
from collections import defaultdict

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers as _atomic_numbers
from ase.data import chemical_symbols as _chemical_symbols
from matid.clustering import SBC, Cluster
from matid.symmetry.symmetryanalyzer import SymmetryAnalyzer
from nomad import atomutils, utils
from nomad.config import config
from nomad.units import ureg

# Conditional import of EntryArchive only works within entire NOMAD
from nomad.datamodel.datamodel import EntryArchive
from nomad.datamodel.metainfo.basesections.v2 import SubSystem as SubSystemV2
from nomad.datamodel.metainfo.basesections.v2 import System as SystemV2
from nomad.datamodel.results import (
    CoreHole,
    Material,
    Relation,
    # Results, # Unused import, uncommented to prevent VSCode from auto-deleting
    System,
    structure_name_map,
)
from nomad.datamodel.results import SymmetryNew as Symmetry

# from nomad.normalizing import Normalizer
from structlog.stdlib import BoundLogger

from nomad_topology_normalizer.normalizers.material import MaterialNormalizer
from nomad_topology_normalizer.normalizers.normalizer import Normalizer
from nomad_topology_normalizer.normalizers.symmetry_adapter import (
    apply_symmetry_data_to_results_symmetry,
    from_legacy_repr_symmetry,
    from_model_system,
    is_symmetry_data_minimally_complete,
)

conventional_description: str = (
    'The conventional cell of the material from which the '
    'subsystem is constructed from.'
)
subsystem_description: str = 'Automatically detected subsystem.'
chemical_symbols: 'NDArray[np.str_]' = np.array(_chemical_symbols)
with open(pathlib.Path(__file__).parent / 'data/top_50k_material_ids.json') as fin:
    top_50k_material_ids = json.load(fin)

# Import functions from plugin's own common.py
from nomad_topology_normalizer.normalizers.common import (
    NOMADAtoms,
    ase_atoms_from_nomad_atoms,
    cell_from_ase_atoms,
    material_id_1d,
    material_id_2d,
    material_id_bulk,
    nomad_atoms_from_ase_atoms,
    structures_2d,
    wyckoff_sets_from_matid,
)


def get_topology_id(index: int) -> str:
    """Retuns a valid topology identifier with the given index.
    Args:
        index: The index of the topology. Must be unique.

    Returns:
        An identifier string that can be stored in topology.system_id.
    """
    return f'results/material/topology/{index}'


def copy_properties_to_system(
    target_system: System,
    v2_properties,
) -> None:
    """Copy atomic_fraction, mass_fraction, and elemental_composition from v2
    schema properties (system_properties or subsystem_properties) to a
    System object.

    Args:
        target_system: The topology to populate
        v2_properties: The v2 schema properties object (system_properties or
            subsystem_properties)
    """
    if not v2_properties:
        return

    af = getattr(v2_properties, 'atomic_fraction', None)
    mf = getattr(v2_properties, 'mass_fraction', None)
    ec = getattr(v2_properties, 'elemental_composition', None)

    if af is not None:
        target_system.atomic_fraction = af
    if mf is not None:
        target_system.mass_fraction = mf
    if ec:
        target_system.elemental_composition = ec


def get_topology_original(
    particles=None, archive: EntryArchive | None = None
) -> System:
    """Creates a new topology item for the original structure.

    For v2 schema, tries to get dimensionality from archive.results.material.
    """
    dimensionality = None
    n_particles = len(particles) if particles else None

    # Try to get dimensionality from results.material
    if archive is not None:
        try:
            material = archive.results.material
            if material and material.dimensionality:
                dimensionality = material.dimensionality
        except Exception:
            pass

    original = System(
        method='parser',
        label='original',
        description='A representative system chosen from the original simulation.',
        dimensionality=dimensionality,
        system_relation=Relation(type='root'),
        n_atoms=n_particles,
    )

    return original


def add_system_info_2(  # noqa: PLR0912
    system: System,
    topologies: dict[str, System],
    parent_system: 'System | None' = None,
) -> None:
    """V2 schema version that populates system info from particle_indices.

    Args:
        system: The topology system to populate
        topologies: Dict of all topology systems
        parent_system: The parent System with positions and particle_states
    """

    def _particle_symbols(states, indices=None) -> list[str]:
        symbols = []
        if not states:
            return symbols
        iterable = range(len(states)) if indices is None else indices
        for idx in iterable:
            if idx >= len(states):
                continue
            state = states[idx]
            if state.chemical_symbol:
                symbols.append(state.chemical_symbol)
            elif state.atomic_number:
                normalized_number = _normalize_atomic_number(state.atomic_number)
                if normalized_number is not None:
                    symbols.append(chemical_symbols[normalized_number])
        return symbols

    def _normalize_atomic_number(number, symbol=None) -> int | None:
        normalized = None
        if number is not None:
            try:
                normalized = int(number)
            except Exception:
                normalized = None
        if normalized is not None:
            if normalized >= len(chemical_symbols) or normalized <= 0:
                reduced = normalized % 100 if normalized > 0 else normalized
                if 0 < reduced < len(chemical_symbols):
                    normalized = reduced
                else:
                    normalized = None
        if normalized is None and symbol:
            normalized = _atomic_numbers.get(symbol)
        return normalized

    def _populate_chemistry(system: System, symbols: list[str]) -> None:
        if not symbols:
            return
        formula = atomutils.Formula(''.join(symbols))
        if not system.elements:
            system.elements = formula.elements()
        if system.chemical_formula_reduced is None:
            system.chemical_formula_reduced = formula.format('reduced')
        if system.chemical_formula_hill is None:
            system.chemical_formula_hill = formula.format('hill')
        if system.chemical_formula_iupac is None:
            system.chemical_formula_iupac = formula.format('iupac')
        if system.chemical_formula_anonymous is None:
            system.chemical_formula_anonymous = formula.format('anonymous')
        if system.chemical_formula_descriptive is None:
            system.chemical_formula_descriptive = formula.format('descriptive')
        if not system.elemental_composition:
            system.elemental_composition = formula.elemental_composition()

    def _nomad_atoms_from_model_system(model_system) -> NOMADAtoms | None:
        positions = model_system.positions
        if positions is None:
            return None

        def _to_meter_array(value) -> np.ndarray | None:
            if value is None:
                return None
            try:
                if hasattr(value, 'to'):
                    arr = np.asarray(value.to(ureg.meter).magnitude, dtype=float)
                else:
                    arr = np.asarray(value, dtype=float)
            except Exception:
                return None
            if arr.ndim != 2 or arr.shape[1] != 3:
                return None
            if not np.isfinite(arr).all():
                return None
            # Heuristic: atomistic coordinates/cell vectors should not be in the
            # micrometer range when expressed in meters. If they are, values are
            # likely angstrom-like payloads that lost units upstream.
            if arr.size and np.nanmax(np.abs(arr)) > 1e-5:
                arr = arr * 1e-10
            return arr

        pos_arr = _to_meter_array(positions)
        if pos_arr is None:
            return None

        particle_states = model_system.particle_states or []
        labels = []
        atomic_numbers = []
        for state in particle_states:
            symbol = state.chemical_symbol
            number = state.atomic_number
            normalized_number = _normalize_atomic_number(number, symbol)
            if symbol is None and normalized_number is not None:
                symbol = chemical_symbols[normalized_number]
            elif symbol is not None and normalized_number is None:
                normalized_number = _atomic_numbers.get(symbol)
            if normalized_number is None or normalized_number <= 0:
                normalized_number = 1
            labels.append(symbol if symbol is not None else 'X')
            atomic_numbers.append(int(normalized_number))

        n_positions = len(pos_arr)
        if len(labels) != n_positions:
            # Keep array lengths consistent with coordinates.
            if len(labels) < n_positions:
                missing = n_positions - len(labels)
                labels.extend(['X'] * missing)
                atomic_numbers.extend([1] * missing)
            else:
                labels = labels[:n_positions]
                atomic_numbers = atomic_numbers[:n_positions]

        atoms = NOMADAtoms()
        atoms.positions = pos_arr * ureg.meter
        atoms.labels = labels
        atoms.atomic_numbers = atomic_numbers
        atoms.species = atomic_numbers
        lattice_vectors = model_system.lattice_vectors
        lattice_arr = _to_meter_array(lattice_vectors)
        if lattice_arr is not None:
            atoms.lattice_vectors = lattice_arr * ureg.meter
        pbc = model_system.periodic_boundary_conditions
        if pbc is not None:
            pbc_arr = np.asarray(pbc, dtype=bool).reshape(-1)
            if pbc_arr.size >= 3:
                atoms.periodic = pbc_arr[:3].tolist()

        return atoms

    # Root/original node: derive atoms/cell directly from representative
    # v2 ModelSystem. Keep indices unset because downstream GUI logic uses
    # "no indices" to identify roots.
    relation = system.system_relation.type if system.system_relation else None
    if relation == 'root' and parent_system is not None:
        try:
            particle_states = parent_system.particle_states
            if system.n_atoms is None and particle_states:
                system.n_atoms = len(particle_states)

            ase_atoms = None
            try:
                ase_atoms = parent_system.to_ase_atoms()
            except Exception:
                ase_atoms = None

            if ase_atoms is not None:
                # Populate System.atoms for GUI compatibility (molecular visualizer needs this)
                # Even though this is runschema which is being phased out, the GUI still depends on it
                system.atoms = nomad_atoms_from_ase_atoms(ase_atoms)

                if system.cell is None:
                    system.cell = cell_from_ase_atoms(ase_atoms)
            elif system.atoms is None:
                # Fallback for inconsistent v2 geometry payloads where ASE conversion fails.
                system.atoms = _nomad_atoms_from_model_system(parent_system)

            symbols = _particle_symbols(particle_states)
            _populate_chemistry(system, symbols)
        except Exception:
            pass

    # Check if indices are available (required for subsystem-specific info)
    if system.indices is None or len(system.indices) == 0:
        return

    # Calculate n_atoms from first instance of particle indices
    first_instance = system.indices[0]
    if system.n_atoms is None:
        system.n_atoms = len(first_instance)

    # Calculate atomic_fraction relative to parent system
    if system.parent_system:
        parent = topologies.get(system.parent_system)
        if parent and parent.n_atoms:
            system.atomic_fraction = system.n_atoms / parent.n_atoms

    # Populate parent system with system info
    if not parent_system:
        return

    try:
        particle_states = parent_system.particle_states
        positions = parent_system.positions

        if not particle_states or len(particle_states) == 0:
            return
        if positions is None or len(positions) == 0:
            return

        # Extract symbols for formula calculation
        symbols = _particle_symbols(particle_states, first_instance)
        if not symbols:
            return

        # Calculate chemistry descriptors from the selected particle symbols.
        _populate_chemistry(system, symbols)
    except Exception:
        pass


def add_system(
    system: System, topologies: dict[str, System], parent: System | None = None
) -> None:
    """Adds the given system to the topology."""
    index = len(topologies)
    system.system_id = get_topology_id(index)
    if parent:
        children = parent.child_systems if parent.child_systems else []
        children.append(system.system_id)
        if parent.child_systems is not children:
            parent.child_systems = children
        system.parent_system = parent.system_id
    topologies[system.system_id] = system


class TopologyNormalizer(Normalizer):
    """Topology normalizer for material structure analysis.

    Inherits from both local Normalizer (for helper methods) and
    nomad.normalizing.Normalizer (for plugin compatibility).
    """

    @staticmethod
    def _merge_missing_symmetry_data(
        primary: dict[str, object], fallback: dict[str, object]
    ) -> dict[str, object]:
        merged = dict(primary)
        for key, value in fallback.items():
            if merged.get(key) is None and value is not None:
                merged[key] = value
        return merged

    def _resolved_symmetry_data(self) -> dict[str, object]:
        symmetry_data = from_model_system(self.repr_system)
        if not is_symmetry_data_minimally_complete(symmetry_data):
            symmetry_data = self._merge_missing_symmetry_data(
                symmetry_data, from_legacy_repr_symmetry(self.repr_symmetry)
            )
        return symmetry_data

    def _has_complete_model_system_symmetry(self) -> bool:
        """True when v2 ModelSystem symmetry is complete without legacy fallback."""
        return is_symmetry_data_minimally_complete(from_model_system(self.repr_system))

    @staticmethod
    def _symmetry_from_data(symmetry_data: dict[str, object]) -> Symmetry | None:
        if not any(value is not None for value in symmetry_data.values()):
            return None
        result = Symmetry()
        apply_symmetry_data_to_results_symmetry(result, symmetry_data)
        return result

    def _initialize_representative_system(
        self, archive: 'EntryArchive', system_v2=None
    ) -> None:
        """Initialize repr_system, repr_symmetry, and conv_atoms.

        Uses v2 data schema (SystemV2/Simulation) only.
        """
        self.repr_system = None
        self.repr_symmetry = None
        self.conv_atoms = None
        self.masses = None

        # Get representative system from data
        self.repr_system = self._representative_system(archive)
        if self.repr_system is None and system_v2 is not None:
            self.repr_system = system_v2

        # Get symmetry from results if available
        self._get_symmetry_from_results(archive)

    def _get_symmetry_from_results(self, archive: 'EntryArchive') -> None:
        """Get symmetry and conv_atoms from results if available."""
        try:
            if archive.results and archive.results.properties:
                structures = archive.results.properties.structures
                if structures:
                    self.repr_symmetry = structures.structure_original
        except Exception:
            pass

    def normalize(
        self, archive: 'EntryArchive', logger: 'BoundLogger', system_v2=None
    ) -> None:
        """
        Main normalization entry point for v2 data schema.

        Called by ResultsNormalizer after schema version detection.
        This only handles v2 data schema (archive.data).
        """
        self.entry_archive = archive
        self.logger = logger

        # Initialize representative system and related attributes
        self._initialize_representative_system(archive, system_v2)

        if self.entry_archive.results.material is None:
            self.entry_archive.results.material = MaterialNormalizer(
                self.entry_archive,
                self.repr_system,
                self.repr_symmetry,
                None,  # spg_number unknown at this stage in v2 path
                self.conv_atoms,
                None,  # wyckoff_sets unknown at this stage in v2 path
                self.entry_archive.results.properties,
                None,  # optimade not used in v2 path
                logger,
            ).material(populate_topology=False)

        if self.entry_archive.results and self.entry_archive.results.material:
            topology = self.topology(self.entry_archive.results.material, system_v2)
            if topology:
                self.entry_archive.results.material.topology.extend(topology)

    def topology(self, material, system_v2=None) -> list[System] | None:
        """Returns a dictionary that contains all of the topologies mapped by id."""
        # If topology already exists (e.g. written by another normalizer), do
        # not overwrite it.
        topology = self.entry_archive.m_xpath('results.material.topology')
        if topology:
            return None

        # First: topology from data schema calculation (v2)
        topology = self.topology_calculation()

        # Second: create topology with MatID
        skip_matid_bulk = bool(
            material
            and material.structural_type == 'bulk'
            and self._has_complete_model_system_symmetry()
        )
        if topology is None and not skip_matid_bulk:
            with utils.timer(self.logger, 'calculating topology with matid'):
                topology = self.topology_matid(material)

        # Third: fallback to topology_data for SystemV2 entries
        if topology is None:
            if system_v2 is not None:
                topology = self.topology_data(system_v2)
            else:
                data = self.entry_archive.data
                if data and isinstance(data, SystemV2):
                    topology = self.topology_data(data)

        return topology

    def topology_calculation(self) -> list[System] | None:
        """Extracts the system topology as defined in the original calculation."""
        system = None
        groups = None
        result = None

        # Prefer the already resolved representative system. This keeps
        # topology/root structure payload aligned with the system selected for
        # results/material population.
        if isinstance(self.repr_system, SystemV2):
            system = self.repr_system

        # Fallback to first model_system for safety if representative resolution
        # is unavailable.
        if system is None:
            data = self.entry_archive.data
            try:
                if (
                    data
                    and isinstance(
                        data,
                        __import__(
                            'nomad_simulations.schema_packages.general',
                            fromlist=['Simulation'],
                        ).Simulation,
                    )
                    and data.model_system
                    and len(data.model_system) > 0
                ):
                    system = data.model_system[0]
            except (AttributeError, IndexError):
                pass

        # Validate system type and extract groups
        if system and isinstance(system, SystemV2):
            try:
                groups = system.sub_systems
            except Exception:
                pass

            positions = getattr(system, 'positions', None)
            particle_states = getattr(system, 'particle_states', None)

            # Validate system has required data
            has_valid_data = (
                groups
                and len(groups) > 0
                and positions is not None
                and len(positions) > 0
                and particle_states
                and len(particle_states) > 0
            )

            if has_valid_data:
                topology: dict[str, System] = {}
                original = get_topology_original(particle_states, self.entry_archive)
                add_system(original, topology)
                label_to_indices: dict[str, list] = defaultdict(list)

                # Define mapping dictionaries once to avoid recreating them
                description_map = {
                    'molecule': 'Molecule extracted from the calculation topology.',
                    'molecule_group': 'Group of molecules extracted from the '
                    'calculation topology.',
                    'monomer_group': 'Group of monomers extracted from the '
                    'calculation topology.',
                    'monomer': 'Monomer extracted from the calculation topology.',
                    'active_orbitals': 'Orbitals targeted by the calculation.',
                }
                structural_type_map = {
                    'active_orbitals': 'active orbitals',
                    'molecule': 'molecule',
                    'molecule_group': 'group',
                    'monomer': 'monomer',
                    'monomer_group': 'group',
                }
                building_block_map = {
                    'molecule': 'molecule',
                    'monomer': 'monomer',
                }
                relation_map = {
                    'active_orbitals': 'group',
                    'molecule': 'subsystem',
                    'molecule_group': 'group',
                    'monomer': 'subsystem',
                    'monomer_group': 'group',
                }

                def add_group(groups, parent=None):
                    if not groups:
                        return None
                    for group in groups:
                        label = group.name
                        # Groups with same label mapped to same system
                        # TODO: change for active orbitals
                        old_labels = label_to_indices[label]
                        instance_indices = group.particle_indices
                        if not old_labels:
                            system = System(
                                method='parser',
                                description=description_map.get(group.branch_label),
                                label=group.name,
                                structural_type=structural_type_map.get(
                                    group.branch_label
                                ),
                                building_block=building_block_map.get(
                                    group.branch_label
                                ),
                                system_relation=Relation(
                                    type=relation_map.get(group.branch_label)
                                ),
                            )
                            add_system(system, topology, parent)
                            add_group(group.sub_systems, system)
                            old_labels.append(instance_indices)
                        elif len(old_labels[0]) == len(instance_indices):
                            old_labels.append(instance_indices)
                        else:
                            self.logger.warn(
                                'The topology contains entries with the same label '
                                'but with different numbers of atoms'
                            )

                add_group(groups, original)
                active_orbital_states = self._extract_orbital()

                # Add derived system information once all indices gathered
                for top in topology.values():
                    top.indices = label_to_indices.get(top.label)
                    add_system_info_2(top, topology, parent_system=system)
                    if top.structural_type == 'active orbitals':
                        try:
                            top.active_orbitals = active_orbital_states[0]
                            active_orbital_states.pop(0)
                        except IndexError:
                            # FIXME: temporary fix to prevent projection parser output
                            self.logger.warn(
                                'Cannot assign all active orbital states to topology.'
                            )

                result = list(topology.values())

        return result

    def topology_matid(self, material: Material) -> list[SystemV2] | None:  # noqa: PLR0912
        """
        Returns a list of systems that have been identified with MatID.
        """
        # See if a system is available

        try:
            # TODO: TEST! Updated to get ase atoms from ModelSystem
            atoms = self.repr_system.to_ase_atoms()
        except Exception:
            return None
        if not atoms or len(atoms) == 0:
            return None

        # Create topology for the original system
        topology: dict[str, System] = {}
        particles = self.repr_system.particle_states
        original = get_topology_original(particles, self.entry_archive)
        # Keep atoms_ref for compatibility with old topology_matid code
        add_system(original, topology)
        add_system_info_2(original, topology, parent_system=self.repr_system)

        # Since we still need to run the old classification code
        # (matid.classification.classify), we use it's results to populate the
        # topology for bulk and 1D systems. Also the new clustering cannot
        # currently be run for systems without a cell. In other cases we run the
        # new classification code (matid.clustering.clusterer).
        n_atoms = len(atoms)
        cell = atoms.get_cell()
        if material.structural_type == 'bulk':
            self._topology_bulk(original, topology, material)
        elif material.structural_type == '1D':
            self._topology_1d(original, topology)
        # Continue creating topology if system size is not too large
        elif n_atoms <= config.normalize.clustering_size_limit:
            # Add all meaningful clusters to the topology
            sbc = SBC()
            try:
                clusters = sbc.get_clusters(atoms, pos_tol=0.8)
            except Exception as e:
                self.logger.warning(
                    'issue in matid clustering',
                    exc_info=e,
                    error=str(e),
                )
            else:
                for cluster in clusters:
                    subsystem = self._create_subsystem(cluster)
                    if not subsystem:
                        continue
                    structural_type = subsystem.structural_type
                    # If the found cell has many basis atoms, it is more likely that
                    # some of the symmetries were not correctly found than the cell
                    # actually being very complicated. Thus we ignore these clusters to
                    # minimize false-positive and to limit the time spent on symmetry
                    # calculation.
                    cell = cluster.get_cell()
                    MAX_CELL_ATOMS = 8
                    if len(cell) > MAX_CELL_ATOMS:
                        self.logger.info(
                            f'cell with many atoms ({len(cell)}) was ignored'
                        )
                        continue
                    try:
                        conventional_cell = self._create_conv_cell_system(
                            cluster, structural_type
                        )
                    except Exception as e:
                        self.logger.error(
                            'conventional cell information could not be created',
                            exc_info=e,
                            error=str(e),
                        )
                        continue
                    # We only accept the subsystem if the material id exists in the top
                    # 50k materials with most entries attached to them. This ensures
                    # that the material_id link points to valid materials and that we
                    # don't report anything too weird. The top 50k materials are
                    # pre-stored in a pickle file that has been created by using ES
                    # terms aggregation.
                    if conventional_cell.material_id in top_50k_material_ids:
                        add_system(subsystem, topology, original)
                        add_system_info_2(
                            subsystem, topology, parent_system=self.repr_system
                        )
                        add_system(conventional_cell, topology, subsystem)
                        add_system_info_2(
                            conventional_cell,
                            topology,
                            parent_system=self.repr_system,
                        )
                    else:
                        self.logger.info(
                            f'material_id {conventional_cell.material_id} '
                            'could not be verified'
                        )

        return list(topology.values())

    def topology_data(self, section: SystemV2, path: str = '') -> list[System]:
        topology: dict[str, System] = {}
        try:
            particles = self.repr_system.particle_states
        except Exception:
            particles = None
        original = get_topology_original(particles, self.entry_archive)
        add_system(original, topology)
        add_system_info_2(original, topology, parent_system=self.repr_system)
        root_id = path or section.name or section.m_def.name
        root = System(
            method='parser',
            label=root_id,
            description='Imported from v2.System',
        )
        v2_props = getattr(section, 'subsystem_properties', None) or getattr(
            section, 'system_properties', None
        )
        copy_properties_to_system(root, v2_props)
        add_system(root, topology, original)
        add_system_info_2(root, topology, parent_system=self.repr_system)

        def recurse(v2sec: SystemV2, parent_sys: System, cur_path: str):
            for proxy in getattr(v2sec, 'sub_systems', []) or []:
                sub = getattr(proxy, 'value', proxy)
                if not isinstance(sub, SubSystemV2):
                    continue

                child_name = getattr(sub, 'name', None)
                child_id = f'{cur_path}/{child_name or ""}'
                child = System(
                    method='parser',
                    label=child_name or 'subsystem',
                    description='Imported from v2.System',
                )

                sub_props = getattr(sub, 'subsystem_properties', None)
                copy_properties_to_system(child, sub_props)
                add_system(child, topology, parent_sys)
                add_system_info_2(child, topology, parent_system=self.repr_system)

                recurse(sub, child, child_id)

        recurse(section, root, root_id)

        return list(topology.values())

    def _topology_bulk(
        self, original, topology, material: Material | None = None
    ) -> None:
        """Creates a topology for bulk structures as detected by the old matid
        classification."""
        if self.conv_atoms is None:
            return None

        # Subsystem
        subsystem = System(
            method='matid',
            label='subsystem',
            dimensionality='3D',
            structural_type='bulk',
            description=subsystem_description,
            system_relation=Relation(type='subsystem'),
            indices=[list(range(original.n_atoms))],
        )
        add_system(subsystem, topology, original)
        add_system_info_2(subsystem, topology, parent_system=self.repr_system)

        # Conventional system
        conv_system = System(
            method='matid',
            label='conventional cell',
            system_relation=Relation(type='conventional_cell'),
            dimensionality='3D',
            structural_type='bulk',
            description=conventional_description,
        )
        conv_system.atoms = nomad_atoms_from_ase_atoms(self.conv_atoms)  # GUI visualizer needs this
        symmetry_analyzer = None
        m_cache = getattr(self.repr_symmetry, 'm_cache', None)
        if m_cache is not None and hasattr(m_cache, 'get'):
            symmetry_analyzer = m_cache.get('symmetry_analyzer')

        symmetry_data = self._resolved_symmetry_data()
        conv_system.symmetry = self._symmetry_from_data(symmetry_data)
        if conv_system.symmetry is None and symmetry_analyzer is not None:
            conv_system.symmetry = self._create_symmetry(symmetry_analyzer)
        conv_system.cell = cell_from_ase_atoms(
            self.conv_atoms, masses=self.masses, atom_labels=None
        )
        if material is not None and material.material_id:
            conv_system.material_id = material.material_id
        elif symmetry_analyzer is not None:
            conv_system.material_id = material_id_bulk(
                symmetry_analyzer.get_space_group_number(),
                symmetry_analyzer.get_wyckoff_sets_conventional(),
            )
        add_system(conv_system, topology, subsystem)
        add_system_info_2(conv_system, topology, parent_system=self.repr_system)

    def _topology_1d(self, original, topology):
        """Creates a topology for 1D structures as detected by the old matid
        classification."""
        if self.conv_atoms is None:
            return None

        # Subsystem
        subsystem = System(
            method='matid',
            label='subsystem',
            dimensionality='1D',
            structural_type='1D',
            description=subsystem_description,
            system_relation=Relation(type='subsystem'),
            indices=[list(range(original.n_atoms))],
        )
        add_system(subsystem, topology, original)
        add_system_info_2(subsystem, topology, parent_system=self.repr_system)

        # Conventional system
        conv_system = System(
            method='matid',
            label='conventional cell',
            system_relation=Relation(type='conventional_cell'),
            dimensionality='1D',
            structural_type='1D',
        )
        conv_system.atoms = nomad_atoms_from_ase_atoms(self.conv_atoms)  # GUI visualizer needs this
        conv_system.cell = cell_from_ase_atoms(
            self.conv_atoms, masses=self.masses, atom_labels=None
        )

        # The lattice parameters that are not well defined for 1D structures are unset
        conv_system.cell.b = None
        conv_system.cell.c = None
        conv_system.cell.alpha = None
        conv_system.cell.beta = None
        conv_system.cell.gamma = None
        conv_system.cell.atomic_density = None
        conv_system.cell.mass_density = None
        conv_system.cell.volume = None

        conv_system.material_id = material_id_1d(self.conv_atoms)
        add_system(conv_system, topology, subsystem)
        add_system_info_2(conv_system, topology, parent_system=self.repr_system)

    def _create_subsystem(self, cluster: Cluster) -> System | None:
        """
        Creates a new subsystem as detected by MatID.
        """
        try:
            dimensionality = cluster.get_dimensionality()
            cell = cluster.get_cell()
            n_repeated_directions = sum(cell.get_pbc())
        except Exception as e:
            self.logger.error(
                'matid cluster classification failed', exc_info=e, error=str(e)
            )
            return None
        structural_type = None
        building_block = None
        BULK_DIM = 3
        SURFACE_DIM = 2
        if dimensionality == BULK_DIM:
            structural_type = 'bulk'
        elif dimensionality == SURFACE_DIM:
            if n_repeated_directions == SURFACE_DIM:
                structural_type = '2D'
                building_block = '2D material'
            elif n_repeated_directions == BULK_DIM:
                structural_type = 'surface'
                building_block = 'surface'
        if not structural_type:
            return None

        subsystem = System(
            method='matid',
            label='subsystem',
            description=subsystem_description,
            system_relation=Relation(type='subsystem'),
            indices=[list(cluster.indices)],
        )

        subsystem.dimensionality = f'{dimensionality}D'
        subsystem.structural_type = structural_type
        subsystem.building_block = building_block

        return subsystem

    def _create_conv_cell_system(self, cluster: Cluster, structural_type: str):
        """
        Creates a new topology item for a conventional cell.
        """
        symmsystem = System(
            method='matid',
            label='conventional cell',
            system_relation=Relation(type='conventional_cell'),
        )
        if structural_type == '2D':
            self._add_conventional_2d(cluster, symmsystem)
        else:
            self._add_conventional_bulk(cluster, symmsystem)
        symmsystem.description = conventional_description

        return symmsystem

    def _add_conventional_bulk(self, cluster: Cluster, subsystem: System) -> None:
        """
        Creates the subsystem with the symmetry information of the conventional cell
        """
        cell = cluster.get_cell()
        # A big tolerance is used here to allow deviations from exact symmetry
        symm = SymmetryAnalyzer(cell, 1.0)
        conv_system = symm.get_conventional_system()
        subsystem.atoms = nomad_atoms_from_ase_atoms(conv_system)  # GUI visualizer needs this
        spg_number = symm.get_space_group_number()
        subsystem.cell = cell_from_ase_atoms(
            conv_system, masses=self.masses, atom_labels=None
        )
        symmetry = self._create_symmetry(symm)
        wyckoff_sets = symm.get_wyckoff_sets_conventional()
        material_id = material_id_bulk(spg_number, wyckoff_sets)
        subsystem.structural_type = 'bulk'
        subsystem.dimensionality = '3D'
        subsystem.material_id = material_id
        subsystem.symmetry = symmetry

    def _add_conventional_2d(self, cluster: Cluster, subsystem: System) -> None:
        """
        Creates the subsystem with the symmetry information of the conventional cell.
        """
        cell = cluster.get_cell()
        conv_atoms, _, wyckoff_sets, spg_number = structures_2d(cell)
        subsystem.cell = cell_from_ase_atoms(
            conv_atoms, masses=self.masses, atom_labels=None
        )
        subsystem.atoms = nomad_atoms_from_ase_atoms(conv_atoms)  # GUI visualizer needs this

        # Here we zero out the irrelevant lattice parameters to correctly handle
        # 2D systems with nonzero thickness (e.g. MoS2).
        subsystem.cell.c = None
        subsystem.cell.alpha = None
        subsystem.cell.beta = None
        subsystem.cell.atomic_density = None
        subsystem.cell.mass_density = None
        subsystem.cell.volume = None

        subsystem.structural_type = '2D'
        subsystem.dimensionality = '2D'
        subsystem.building_block = '2D material'
        subsystem.material_id = material_id_2d(spg_number, wyckoff_sets)

    def _create_symmetry(self, symm: SymmetryAnalyzer) -> Symmetry:
        international_short = symm.get_space_group_international_short()
        conv_system = symm.get_conventional_system()

        sec_symmetry = Symmetry()
        sec_symmetry.symmetry_method = 'MatID'
        sec_symmetry.space_group_number = symm.get_space_group_number()
        sec_symmetry.space_group_symbol = international_short
        sec_symmetry.hall_number = symm.get_hall_number()
        sec_symmetry.hall_symbol = symm.get_hall_symbol()
        sec_symmetry.point_group = symm.get_point_group()
        sec_symmetry.crystal_system = symm.get_crystal_system()
        sec_symmetry.bravais_lattice = symm.get_bravais_lattice()
        sec_symmetry.origin_shift = symm._get_spglib_origin_shift()
        sec_symmetry.transformation_matrix = symm._get_spglib_transformation_matrix()
        sec_symmetry.wyckoff_sets = wyckoff_sets_from_matid(
            symm.get_wyckoff_sets_conventional()
        )

        spg_number = symm.get_space_group_number()
        atom_species = conv_system.get_atomic_numbers()
        if type(conv_system) is Atoms or conv_system.wyckoff_letters is None:
            wyckoffs = symm.get_wyckoff_letters_conventional()
        else:
            wyckoffs = conv_system.wyckoff_letters
        norm_wyckoff = atomutils.get_normalized_wyckoff(atom_species, wyckoffs)
        protoDict = atomutils.search_aflow_prototype(spg_number, norm_wyckoff)

        if protoDict is not None:
            sec_symmetry.prototype_label_aflow = protoDict.get('aflow_prototype_id')
            sec_symmetry.prototype_name = structure_name_map.get(protoDict.get('Notes'))

        return sec_symmetry

    def _extract_orbital(self) -> list[CoreHole]:
        """
        Gather atomic orbitals from v2 schema:
        `data.model_system[].particle_states[].core_hole`.
        """
        # Validate data structure exists
        data = self.entry_archive.data
        if not data or not data.model_system:
            return []

        # Search for first core_hole in model_system hierarchy
        for model_system in data.model_system:
            particle_states = model_system.particle_states or []
            for particle_state in particle_states:
                core_hole = getattr(particle_state, 'core_hole', None)
                if core_hole is not None:
                    # Normalize and create a deep copy
                    core_hole.normalize(EntryArchive(), None)
                    core_hole_new = CoreHole()
                    for quantity_name in core_hole.quantities:
                        setattr(
                            core_hole_new,
                            quantity_name,
                            getattr(core_hole, quantity_name),
                        )
                    core_hole_new.normalize(None, None)
                    return [core_hole_new]

        return []
