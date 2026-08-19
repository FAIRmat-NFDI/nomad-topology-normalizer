#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Results Normalizer - Entry Point for Data-Schema Normalization
==============================================================

This module provides the main entry point for nomad-simulations data-schema
results normalization.

NORMALIZATION CASCADE ARCHITECTURE
-----------------------------------

Plugin Entry Point: results_normalizer_plugin (level 3)
    │
    ├─── Detect data-schema availability: _is_data_schema(archive)
    │
    └─── When data-schema is available:
            │
            └─ Data-schema normalization
                              │
                              ├─ Initialize results sections
                              └─ TopologyNormalizer.normalize()

Schema Version Detection
------------------------

The _is_data_schema() method checks whether an entry additionally uses the
nomad-simulations data-schema path by checking:
1. archive.data attribute exists
2. archive.data.model_system attribute exists

Entries with nomad-simulations data run the plugin's data-schema
normalization. Other archives are left to the built-in NOMAD normalizers.

Design Principles
-----------------

- Single entry point: Adds data-schema result population after built-in normalizers
- Precise detection: Only simulation data-schema archives trigger the plugin pass
- Clean separation: data-schema code stays in plugin, legacy code stays in nomad-FAIR
- No breaking changes: Existing entries continue to receive legacy results
"""

import re
from typing import Any

import ase.data
import numpy as np
from nomad.atomutils import Formula
from nomad.datamodel import EntryArchive
from nomad.datamodel.results import (
    BSE,
    DFT,
    DMFT,
    GW,
    TB,
    BandGap,
    EELSMethodology,
    ElectronicProperties,
    EnergyDynamic,
    GeometryOptimization,
    Material,
    MDProvenance,
    Method,
    Properties,
    RadiusOfGyration,
    Results,
    Simulation,
    Spectra,
    SpectraProvenance,
    SpectroscopicProperties,
    StructuralProperties,
    TemperatureDynamic,
    ThermodynamicProperties,
    Trajectory,
)
from nomad.units import ureg
from nomad.utils import traverse_reversed

# Don't import local Normalizer to avoid circular import
# ResultsNormalizer inherits directly from nomad.normalizing.Normalizer

re_label = re.compile('^([a-zA-Z][a-zA-Z]?)[^a-zA-Z]*')
elements = set(ase.data.chemical_symbols)

DATA_SCHEMA_COMPATIBILITY_ANNOTATION = (
    'nomad_results_normalizer_data_schema_compatibility'
)
_LEGACY_DATA_SCHEMA_COMPATIBILITY_ANNOTATION = (
    'nomad_topology_normalizer_v2_compatibility'
)
_GENERATED_COMPATIBILITY_ANNOTATIONS = {
    DATA_SCHEMA_COMPATIBILITY_ANNOTATION,
    _LEGACY_DATA_SCHEMA_COMPATIBILITY_ANNOTATION,
}


def valid_array(array: Any) -> bool:
    """Checks if the given variable is a non-empty array."""
    return array is not None and len(array) > 0


def isint(value: Any) -> bool:
    """Checks if the given variable can be interpreted as an integer."""
    try:
        int(value)
        return True
    except ValueError:
        return False


# CIRCULAR IMPORT WORKAROUND:
# =========================
# This class does NOT inherit from nomad.normalizing.Normalizer to avoid
# circular imports.
#
# The Problem:
# When nomad.normalizing.__init__.py loads entry points, it imports this module.
# If this class inherited from nomad.normalizing.Normalizer, it would trigger:
#   results.py -> import nomad.normalizing.Normalizer
#              -> nomad.normalizing.__init__.py (still loading)
#              -> loads entry points
#              -> imports results.py again (CIRCULAR!)
#
# The Solution:
# 1. ResultsNormalizerBase is a plain class (no base class)
# 2. The entry point's load() method creates the actual ResultsNormalizer class
#    dynamically using type() with proper inheritance from nomad.normalizing.Normalizer
# 3. This ensures nomad.normalizing has finished initializing before we inherit from it
#
# See:
# nomad_results_normalizer/normalizers/__init__.py::ResultsNormalizerEntryPoint.load()
class ResultsNormalizerBase:
    """Results normalizer implementation with schema version detection.

    WARNING: This is NOT the actual normalizer class used at runtime!
    The entry point creates a proper subclass dynamically. See class docstring above.

    Strategy:
    If nomad-simulations data is present, run the data-schema cascade to
    add/override data-schema-derived results. Archives without data-schema
    content are handled by NOMAD's built-in normalizers.
    """

    domain = None
    normalizer_level = 3

    def normalize(self, archive: EntryArchive, logger=None) -> None:
        self.entry_archive = archive

        # Setup logger
        if logger is not None:
            self.logger = logger.bind(normalizer=self.__class__.__name__)

        # ========== LOGICAL SWITCH: data-schema pass ==========
        data_schema_info = self._is_data_schema(archive)

        if data_schema_info:
            from nomad_results_normalizer.normalizers.topology import TopologyNormalizer

            system_v2 = data_schema_info if data_schema_info is not True else None
            self.logger.info('Running data-schema results normalization')

            results = self.entry_archive.results
            if results is None:
                results = self.entry_archive.m_create(Results)
            if results.properties is None:
                results.m_create(Properties)

            topology_normalizer = TopologyNormalizer()
            topology_normalizer.normalize(archive, self.logger, system_v2=system_v2)
            self._normalize_method_with_data_schema(archive)
            self._normalize_outputs_with_data_schema(archive)
        else:
            self.logger.info('Skipping data-schema results normalization')

        self.entry_archive = None

    def _is_data_schema(self, archive: EntryArchive) -> Any:
        """Check if archive uses the nomad-simulations data-schema path.

        Returns a SystemV2 instance if found, True if indicated by
        model_system but no instance found, or False.
        """
        if archive.data is None:
            return False

        # Check if data has model_system (Simulation-style schema indicator).
        try:
            has_model_system = archive.data.model_system is not None
        except Exception:
            has_model_system = False

        # Some data-schema archives still embed a basesections v2 System.
        from nomad.datamodel.metainfo.basesections.v2 import System as SystemV2

        for sec in archive.data.m_all_contents(include_self=True):
            if isinstance(sec, SystemV2):
                return sec

        # If model_system attribute exists but is empty, still consider it a
        # data-schema archive (e.g. partially parsed Simulation).
        if has_model_system:
            return True

        # Non-simulation custom data sections (and any other data without
        # model_system) are not handled by this plugin.
        return False

    def _normalize_method_with_data_schema(self, archive: EntryArchive) -> None:
        """Populate results.method from nomad-simulations data when available."""

        def _enum_values(section_cls, quantity_name: str) -> set[str]:
            return set(section_cls.m_def.all_quantities[quantity_name].type)

        def _set_if_enum(target, quantity_name: str, value) -> None:
            if value is None:
                return
            valid_values = _enum_values(type(target), quantity_name)
            if value in valid_values:
                setattr(target, quantity_name, value)

        def _xc_names_from_model_method(model_method) -> list[str]:
            names = []
            xc = getattr(model_method, 'xc', None)
            components = getattr(xc, 'components', None) if xc else None
            for comp in components or []:
                label = getattr(comp, 'canonical_label', None) or getattr(
                    comp, 'name', None
                )
                if label:
                    names.append(label)
            # de-duplicate while preserving order
            return list(dict.fromkeys(names))

        def _jacobs_from_xc_names(xc_names: list[str]) -> str | None:
            if not xc_names:
                return None
            rung_order = {'lda': 0, 'gga': 1, 'mgg': 2, 'hyb_mgg': 3, 'hyb': 4}
            rung_to_value = {
                'lda': 'LDA',
                'gga': 'GGA',
                'mgg': 'meta-GGA',
                'hyb_mgg': 'hyper-GGA',
                'hyb': 'hybrid',
            }
            regex = re.compile(r'((HYB_)?[A-Z]{3})')
            abbrevs = []
            for name in xc_names:
                match = regex.match(name)
                if not match:
                    continue
                token = match.group(1)
                token = token.lower() if token == 'HYB_MGG' else token[:3].lower()
                if token in rung_order:
                    abbrevs.append(token)
            if not abbrevs:
                return None
            highest = max(abbrevs, key=lambda x: rung_order[x])
            return rung_to_value.get(highest)

        def _basis_set_type_from_model_method(model_method, program_name: str | None):
            settings = model_method.numerical_settings or []
            for setting in settings:
                setting_name = getattr(getattr(setting, 'm_def', None), 'name', None)
                if setting_name != 'BasisSetContainer':
                    continue
                components = setting.basis_set_components or []
                component_names = [
                    getattr(getattr(comp, 'm_def', None), 'name', '')
                    for comp in components
                ]
                if any('PlaneWave' in name for name in component_names):
                    return 'plane waves'
                if any('APW' in name for name in component_names):
                    return '(L)APW+lo'
            # Legacy-equivalent fallback for common plane-wave codes.
            if program_name and program_name.lower() in ('vasp', 'quantum espresso'):
                return 'plane waves'
            return None

        def _core_electron_treatment_from_model_method(
            model_method, program_name: str | None
        ):
            settings = model_method.numerical_settings or []
            for setting in settings:
                setting_name = getattr(getattr(setting, 'm_def', None), 'name', None)
                if setting_name == 'Pseudopotential':
                    return 'pseudopotential'
            if program_name and program_name.lower() in ('vasp', 'quantum espresso'):
                return 'pseudopotential'
            if program_name and program_name.lower() in (
                'fhi-aims',
                'exciting',
                'elk',
                'wien2k',
            ):
                return 'full all electron'
            return None

        def _scf_threshold_from_model_method(model_method):
            settings = model_method.numerical_settings or []
            for setting in settings:
                setting_name = getattr(getattr(setting, 'm_def', None), 'name', None)
                if setting_name == 'SelfConsistency':
                    threshold_change = getattr(setting, 'threshold_change', None)
                    if threshold_change is not None:
                        # The current flexible-unit schema returns a Pint quantity
                        # directly. Older schemas stored a separate unit string.
                        if getattr(threshold_change, 'units', None) is not None:
                            return threshold_change
                        threshold_unit = getattr(setting, 'threshold_change_unit', None)
                        if threshold_unit:
                            try:
                                return threshold_change * ureg(threshold_unit)
                            except Exception:
                                pass
                        return threshold_change
            return None

        def _energy_threshold_for_results(value):
            try:
                if getattr(value, 'units', None) is not None:
                    return value.to('joule').magnitude
                # Some released nomad-simulations versions accept Pint input
                # for flexible-unit quantities but expose only the original
                # magnitude. For SCF energy thresholds, legacy results expect
                # eV-like values to be converted into their joule storage unit.
                return (value * ureg.eV).to('joule').magnitude
            except Exception:
                return value

        def _map_dft_fields(
            model_method, simulation_method: DFT, program_name: str | None
        ) -> None:
            # Keep legacy-equivalent transfer only.
            is_spin_polarized = model_method.is_spin_polarized
            if is_spin_polarized is not None:
                simulation_method.spin_polarized = bool(is_spin_polarized)

            jacobs_ladder = model_method.jacobs_ladder
            xc_names = _xc_names_from_model_method(model_method)
            if xc_names:
                simulation_method.xc_functional_names = xc_names
            if jacobs_ladder is None:
                jacobs_ladder = _jacobs_from_xc_names(xc_names)

            if jacobs_ladder in _enum_values(DFT, 'jacobs_ladder'):
                simulation_method.jacobs_ladder = jacobs_ladder
                simulation_method.xc_functional_type = jacobs_ladder
            elif jacobs_ladder == 'hybrid-GGA':
                simulation_method.jacobs_ladder = 'hybrid'
                simulation_method.xc_functional_type = 'hybrid'
            elif jacobs_ladder == 'hybrid-meta-GGA':
                simulation_method.jacobs_ladder = 'hyper-GGA'
                simulation_method.xc_functional_type = 'hyper-GGA'

            basis_set_type = _basis_set_type_from_model_method(
                model_method, program_name
            )
            if basis_set_type in _enum_values(DFT, 'basis_set_type'):
                simulation_method.basis_set_type = basis_set_type

            core_treatment = _core_electron_treatment_from_model_method(
                model_method, program_name
            )
            if core_treatment in _enum_values(DFT, 'core_electron_treatment'):
                simulation_method.core_electron_treatment = core_treatment

            scf_threshold = _scf_threshold_from_model_method(model_method)
            if scf_threshold is not None:
                simulation_method.scf_threshold_energy_change = (
                    _energy_threshold_for_results(scf_threshold)
                )

            xc = model_method.xc
            exact_exchange = getattr(xc, 'global_exact_exchange', None) if xc else None
            if exact_exchange is not None:
                simulation_method.exact_exchange_mixing_factor = exact_exchange

        def _map_excited_state_starting_point(
            target_method, dft_method: DFT | None
        ) -> None:
            if dft_method is None:
                return
            dft_basis = dft_method.basis_set_type
            if dft_basis in _enum_values(type(target_method), 'basis_set_type'):
                target_method.basis_set_type = dft_basis
            dft_names = dft_method.xc_functional_names
            if dft_names:
                target_method.starting_point_names = dft_names
            dft_xc_type = dft_method.xc_functional_type
            if dft_xc_type in _enum_values(type(target_method), 'starting_point_type'):
                target_method.starting_point_type = dft_xc_type

        data = archive.data
        try:
            model_methods = data.model_method if data else None
        except Exception:
            model_methods = None
        if not model_methods:
            return

        results = archive.results
        method = results.method
        if method is None:
            method = results.m_create(Method)

        if archive.workflow2:
            method.workflow_name = (
                archive.workflow2.name
                if archive.workflow2.name
                else archive.workflow2.m_def.name
            )

        simulation = method.simulation
        if simulation is None:
            simulation = method.m_create(Simulation)

        try:
            program = data.program
        except Exception:
            program = None
        if program:
            simulation.program_name = program.name
            simulation.program_version = program.version
            simulation.program_version_internal = program.version_internal

        method_name_enum = set(Method.m_def.all_quantities['method_name'].type)
        method_tokens = []
        tb_method_subtypes = {'Wannier', 'DFTB', 'xTB', 'SlaterKoster'}
        for model_method in model_methods:
            section_method_type = getattr(
                getattr(model_method, 'm_def', None), 'name', None
            )
            name_method_type = model_method.name
            method_type = section_method_type
            if method_type in tb_method_subtypes:
                method_type = 'TB'
            if (
                method_type not in method_name_enum
                and name_method_type in method_name_enum
            ):
                method_type = name_method_type
            if not method_type:
                continue
            if method_type not in method_name_enum:
                continue
            method_tokens.append(method_type)

            if method_type == 'DFT' and simulation.dft is None:
                simulation.dft = DFT()
                _map_dft_fields(model_method, simulation.dft, simulation.program_name)
            elif method_type == 'DFT':
                _map_dft_fields(model_method, simulation.dft, simulation.program_name)
            elif method_type == 'TB':
                if simulation.tb is None:
                    simulation.tb = TB()
                tb_type = model_method.type
                tb_type_enum = _enum_values(TB, 'type')
                if tb_type not in tb_type_enum and section_method_type in tb_type_enum:
                    tb_type = section_method_type
                if tb_type is not None:
                    _set_if_enum(simulation.tb, 'type', tb_type)
                _set_if_enum(
                    simulation.tb,
                    'localization_type',
                    model_method.localization_type,
                )
            elif method_type == 'GW' and simulation.gw is None:
                simulation.gw = GW()
                _set_if_enum(simulation.gw, 'type', model_method.type)
                gw_basis = _basis_set_type_from_model_method(
                    model_method, simulation.program_name
                )
                _set_if_enum(simulation.gw, 'basis_set_type', gw_basis)
            elif method_type == 'BSE' and simulation.bse is None:
                simulation.bse = BSE()
                _set_if_enum(simulation.bse, 'type', model_method.type)
                _set_if_enum(simulation.bse, 'solver', model_method.solver)
                bse_basis = _basis_set_type_from_model_method(
                    model_method, simulation.program_name
                )
                _set_if_enum(simulation.bse, 'basis_set_type', bse_basis)
            elif method_type == 'DMFT' and simulation.dmft is None:
                simulation.dmft = DMFT()
                _set_if_enum(
                    simulation.dmft,
                    'impurity_solver_type',
                    model_method.impurity_solver,
                )
                _set_if_enum(
                    simulation.dmft,
                    'magnetic_state',
                    model_method.magnetic_state,
                )
                if model_method.inverse_temperature is not None:
                    simulation.dmft.inverse_temperature = (
                        model_method.inverse_temperature
                    )

        if method_tokens:
            # NOTE(migration): results.method.method_name is currently a single
            # enum value in nomad-FAIR results schema. For data-schema multi-method
            # inputs, we intentionally use the first supported model_method as
            # the canonical method_name for backward-compatible search behavior.
            # The full multi-method design is deferred to results/search redesign.
            method.method_name = method_tokens[0]
            if len(method_tokens) > 1:
                self.logger.warning(
                    'multiple data-schema model_method sections present; using first '
                    'supported method_name for results compatibility',
                    chosen_method=method_tokens[0],
                    available_methods=method_tokens,
                )

        # Legacy-equivalent behavior for excited-state methods: when a DFT method
        # is present in the same entry, carry over starting-point metadata.
        if simulation.gw is not None:
            _map_excited_state_starting_point(simulation.gw, simulation.dft)
        if simulation.bse is not None:
            _map_excited_state_starting_point(simulation.bse, simulation.dft)

    @staticmethod
    def _method_label(model_method_ref=None) -> str | None:
        """Return scientific method provenance without an ownership marker."""
        return getattr(
            getattr(model_method_ref, 'm_def', None), 'name', None
        ) or getattr(
            model_method_ref,
            'name',
            None,
        )

    @staticmethod
    def _mark_generated_compatibility_section(section):
        """Mark a generated section without changing user-facing quantities."""
        section.m_annotations[DATA_SCHEMA_COMPATIBILITY_ANNOTATION] = {
            'generated': True
        }
        return section

    @staticmethod
    def _is_generated_compatibility_section(section) -> bool:
        """Whether this normalizer generated the section on an earlier pass.

        The marker is an annotation rather than a quantity, so nothing the
        parser owns is inspected and no user-facing field is claimed.
        """
        try:
            return any(
                annotation in section.m_annotations
                for annotation in _GENERATED_COMPATIBILITY_ANNOTATIONS
            )
        except Exception:
            return False

    def _remove_generated_compatibility_results(self, properties: Properties) -> None:
        """Remove only compatibility sections created by this normalizer."""
        electronic = properties.electronic
        if electronic is not None:
            for name in (
                'band_gap',
                'dos_electronic',
                'band_structure_electronic',
                'greens_functions_electronic',
            ):
                sections = getattr(electronic, name, None) or []
                setattr(
                    electronic,
                    name,
                    [
                        section
                        for section in sections
                        if not self._is_generated_compatibility_section(section)
                    ],
                )

        spectroscopic = properties.spectroscopic
        if spectroscopic is not None:
            spectroscopic.spectra = [
                section
                for section in spectroscopic.spectra or []
                if not self._is_generated_compatibility_section(section)
            ]

        structural = properties.structural
        if structural is not None:
            structural.radius_of_gyration = [
                section
                for section in structural.radius_of_gyration or []
                if not self._is_generated_compatibility_section(section)
            ]

        thermodynamic = properties.thermodynamic
        if thermodynamic is not None:
            thermodynamic.trajectory = [
                section
                for section in thermodynamic.trajectory or []
                if not self._is_generated_compatibility_section(section)
            ]

        geometry_optimization = properties.geometry_optimization
        if self._is_generated_compatibility_section(geometry_optimization):
            properties.geometry_optimization = None

    def _map_spectrum(self, spectrum_section, spectrum_type: str) -> Spectra | None:
        energies_section = spectrum_section.energies
        energies = energies_section.points if energies_section is not None else None
        intensities = spectrum_section.value
        if intensities is None:
            return None
        intensities_array, intensities_units = self._array_and_units(
            intensities, default_units='arbitrary'
        )
        if not valid_array(intensities_array):
            return None
        if energies is None or not valid_array(energies):
            return None

        spectra = Spectra()
        spectra.type = spectrum_type
        spectra.label = 'computation'
        spectra.n_energies = len(energies)
        spectra.energies = energies
        spectra.intensities = intensities_array
        spectra.intensities_units = intensities_units
        return spectra

    def _map_radius_of_gyration(self, rg_section) -> RadiusOfGyration | None:
        value = rg_section.value
        if value is None:
            return None
        rg = RadiusOfGyration()
        rg.value = value
        rg.label = rg_section.name or 'radius_of_gyration'
        return rg

    @staticmethod
    def _array_and_units(value, default_units: str = '') -> tuple[np.ndarray, str]:
        if hasattr(value, 'magnitude'):
            return np.array(value.magnitude), str(value.u)
        return np.array(value), default_units

    def _log_unmapped_output_groups(self, outputs) -> None:
        """Log unmapped results groups when potentially relevant outputs are present."""
        group_sources = {
            'electronic': [
                'crystal_field_splittings',
                'electronic_eigenvalues',
                'fermi_surfaces',
                'hopping_matrices',
                'kinetic_energies',
                'permittivities',
            ],
            'magnetic': [],
            'vibrational': [],
            'mechanical': ['total_forces'],
            'dynamical': [],
        }
        counts = {group: 0 for group in group_sources}
        for output in outputs:
            for group, fields in group_sources.items():
                for field in fields:
                    counts[group] += len(getattr(output, field, []) or [])

        present_counts = {k: v for k, v in counts.items() if v > 0}
        if present_counts:
            self.logger.info(
                'TODO outputs mapping unsupported groups present',
                counts=present_counts,
            )

    def _normalize_outputs_with_data_schema(self, archive: EntryArchive) -> None:
        """Map nomad-simulations outputs into results.properties."""

        def _is_valid_legacy_dos_entry(entry) -> bool:
            if entry is None:
                return False
            if isinstance(entry, dict):
                energies = entry.get('energies')
                totals = entry.get('total')
                if not isinstance(energies, str) or not energies:
                    return False
                if not isinstance(totals, (list, tuple)) or not totals:
                    return False
                if any(
                    not isinstance(total_ref, str) or not total_ref
                    for total_ref in totals
                ):
                    return False
                return True

            # Accessing deprecated DOS refs can trigger proxy resolution; if that
            # fails we treat the entry as malformed instead of crashing normalization.
            try:
                energies = getattr(entry, 'energies', None)
                totals = getattr(entry, 'total', None)
            except Exception:
                return False

            if energies is None or (isinstance(energies, str) and not energies):
                return False
            if not totals:
                return False
            return True

        data = archive.data
        properties = archive.results.properties
        if properties is None:
            properties = archive.results.m_create(Properties)
        else:
            # Guard against malformed stale deprecated DOS entries that can make
            # GUI DOS resolver crash (reference.energies undefined).
            electronic_existing = properties.electronic
            if electronic_existing is not None:
                existing_dos = electronic_existing.dos_electronic or []
                if existing_dos:
                    electronic_existing.dos_electronic = [
                        dos for dos in existing_dos if _is_valid_legacy_dos_entry(dos)
                    ]

        self._remove_generated_compatibility_results(properties)
        # Geometry optimization workflow metadata is consumed directly by the
        # geometry optimization card and should be available even when there is
        # no electronic/spectroscopic payload in outputs.
        geometry_optimization = self.geometry_optimization()
        if geometry_optimization is not None:
            self._mark_generated_compatibility_section(geometry_optimization)
            properties.geometry_optimization = geometry_optimization

        try:
            outputs = data.outputs if data else None
        except Exception:
            outputs = None
        if not outputs:
            return

        # Deliberately skip data-schema properties whose current results fields
        # require references into legacy run/calculation sections. Those should
        # come from legacy parsers until results can visualize them directly
        # from archive.data:
        # - Outputs.electronic_dos -> properties.electronic.dos_electronic
        # - Outputs.electronic_band_structures
        #   -> properties.electronic.band_structure_electronic
        # - Outputs.electronic_greens_functions, electronic_self_energies,
        #   hybridization_functions, quasiparticle_weights, chemical_potentials
        #   -> properties.electronic.greens_functions_electronic
        # Electronic properties are selected as one coherent source group. Outputs
        # with different explicit system/method references must not be combined.
        latest_band_gaps: list[BandGap] = []
        electronic_groups: dict[tuple[int | None, int | None], dict[str, Any]] = {}
        spectra_sections: list[Spectra] = []
        rg_sections: list[RadiusOfGyration] = []
        temperature_series: list[float] = []
        temperature_time: list[float] = []
        potential_energy_series: list[float] = []
        potential_energy_time: list[float] = []
        outputs_dropped_without_time = 0

        representative_system = None
        try:
            model_systems = data.model_system or []
        except Exception:
            model_systems = []
        try:
            representative_index = data.representative_index
        except Exception:
            try:
                representative_index = data.representative_system_index
            except Exception:
                representative_index = None
        if (
            isinstance(representative_index, int)
            and representative_index >= 0
            and representative_index < len(model_systems)
        ):
            representative_system = model_systems[representative_index]
        elif model_systems:
            representative_system = next(
                (system for system in model_systems if system.is_representative),
                None,
            )

        for index, output in enumerate(outputs):
            output_band_gaps: list[BandGap] = []
            output_system_ref = output.model_system_ref
            output_method_ref = output.model_method_ref
            method_label = self._method_label(output_method_ref)

            for bg in output.electronic_band_gaps or []:
                if bg.value is None:
                    continue
                bg_result = BandGap()
                bg_result.value = bg.value
                bg_result.type = bg.type
                output_band_gaps.append(bg_result)

            if output_band_gaps:
                group_key = (
                    id(output_system_ref) if output_system_ref is not None else None,
                    id(output_method_ref) if output_method_ref is not None else None,
                )
                group = electronic_groups.setdefault(
                    group_key,
                    {
                        'model_system_ref': output_system_ref,
                        'model_method_ref': output_method_ref,
                        'last_index': index,
                        'band_gaps': [],
                    },
                )
                group['last_index'] = index
                if output_band_gaps:
                    group['band_gaps'] = output_band_gaps

            for absorption in output.absorption_spectra or []:
                mapped_spectrum = self._map_spectrum(absorption, 'unavailable')
                if mapped_spectrum and method_label:
                    mapped_spectrum.provenance = SpectraProvenance(label=method_label)
                if mapped_spectrum:
                    spectra_sections.append(mapped_spectrum)
            for xas in output.xas_spectra or []:
                mapped_spectrum = self._map_spectrum(xas, 'XAS')
                if mapped_spectrum and method_label:
                    mapped_spectrum.provenance = SpectraProvenance(label=method_label)
                if mapped_spectrum:
                    spectra_sections.append(mapped_spectrum)

            for rg in output.radii_of_gyration or []:
                mapped_rg = self._map_radius_of_gyration(rg)
                if mapped_rg and method_label:
                    mapped_rg.provenance = MDProvenance(label=method_label)
                if mapped_rg:
                    rg_sections.append(mapped_rg)

            point_time = getattr(output, 'time', None)
            if point_time is not None:
                try:
                    point_time = float(point_time.to('second').magnitude)
                except Exception:
                    self.logger.warning(
                        'skipping trajectory point with invalid physical time',
                        output_index=index,
                    )
                    point_time = None

            temperatures = output.temperatures or []
            if (
                point_time is not None
                and temperatures
                and temperatures[0].value is not None
            ):
                temperature_series.append(float(temperatures[0].value.magnitude))
                temperature_time.append(point_time)

            potential_energies = output.potential_energies or []
            total_energies = output.total_energies or []
            energy_source = (
                potential_energies[0]
                if potential_energies
                else total_energies[0]
                if total_energies
                else None
            )
            if (
                point_time is not None
                and energy_source is not None
                and energy_source.value is not None
            ):
                potential_energy_series.append(float(energy_source.value.magnitude))
                potential_energy_time.append(point_time)

            if point_time is None and (temperatures or energy_source is not None):
                outputs_dropped_without_time += 1

        if outputs_dropped_without_time:
            # `time` only exists on `TrajectoryOutputs`. A trajectory axis cannot be
            # invented from the output index, so the series is dropped rather than
            # plotted against a fabricated time - but never silently.
            self.logger.warning(
                'skipping trajectory series without physical time; '
                'temperature/energy outputs are only mapped for TrajectoryOutputs '
                'carrying `time`',
                n_outputs=outputs_dropped_without_time,
            )

        selected_groups: list[dict[str, Any]] = []
        if electronic_groups:
            # Outputs describing different systems must not be combined. Different
            # methods on the same system are a different matter: legacy
            # `get_gw_workflow_properties` publishes DFT and GW results side by
            # side, so keep one labelled section per method instead of one overall.
            def _system_key(group: dict[str, Any]) -> int | None:
                system_ref = group['model_system_ref']
                return id(system_ref) if system_ref is not None else None

            candidate_groups = [
                group
                for group in electronic_groups.values()
                if group['model_system_ref'] is None
                or (
                    representative_system is not None
                    and group['model_system_ref'] is representative_system
                )
            ]
            if not candidate_groups:
                # Nothing points at the representative system, so fall back to the
                # single most recent system rather than mixing systems.
                newest_group = max(
                    electronic_groups.values(), key=lambda group: group['last_index']
                )
                newest_system_key = _system_key(newest_group)
                candidate_groups = [
                    group
                    for group in electronic_groups.values()
                    if _system_key(group) == newest_system_key
                ]

            # One section per method, in output order; a repeated method keeps its
            # most recent outputs.
            groups_by_method: dict[str | None, dict[str, Any]] = {}
            for group in sorted(candidate_groups, key=lambda g: g['last_index']):
                groups_by_method[self._method_label(group['model_method_ref'])] = group
            selected_groups = list(groups_by_method.values())

            for group in selected_groups:
                latest_band_gaps.extend(group['band_gaps'])

            discarded_groups = len(electronic_groups) - len(selected_groups)
            if discarded_groups:
                self.logger.warning(
                    'discarding electronic outputs from non-representative systems',
                    discarded_groups=discarded_groups,
                )

        if not (
            latest_band_gaps
            or spectra_sections
            or rg_sections
            or temperature_series
            or potential_energy_series
        ):
            return

        if latest_band_gaps:
            electronic = properties.electronic
            if electronic is None:
                electronic = properties.m_create(ElectronicProperties)

            for band_gap in latest_band_gaps:
                self._mark_generated_compatibility_section(band_gap)
                electronic.m_add_sub_section(ElectronicProperties.band_gap, band_gap)

        if spectra_sections:
            spectroscopic = properties.spectroscopic
            if spectroscopic is None:
                spectroscopic = properties.m_create(SpectroscopicProperties)
            for spectrum in spectra_sections:
                self._mark_generated_compatibility_section(spectrum)
                spectroscopic.m_add_sub_section(
                    SpectroscopicProperties.spectra, spectrum
                )

        if rg_sections:
            structural = properties.structural
            if structural is None:
                structural = properties.m_create(StructuralProperties)
            for rg in rg_sections:
                self._mark_generated_compatibility_section(rg)
                structural.m_add_sub_section(
                    StructuralProperties.radius_of_gyration, rg
                )

        if temperature_series or potential_energy_series:
            thermodynamic = properties.thermodynamic
            if thermodynamic is None:
                thermodynamic = properties.m_create(ThermodynamicProperties)
            trajectory = Trajectory()
            self._mark_generated_compatibility_section(trajectory)
            available_properties: list[str] = []
            if temperature_series:
                trajectory.temperature = TemperatureDynamic(
                    value=temperature_series, time=temperature_time
                )
                available_properties.append('temperature')
            if potential_energy_series:
                trajectory.energy_potential = EnergyDynamic(
                    value=potential_energy_series, time=potential_energy_time
                )
                available_properties.append('energy_potential')
            if available_properties:
                trajectory.available_properties = available_properties
                thermodynamic.m_add_sub_section(
                    ThermodynamicProperties.trajectory, trajectory
                )

        self._log_unmapped_output_groups(outputs)

    def normalize_sample(self, sample) -> None:
        material = self.entry_archive.m_setdefault('results.material')

        if sample.elements and len(sample.elements) > 0:
            material.elements = sample.elements
        # Try to guess elements from sample formula or name
        elif sample.chemical_formula:
            try:
                material.elements = list(
                    set(ase.Atoms(sample.chemical_formula).get_chemical_symbols())
                )
            except Exception:
                if sample.name:
                    try:
                        material.elements = list(
                            set(ase.Atoms(sample.name).get_chemical_symbols())
                        )
                    except Exception:
                        pass
        if sample.chemical_formula:
            material.chemical_formula_descriptive = sample.chemical_formula

        try:
            if material.chemical_formula_descriptive:
                formula = Formula(material.chemical_formula_descriptive)
                if not material.elements:
                    material.elements = formula.elements()
                material.elemental_composition = formula.elemental_composition()
                material.chemical_formula_hill = formula.format('hill')
                material.chemical_formula_reduced = formula.format('reduced')
                material.chemical_formula_iupac = formula.format('iupac')
                material.chemical_formula_descriptive = formula.format('descriptive')
        except Exception as e:
            self.logger.warn('could not normalize material', exc_info=e)

    def normalize_measurement(self, measurement) -> None:
        results = self.entry_archive.results

        # Method
        if results.method is None:
            results.method = Method(method_name=measurement.method_abbreviation)

        # Sample
        if results.material is None:
            results.material = Material(elements=[])
        if len(measurement.sample) > 0:
            self.normalize_sample(measurement.sample[0])

        # Results properties for EELSDB
        if measurement.m_xpath('eels.spectrum'):
            properties = results.properties
            spectroscopic = properties.m_create(SpectroscopicProperties)

            spectra = Spectra(
                type='EELS',
                label='experiment',
                n_energies=measurement.eels.spectrum.n_values,
                energies=measurement.eels.spectrum.energy,
                intensities=measurement.eels.spectrum.count,
                intensities_units='counts',
            )
            if measurement.instrument:
                provenance = spectra.m_create(SpectraProvenance)
                provenance.label = 'EELSDB'
                methodology = EELSMethodology(
                    resolution=measurement.instrument[0].eels.resolution,
                    detector_type=measurement.instrument[0].eels.detector_type,
                    min_energy=measurement.instrument[0].eels.min_energy,
                    max_energy=measurement.instrument[0].eels.max_energy,
                )
                provenance.m_add_sub_section(SpectraProvenance.eels, methodology)
            spectroscopic.m_add_sub_section(SpectroscopicProperties.spectra, spectra)

    def geometry_optimization(self) -> GeometryOptimization | None:
        """Populates both geometry optimization methodology and calculated
        properties based on the first found geometry optimization workflow.
        """
        path = ['workflow2']
        for workflow in traverse_reversed(self.entry_archive, path):
            workflow_sub_sections = workflow.m_def.all_sub_sections
            method = workflow.method if 'method' in workflow_sub_sections else None
            results = workflow.results if 'results' in workflow_sub_sections else None

            optimization_type_value = None
            if method is not None:
                method_quantities = method.m_def.all_quantities
                if 'optimization_type' in method_quantities:
                    optimization_type_value = method.optimization_type
                elif 'type' in method_quantities:
                    optimization_type_value = method.type

            has_geo_method = optimization_type_value is not None
            has_geo_results = False
            if results is not None:
                results_quantities = results.m_def.all_quantities
                has_geo_results = (
                    (
                        'final_energy_difference' in results_quantities
                        and results.final_energy_difference is not None
                    )
                    or (
                        'final_force_maximum' in results_quantities
                        and results.final_force_maximum is not None
                    )
                    or (
                        'final_displacement_maximum' in results_quantities
                        and results.final_displacement_maximum is not None
                    )
                )

            if (
                workflow.__class__.__name__ == 'GeometryOptimization'
                or has_geo_method
                or has_geo_results
            ):
                geo_opt = GeometryOptimization()

                # KNOWN LIMITATION:
                # Trajectory visualization unavailable for new schema workflows.
                # Both trajectory and system_optimized expect legacy runschema types:
                # - trajectory: runschema.calculation.Calculation
                #   (not nomad-simulations Outputs)
                # - system_optimized: runschema.system.System
                #   (not nomad-simulations ModelSystem)
                #
                # Assigning nomad-simulations sections to these fields raises
                # TypeError.
                # Migration policy decision: Skip compatibility population
                # cleanly rather than
                # mixing schema types or creating complex translation layers.
                #
                # Impact: GUI geometry optimization trajectory graph
                # (energy vs steps) will show "no data" for new schema
                # workflows. Only convergence values are populated.
                #
                # Resolution: GUI must be updated to read from
                # archive.data.outputs directly.
                # See dev_notes/MIGRATION_STATUS.md "Known Limitations" section.

                if results:
                    results_quantities = results.m_def.all_quantities
                    # Legacy workflow schemas expose calculation references
                    if (
                        'calculations_ref' in results_quantities
                        and results.calculations_ref
                    ):
                        geo_opt.trajectory = results.calculations_ref

                    if (
                        'calculation_result_ref' in results_quantities
                        and results.calculation_result_ref
                    ):
                        if not geo_opt.system_optimized:
                            geo_opt.system_optimized = (
                                results.calculation_result_ref.system_ref
                            )

                    if 'final_energy_difference' in results_quantities:
                        final_energy_difference = results.final_energy_difference
                        if final_energy_difference is not None:
                            geo_opt.final_energy_difference = final_energy_difference

                    if 'final_force_maximum' in results_quantities:
                        final_force_maximum = results.final_force_maximum
                        if final_force_maximum is not None:
                            geo_opt.final_force_maximum = final_force_maximum

                    if 'final_displacement_maximum' in results_quantities:
                        final_displacement_maximum = results.final_displacement_maximum
                        if final_displacement_maximum is not None:
                            geo_opt.final_displacement_maximum = (
                                final_displacement_maximum
                            )
                if method is not None:
                    method_quantities = method.m_def.all_quantities
                    method_sub_sections = method.m_def.all_sub_sections
                    optimization_type = optimization_type_value
                    if optimization_type is not None:
                        geo_opt.type = optimization_type

                    energy_tolerance = None
                    if 'convergence_tolerance_energy_difference' in method_quantities:
                        energy_tolerance = (
                            method.convergence_tolerance_energy_difference
                        )
                    elif hasattr(method, 'convergence_tolerance_energy_difference'):
                        energy_tolerance = (
                            method.convergence_tolerance_energy_difference
                        )

                    force_tolerance = None
                    if 'convergence_tolerance_force_maximum' in method_quantities:
                        force_tolerance = method.convergence_tolerance_force_maximum
                    elif hasattr(method, 'convergence_tolerance_force_maximum'):
                        force_tolerance = method.convergence_tolerance_force_maximum

                    # nomad-simulations workflows store geometry convergence
                    # thresholds under method.convergence_targets.
                    if energy_tolerance is None or force_tolerance is None:
                        for target in (
                            method.convergence_targets
                            if 'convergence_targets' in method_sub_sections
                            else []
                        ) or []:
                            target_name = target.m_def.name
                            if (
                                energy_tolerance is None
                                and target_name == 'EnergyConvergenceTarget'
                            ):
                                energy_tolerance = target.threshold
                            elif (
                                force_tolerance is None
                                and target_name == 'ForceConvergenceTarget'
                            ):
                                force_tolerance = target.threshold

                    if energy_tolerance is not None:
                        geo_opt.convergence_tolerance_energy_difference = (
                            energy_tolerance
                        )
                    if force_tolerance is not None:
                        geo_opt.convergence_tolerance_force_maximum = force_tolerance
                return geo_opt

        return None
