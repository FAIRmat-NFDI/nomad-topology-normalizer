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
from nomad.datamodel.metainfo.simulation.calculation import BandEnergies
from nomad.datamodel.results import (
    BSE,
    DFT,
    DMFT,
    GW,
    TB,
    BandGap,
    BandGapDeprecated,
    BandStructureElectronic,
    DOSElectronic,
    EELSMethodology,
    ElectronicProperties,
    EnergyDynamic,
    GeometryOptimization,
    GreensFunctionsElectronic,
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

# Import runschema for DOS compatibility mapping
try:
    import runschema.calculation
    import runschema.run
except ImportError:
    runschema = None

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

    def _map_dos_data(
        self, dos_section, output_index: int, dos_index: int
    ) -> dict | None:
        """Map one ElectronicDensityOfStates into DOSElectronic-compatible refs."""
        energies = dos_section.energies
        energies_points = energies.points if energies is not None else None
        values = dos_section.value
        if not valid_array(energies_points) or values is None:
            return None
        try:
            if not valid_array(np.array(values.magnitude)):
                return None
        except Exception:
            return None

        spin_ch = dos_section.spin_channel or 0
        has_projected = bool(dos_section.projected_dos)
        dos_base_ref = f'/data/outputs/{output_index}/electronic_dos/{dos_index}'
        return {
            'energies_ref': f'{dos_base_ref}/energies/points',
            'total_ref': dos_base_ref,
            'spin_channel': spin_ch,
            'has_projected': has_projected,
            'energies_points': energies_points,
            'values': values,
        }

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

    def _ensure_legacy_run_calculation(
        self, archive: EntryArchive
    ) -> tuple[Any, int, int] | None:
        """Return a dedicated, normalizer-owned compatibility calculation."""
        if not runschema:
            return None

        runs = archive.run or []
        run = next(
            (
                candidate
                for candidate in runs
                if self._is_generated_compatibility_section(candidate)
            ),
            None,
        )
        if run is None:
            run = runschema.run.Run()
            archive.run.append(run)
        self._mark_generated_compatibility_section(run)

        calculations = run.calculation
        if calculations and len(calculations) > 0:
            calculation = calculations[0]
        else:
            calculation = runschema.calculation.Calculation()
            run.calculation.append(calculation)
        self._mark_generated_compatibility_section(calculation)

        return (
            calculation,
            list(archive.run).index(run),
            list(run.calculation).index(calculation),
        )

    def _map_band_structure(
        self, band_structure_section
    ) -> BandStructureElectronic | None:
        """Map one ElectronicBandStructure into results BandStructureElectronic."""
        values = band_structure_section.value
        if values is None:
            return None
        try:
            if not valid_array(np.array(values.magnitude)):
                return None
        except Exception:
            return None

        energies_array = np.array(values.magnitude)
        if energies_array.ndim == 0:
            energies_array = energies_array[np.newaxis, np.newaxis, np.newaxis]
        elif energies_array.ndim == 1:
            energies_array = energies_array[np.newaxis, :, np.newaxis]
        elif energies_array.ndim == 2:
            energies_array = energies_array[np.newaxis, :, :]

        segment_cls = BandEnergies
        if runschema and hasattr(runschema, 'calculation'):
            segment_cls = runschema.calculation.BandEnergies
        legacy_segment = segment_cls()
        legacy_segment.energies = energies_array * values.u

        k_path = band_structure_section.k_path
        k_points = k_path.points if k_path is not None else None
        if not valid_array(k_points):
            # Fallback for parsers that only store band-path information in
            # model method numerical settings.
            try:
                numerical_settings = band_structure_section.m_xpath(
                    'm_parent.m_parent.model_method[-1].numerical_settings',
                    dict=False,
                )
            except Exception:
                numerical_settings = None
            for setting in numerical_settings or []:
                k_line_path = getattr(setting, 'k_line_path', None)
                k_points = k_line_path.points if k_line_path is not None else None
                if not valid_array(k_points):
                    # Some parsers expose only vertex path values without the
                    # expanded line-point list.
                    k_points = (
                        k_line_path.high_symmetry_path_values
                        if k_line_path is not None
                        else None
                    )
                if valid_array(k_points):
                    break
        if not valid_array(k_points):
            # Legacy/GUI path requires kpoints for distance axis construction.
            return None

        # A vertex-only or otherwise incomplete path cannot be reconstructed
        # without knowing the parser's segment sampling and discontinuities.
        n_energy_kpoints = int(energies_array.shape[1])
        if np.array(k_points).shape[0] != n_energy_kpoints:
            self.logger.warning(
                'skipping band structure with mismatched k-point and energy axes',
                n_kpoints=int(np.array(k_points).shape[0]),
                n_energy_kpoints=n_energy_kpoints,
            )
            return None
        legacy_segment.kpoints = k_points

        reciprocal_cell = band_structure_section.reciprocal_cell

        band_structure = BandStructureElectronic()
        band_structure.segment = [legacy_segment]
        try:
            if reciprocal_cell is not None:
                band_structure.reciprocal_cell = reciprocal_cell
        except Exception:
            pass

        band_structure.spin_polarized = energies_array.shape[0] == 2

        highest_occupied = band_structure_section.highest_occupied
        lowest_unoccupied = band_structure_section.lowest_unoccupied
        if highest_occupied is not None and lowest_unoccupied is not None:
            band_gap = BandGapDeprecated()
            band_gap.energy_highest_occupied = highest_occupied
            band_gap.energy_lowest_unoccupied = lowest_unoccupied
            band_gap.value = max(0.0, (lowest_unoccupied - highest_occupied).magnitude)
            band_structure.m_add_sub_section(BandStructureElectronic.band_gap, band_gap)
        return band_structure

    def _map_greens_functions(self, output_section) -> GreensFunctionsElectronic | None:
        """Map Green's function outputs into GreensFunctionsElectronic."""

        def _to_array_quantity(value):
            try:
                array = np.array(value.magnitude)
            except Exception:
                return value
            # TODO(results-compat): Legacy Greens fields have inconsistent/different
            # shape+dtype expectations vs nomad-simulations complex payloads.
            # Keep axis/metadata transfer for now and skip complex payload values.
            if np.iscomplexobj(array):
                return None
            if array.ndim == 0:
                return np.array([[array.item()]]) * value.u
            return value

        def _safe_set(section, name: str, value) -> bool:
            if name not in available_fields:
                return False
            try:
                setattr(section, name, value)
                return True
            except Exception:
                self.logger.warning('skipping incompatible greens field', field=name)
                return False

        def _safe_set_axis_payload(
            axis_name: str, axis_value, payload_name: str, payload_value
        ) -> bool:
            """Set an axis only when its corresponding physical payload is valid."""
            converted_payload = _to_array_quantity(payload_value)
            if converted_payload is None:
                return False
            if (
                axis_name not in available_fields
                or payload_name not in available_fields
            ):
                return False
            try:
                setattr(legacy_gf, payload_name, converted_payload)
                setattr(legacy_gf, axis_name, axis_value)
                return True
            except Exception:
                # MSection has no m_unset API. Passing None to m_set is the
                # supported overwrite-mode mechanism for clearing a property.
                legacy_gf.m_set(payload_name, None)
                legacy_gf.m_set(axis_name, None)
                self.logger.warning(
                    'skipping incompatible greens axis/payload pair',
                    axis=axis_name,
                    payload=payload_name,
                )
                return False

        # The legacy Green's-function payload quantities were removed from the
        # current results schema. Skip this compatibility mapping when they are
        # unavailable instead of failing every output normalization.
        tau_quantity = GreensFunctionsElectronic.m_def.all_quantities.get('tau')
        if tau_quantity is None:
            return None
        gf_type = tau_quantity.type
        gf_cls = gf_type.target_quantity_def.m_parent.section_cls
        available_fields = set(gf_cls.m_def.all_quantities.keys())
        legacy_gf = gf_cls()
        payload_mapped = False

        for greens in output_section.electronic_greens_functions or []:
            if (
                greens.imaginary_time is not None
                and valid_array(greens.imaginary_time.points)
                and greens.value is not None
            ):
                payload_mapped = (
                    _safe_set_axis_payload(
                        'tau',
                        greens.imaginary_time.points,
                        'greens_function_tau',
                        greens.value,
                    )
                    or payload_mapped
                )
            if (
                greens.matsubara_frequency is not None
                and valid_array(greens.matsubara_frequency.points)
                and greens.value is not None
            ):
                matsubara_points = greens.matsubara_frequency.points
                if np.iscomplexobj(np.array(matsubara_points.magnitude)):
                    self.logger.warning(
                        'skipping complex matsubara greens payload in results mapping'
                    )
                    continue
                payload_mapped = (
                    _safe_set_axis_payload(
                        'matsubara_freq',
                        matsubara_points,
                        'greens_function_iw',
                        greens.value,
                    )
                    or payload_mapped
                )
            if (
                greens.real_frequency is not None
                and valid_array(greens.real_frequency.points)
                and greens.value is not None
            ):
                payload_mapped = (
                    _safe_set_axis_payload(
                        'frequencies',
                        greens.real_frequency.points,
                        'greens_function_freq',
                        greens.value,
                    )
                    or payload_mapped
                )

        for self_energy in output_section.electronic_self_energies or []:
            if (
                self_energy.matsubara_frequency is not None
                and valid_array(self_energy.matsubara_frequency.points)
                and self_energy.value is not None
            ):
                matsubara_points = self_energy.matsubara_frequency.points
                if np.iscomplexobj(np.array(matsubara_points.magnitude)):
                    continue
                payload_mapped = (
                    _safe_set_axis_payload(
                        'matsubara_freq',
                        matsubara_points,
                        'self_energy_iw',
                        self_energy.value,
                    )
                    or payload_mapped
                )
            if (
                self_energy.real_frequency is not None
                and valid_array(self_energy.real_frequency.points)
                and self_energy.value is not None
            ):
                payload_mapped = (
                    _safe_set_axis_payload(
                        'frequencies',
                        self_energy.real_frequency.points,
                        'self_energy_freq',
                        self_energy.value,
                    )
                    or payload_mapped
                )

        for hybridization in output_section.hybridization_functions or []:
            if (
                hybridization.matsubara_frequency is not None
                and valid_array(hybridization.matsubara_frequency.points)
                and hybridization.value is not None
            ):
                matsubara_points = hybridization.matsubara_frequency.points
                if np.iscomplexobj(np.array(matsubara_points.magnitude)):
                    continue
                if 'hybridization_function_iw' in available_fields:
                    payload_mapped = (
                        _safe_set_axis_payload(
                            'matsubara_freq',
                            matsubara_points,
                            'hybridization_function_iw',
                            hybridization.value,
                        )
                        or payload_mapped
                    )
            if (
                hybridization.real_frequency is not None
                and valid_array(hybridization.real_frequency.points)
                and hybridization.value is not None
            ):
                payload_mapped = (
                    _safe_set_axis_payload(
                        'frequencies',
                        hybridization.real_frequency.points,
                        'hybridization_function_freq',
                        hybridization.value,
                    )
                    or payload_mapped
                )

        for qp_weight in output_section.quasiparticle_weights or []:
            if qp_weight.value is not None and valid_array(np.array(qp_weight.value)):
                payload_mapped = (
                    _safe_set(legacy_gf, 'quasiparticle_weights', qp_weight.value)
                    or payload_mapped
                )

        chemical_potentials = output_section.chemical_potentials or []
        if chemical_potentials and chemical_potentials[0].value is not None:
            payload_mapped = (
                _safe_set(legacy_gf, 'chemical_potential', chemical_potentials[0].value)
                or payload_mapped
            )

        if not payload_mapped:
            return None

        greens_functions = GreensFunctionsElectronic()
        if legacy_gf.m_is_set('tau'):
            greens_functions.tau = legacy_gf
        if legacy_gf.m_is_set('matsubara_freq'):
            greens_functions.matsubara_freq = legacy_gf
        if legacy_gf.m_is_set('frequencies'):
            greens_functions.frequencies = legacy_gf
        if legacy_gf.m_is_set('greens_function_tau'):
            greens_functions.greens_function_tau = legacy_gf
        if legacy_gf.m_is_set('greens_function_iw'):
            greens_functions.greens_function_iw = legacy_gf
        if legacy_gf.m_is_set('self_energy_iw'):
            greens_functions.self_energy_iw = legacy_gf
        if legacy_gf.m_is_set('greens_function_freq'):
            greens_functions.greens_function_freq = legacy_gf
        if legacy_gf.m_is_set('self_energy_freq'):
            greens_functions.self_energy_freq = legacy_gf
        if legacy_gf.m_is_set('hybridization_function_freq'):
            greens_functions.hybridization_function_freq = legacy_gf
        if legacy_gf.m_is_set('orbital_occupations'):
            greens_functions.orbital_occupations = legacy_gf
        if legacy_gf.m_is_set('quasiparticle_weights'):
            greens_functions.quasiparticle_weights = legacy_gf
        if legacy_gf.m_is_set('chemical_potential'):
            greens_functions.chemical_potential = legacy_gf
        return greens_functions

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
        if runschema:
            for compatibility_run in archive.run or []:
                if not self._is_generated_compatibility_section(compatibility_run):
                    continue
                for compatibility_calculation in compatibility_run.calculation or []:
                    self._mark_generated_compatibility_section(compatibility_calculation)
                    compatibility_calculation.dos_electronic = []
                    compatibility_calculation.band_structure_electronic = []

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

        had_dos_input = any(len(output.electronic_dos or []) > 0 for output in outputs)

        # Electronic properties are selected as one coherent source group. Outputs
        # with different explicit system/method references must not be combined.
        latest_band_gaps: list[BandGap] = []
        latest_dos_sections: list[DOSElectronic] = []
        latest_band_structures: list[BandStructureElectronic] = []
        latest_greens_functions: list[GreensFunctionsElectronic] = []
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
            output_dos_sections: list[DOSElectronic] = []
            output_band_structures: list[BandStructureElectronic] = []
            output_greens_functions: list[GreensFunctionsElectronic] = []
            output_highest_occupied = None
            output_system_ref = output.model_system_ref
            output_method_ref = output.model_method_ref
            method_label = self._method_label(output_method_ref)

            for band_structure in output.electronic_band_structures or []:
                output_highest_occupied = band_structure.highest_occupied
                if output_highest_occupied is not None:
                    break
            if output_highest_occupied is None:
                for eigenvalues in output.electronic_eigenvalues or []:
                    output_highest_occupied = eigenvalues.highest_occupied
                    if output_highest_occupied is not None:
                        break
            if output_highest_occupied is None:
                for dos in output.electronic_dos or []:
                    output_highest_occupied = dos.energies_origin
                    if output_highest_occupied is not None:
                        break

            for bg in output.electronic_band_gaps or []:
                if bg.value is None:
                    continue
                bg_result = BandGap()
                bg_result.value = bg.value
                bg_result.type = bg.type
                if runschema and method_label:
                    bg_result.provenance = (
                        runschema.calculation.ElectronicStructureProvenance(
                            label=method_label
                        )
                    )
                if output_highest_occupied is not None:
                    bg_result.energy_highest_occupied = output_highest_occupied
                    bg_result.energy_lowest_unoccupied = (
                        output_highest_occupied + bg.value
                    )
                output_band_gaps.append(bg_result)

            dos_data_sections: list[dict] = []
            has_projected = False
            for dos_index, dos in enumerate(output.electronic_dos or []):
                mapped = self._map_dos_data(dos, index, dos_index)
                if mapped is None:
                    continue
                dos_data_sections.append(mapped)
                has_projected = has_projected or mapped['has_projected']

            if dos_data_sections:
                totals = [
                    d['total_ref'] for d in dos_data_sections if d.get('total_ref')
                ]
                if totals:
                    dos_result = DOSElectronic()
                    dos_result.label = method_label
                    dos_result.energies = dos_data_sections[0]['energies_ref']
                    dos_result.total = totals
                    dos_result.spin_polarized = len(dos_data_sections) == 2
                    if output_highest_occupied is not None:
                        dos_result.energy_fermi = output_highest_occupied
                        dos_bg = BandGapDeprecated()
                        dos_bg.energy_highest_occupied = output_highest_occupied
                        dos_result.m_add_sub_section(DOSElectronic.band_gap, dos_bg)
                    output_dos_sections.append(dos_result)

            for band_structure in output.electronic_band_structures or []:
                mapped_band_structure = self._map_band_structure(band_structure)
                if mapped_band_structure:
                    mapped_band_structure.label = method_label
                    output_band_structures.append(mapped_band_structure)

            # Ensure compatibility card consumers can read a complete band-gap
            # entry from band-structure references when output-level band gaps
            # are available but parser band-structure sections do not carry
            # explicit lowest-unoccupied values.
            if output_band_gaps:
                fallback_bg = next(
                    (
                        bg
                        for bg in output_band_gaps
                        if bg.value is not None
                        and bg.energy_highest_occupied is not None
                        and bg.energy_lowest_unoccupied is not None
                    ),
                    None,
                )
                if fallback_bg is not None:
                    for mapped_band_structure in output_band_structures:
                        existing_bg = mapped_band_structure.band_gap or []
                        has_complete_bg = any(
                            bg.value is not None
                            and bg.energy_highest_occupied is not None
                            and bg.energy_lowest_unoccupied is not None
                            for bg in existing_bg
                        )
                        if has_complete_bg:
                            continue
                        mapped_band_structure.m_add_sub_section(
                            BandStructureElectronic.band_gap,
                            BandGapDeprecated().m_from_dict(fallback_bg.m_to_dict()),
                        )

            mapped_greens_functions = self._map_greens_functions(output)
            if mapped_greens_functions:
                mapped_greens_functions.label = method_label
                output_greens_functions.append(mapped_greens_functions)

            if (
                output_band_gaps
                or output_dos_sections
                or output_band_structures
                or output_greens_functions
            ):
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
                        'dos_sections': [],
                        'dos_payload': [],
                        'band_structures': [],
                        'greens_functions': [],
                    },
                )
                group['last_index'] = index
                if output_band_gaps:
                    group['band_gaps'] = output_band_gaps
                if output_dos_sections:
                    group['dos_sections'] = output_dos_sections
                    group['dos_payload'] = dos_data_sections
                if output_band_structures:
                    group['band_structures'] = output_band_structures
                if output_greens_functions:
                    group['greens_functions'] = output_greens_functions

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
                latest_dos_sections.extend(group['dos_sections'])
                latest_band_structures.extend(group['band_structures'])
                latest_greens_functions.extend(group['greens_functions'])

            discarded_groups = len(electronic_groups) - len(selected_groups)
            if discarded_groups:
                self.logger.warning(
                    'discarding electronic outputs from non-representative systems',
                    discarded_groups=discarded_groups,
                )

        if not (
            latest_band_gaps
            or latest_dos_sections
            or latest_band_structures
            or latest_greens_functions
            or spectra_sections
            or rg_sections
            or temperature_series
            or potential_energy_series
        ):
            if had_dos_input:
                dos_warning = (
                    'Skipping DOS mapping for results.properties.electronic.'
                    'dos_electronic: could not build payload from data-schema outputs.'
                )
                self.logger.warning(dos_warning)
            return

        has_electronic_payload = bool(
            latest_band_gaps
            or latest_dos_sections
            or latest_band_structures
            or latest_greens_functions
        )

        # Prefer run/calculation DOS compatibility paths when runschema is available,
        # so legacy GUI resolvers can follow runschema-typed references robustly.
        if runschema and latest_dos_sections:
            compatibility_target = self._ensure_legacy_run_calculation(archive)
            if compatibility_target is not None:
                calculation, run_index, calculation_index = compatibility_target
                calculation.dos_electronic = []
                calculation_path = f'/run/{run_index}/calculation/{calculation_index}'
                # Each method contributes its own legacy Dos sections, so indices
                # run across the shared list instead of restarting per group.
                for group in selected_groups:
                    dos_sections = group['dos_sections']
                    dos_payload = group['dos_payload']
                    if not (dos_sections and dos_payload):
                        continue
                    energies_index = len(calculation.dos_electronic)
                    run_total_refs: list[str] = []
                    for dos_entry in dos_payload:
                        legacy_dos = runschema.calculation.Dos()
                        legacy_dos.energies = dos_entry['energies_points']

                        legacy_total = runschema.calculation.DosValues()
                        legacy_total.value = dos_entry['values']
                        legacy_total.spin = int(dos_entry.get('spin_channel', 0) or 0)
                        legacy_dos.total.append(legacy_total)

                        run_total_refs.append(
                            f'{calculation_path}/dos_electronic'
                            f'/{len(calculation.dos_electronic)}/total/0'
                        )
                        calculation.dos_electronic.append(legacy_dos)

                    dos_sections[
                        0
                    ].energies = (
                        f'{calculation_path}/dos_electronic/{energies_index}/energies'
                    )
                    dos_sections[0].total = run_total_refs

        # `results.properties.electronic.band_structure_electronic.segment` is a
        # reference-typed quantity. For valid references, materialize compatible
        # legacy sections under run/calculation and point results segments there.
        if runschema and latest_band_structures:
            compatibility_target = self._ensure_legacy_run_calculation(archive)
            if compatibility_target is not None:
                calculation, _, _ = compatibility_target
                calculation.band_structure_electronic = []
                run_band_structures: list[BandStructureElectronic] = []
                for band_structure in latest_band_structures:
                    legacy_bs = runschema.calculation.BandStructure()
                    if band_structure.reciprocal_cell is not None:
                        legacy_bs.reciprocal_cell = band_structure.reciprocal_cell
                    if band_structure.energy_fermi is not None:
                        legacy_bs.energy_fermi = band_structure.energy_fermi

                    for segment in band_structure.segment or []:
                        legacy_segment = runschema.calculation.BandEnergies()
                        legacy_segment.energies = segment.energies
                        legacy_segment.kpoints = segment.kpoints
                        legacy_bs.segment.append(legacy_segment)

                    calculation.band_structure_electronic.append(legacy_bs)

                    bs_result = BandStructureElectronic()
                    bs_result.label = band_structure.label
                    bs_result.spin_polarized = band_structure.spin_polarized
                    bs_result.energy_fermi = band_structure.energy_fermi
                    bs_result.reciprocal_cell = band_structure.reciprocal_cell
                    bs_result.segment = legacy_bs.segment
                    for info in band_structure.band_gap or []:
                        info_new = BandGapDeprecated().m_from_dict(info.m_to_dict())
                        bs_result.m_add_sub_section(
                            BandStructureElectronic.band_gap, info_new
                        )
                    run_band_structures.append(bs_result)

                latest_band_structures = run_band_structures

        if has_electronic_payload:
            electronic = properties.electronic
            if electronic is None:
                electronic = properties.m_create(ElectronicProperties)

            for band_gap in latest_band_gaps:
                self._mark_generated_compatibility_section(band_gap)
                electronic.m_add_sub_section(ElectronicProperties.band_gap, band_gap)
            for dos in latest_dos_sections:
                self._mark_generated_compatibility_section(dos)
                electronic.m_add_sub_section(ElectronicProperties.dos_electronic, dos)
            for band_structure in latest_band_structures:
                self._mark_generated_compatibility_section(band_structure)
                electronic.m_add_sub_section(
                    ElectronicProperties.band_structure_electronic, band_structure
                )
            for greens in latest_greens_functions:
                self._mark_generated_compatibility_section(greens)
                electronic.m_add_sub_section(
                    ElectronicProperties.greens_functions_electronic, greens
                )

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

        if had_dos_input and not latest_dos_sections:
            dos_warning = (
                'Skipping DOS mapping for results.properties.electronic.'
                'dos_electronic: could not build payload from data-schema outputs.'
            )
            self.logger.warning(dos_warning)

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
