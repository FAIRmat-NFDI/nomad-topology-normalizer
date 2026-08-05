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
Results Normalizer - Entry Point for v2 Data Schema Normalization
===================================================================

This module provides the main entry point for results normalization with
backward compatibility support for both v2 data schema and legacy schemas.

NORMALIZATION CASCADE ARCHITECTURE
-----------------------------------

Plugin Entry Point: results_normalizer_plugin (level 3)
    │
    └─── Schema Detection: _is_v2_data_schema(archive)
            │
            ├─ Checks: archive.data exists?
            │          archive.data.model_system exists?
            │          Uses basesections.v2.System?
            │
            ├─ v2 Schema → _normalize_with_data_schema()
            │                 │
            │                 ├─ Initialize results sections
            │                 └─ TopologyNormalizer.normalize()
            │                       │
            │                       ├─ MaterialNormalizer (creates material info)
            │                       └─ topology() waterfall:
            │                             ├─ topology_calculation() (parser-defined)
            │                             ├─ topology_matid() (algorithmic)
            │                             └─ topology_data() (fallback)
            │
            └─ Legacy Schema → _normalize_with_legacy()
                                  │
                                  └─ Delegates to
                                     nomad.normalizing.results.ResultsNormalizer
                                        │
                                        └─ Handles: v1 run schema
                                                    Old data schemas
                                                    Any other legacy formats

Schema Version Detection
------------------------

The _is_v2_data_schema() method validates that an entry uses the new v2
basesections.v2 schema by checking:
1. archive.data attribute exists
2. archive.data.model_system attribute exists
3. model_system contains basesections.v2.System instances

All other cases (including v1 run schema, old data schemas, or empty archives)
are delegated to the legacy normalizer which handles them appropriately.

Design Principles
-----------------

- Single entry point: Avoids double execution and cascade ordering issues
- Precise detection: Only v2 basesections.v2 triggers new path
- Automatic fallback: Legacy handles all non-v2 cases without explicit checks
- Clean separation: v2 code stays in plugin, legacy code stays in nomad-FAIR
- No breaking changes: Existing entries continue to work via legacy path
"""

import os
import re
from typing import Any

import ase.data
import matid.geometry  # pylint: disable=import-error
import numpy as np
from matid import SymmetryAnalyzer  # pylint: disable=import-error
from nomad import atomutils
from nomad.atomutils import Formula
from nomad.config import config
from nomad.datamodel import EntryArchive
from nomad.datamodel.data import ArchiveSection
from nomad.datamodel.metainfo.simulation.calculation import BandEnergies
from nomad.datamodel.metainfo.workflow import Workflow
from nomad.datamodel.results import (
    BSE,
    DFT,
    DMFT,
    GW,
    TB,
    BandGap,
    BandGapDeprecated,
    BandStructureElectronic,
    BandStructurePhonon,
    BulkModulus,
    DensityCharge,
    DOSElectronic,
    DOSElectronicNew,
    DOSNew,
    DOSPhonon,
    DynamicalProperties,
    EELSMethodology,
    ElectricFieldGradient,
    ElectronicProperties,
    EnergyDynamic,
    EnergyFreeHelmholtz,
    EnergyVolumeCurve,
    GeometryOptimization,
    GreensFunctionsElectronic,
    HeatCapacityConstantVolume,
    MagneticProperties,
    MagneticShielding,
    MagneticSusceptibility,
    Material,
    MDProvenance,
    MeanSquaredDisplacement,
    MechanicalProperties,
    Method,
    MolecularDynamics,
    PressureDynamic,
    Properties,
    RadialDistributionFunction,
    RadiusOfGyration,
    Results,
    ShearModulus,
    Simulation,
    Spectra,
    SpectraProvenance,
    SpectroscopicProperties,
    SpinSpinCoupling,
    StructuralProperties,
    TemperatureDynamic,
    ThermodynamicProperties,
    Trajectory,
    VibrationalProperties,
    VolumeDynamic,
)
from nomad.units import ureg
from nomad.utils import extract_section, traverse_reversed

# Import runschema for DOS compatibility mapping
try:
    import runschema.calculation
    import runschema.run
except ImportError:
    runschema = None

from nomad_topology_normalizer.normalizers.common import structures_2d
from nomad_topology_normalizer.normalizers.method import MethodNormalizer

# Don't import local Normalizer to avoid circular import
# ResultsNormalizer inherits directly from nomad.normalizing.Normalizer

re_label = re.compile('^([a-zA-Z][a-zA-Z]?)[^a-zA-Z]*')
elements = set(ase.data.chemical_symbols)

V2_COMPATIBILITY_ANNOTATION = 'nomad_topology_normalizer_v2_compatibility'

# Recognize sections written by the first compatibility implementation so that
# one normalization pass migrates them to the invisible annotation marker.
_LEGACY_V2_COMPATIBILITY_LABEL = 'nomad-topology-normalizer:v2-compatibility'
_LEGACY_V2_COMPATIBILITY_RUN_ID = f'{_LEGACY_V2_COMPATIBILITY_LABEL}:run'


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
# nomad_topology_normalizer/normalizers/__init__.py::ResultsNormalizerEntryPoint.load()
class ResultsNormalizerBase:
    """Results normalizer implementation with schema version detection.

    WARNING: This is NOT the actual normalizer class used at runtime!
    The entry point creates a proper subclass dynamically. See class docstring above.

    Strategy:
    1. Check if archive.data exists (v2 schema) → use new normalization cascade
    2. If not, fall back to archive.run (v1 schema) → delegate to legacy normalizer

    This ensures backward compatibility during the transition period.
    """

    domain = None
    normalizer_level = 3

    def normalize(self, archive: EntryArchive, logger=None) -> None:
        self.entry_archive = archive
        legacy_delegated = False

        # Setup logger
        if logger is not None:
            self.logger = logger.bind(normalizer=self.__class__.__name__)

        # ========== LOGICAL SWITCH: v2 basesections vs legacy ==========
        # Check if v2 data schema with basesections.v2 is present
        v2_schema_info = self._is_v2_data_schema(archive)

        if v2_schema_info:
            system_v2 = v2_schema_info if v2_schema_info is not True else None
            # NEW PATH: Use v2 data schema normalization (this plugin)
            self.logger.info('Using v2 data schema results normalization')
            self._normalize_with_data_schema(archive, self.logger, system_v2)
        else:
            # LEGACY PATH: Delegate to legacy normalizer (handles run schema,
            # old data schema, etc.)
            self.logger.info('Falling back to legacy results normalization')
            self._normalize_with_legacy(archive, self.logger)
            legacy_delegated = True

        # Legacy delegate handles measurements itself.
        if not legacy_delegated:
            for measurement in self.entry_archive.measurement:
                self.normalize_measurement(measurement)

        self.entry_archive = None
        self.section_run = None

    def _is_v2_data_schema(self, archive: EntryArchive) -> Any:
        """Check if archive uses v2 data schema with basesections.v2.

        Returns SystemV2 instance if found, True if indicated by model_system but no
        instance found, or False.
        """
        if archive.data is None:
            return False

        # Check if data has model_system (Simulation-style v2 schema indicator)
        try:
            has_model_system = archive.data.model_system is not None
        except Exception:
            has_model_system = False

        # Verify it's using basesections.v2 by checking the class origin
        from nomad.datamodel.metainfo.basesections.v2 import System as SystemV2

        for sec in archive.data.m_all_contents(include_self=True):
            if isinstance(sec, SystemV2):
                return sec

        # If model_system attribute exists but is empty, still consider it v2
        # schema (e.g. partially parsed Simulation).
        if has_model_system:
            return True

        # Non-simulation custom data sections (and any other data without
        # model_system) must go through the legacy fallback path.
        return False

    def _normalize_with_data_schema(
        self, archive: EntryArchive, logger, system_v2=None
    ) -> None:
        """Normalization cascade for v2 data schema (archive.data)."""
        from nomad_topology_normalizer.normalizers.topology import TopologyNormalizer

        # Initialize results sections
        results = self.entry_archive.results
        if results is None:
            results = self.entry_archive.m_create(Results)
        if results.properties is None:
            results.m_create(Properties)

        # Run topology normalizer for v2 schema
        topology_normalizer = TopologyNormalizer()
        topology_normalizer.normalize(archive, logger, system_v2=system_v2)
        self._normalize_method_with_data_schema(archive)
        self._normalize_outputs_with_data_schema(archive)

    def _normalize_method_with_data_schema(self, archive: EntryArchive) -> None:
        """Populate results.method from v2 Simulation data when available."""

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
                simulation_method.scf_threshold_energy_change = scf_threshold

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
            # enum value in nomad-FAIR results schema. For v2 multi-method
            # inputs, we intentionally use the first supported model_method as
            # the canonical method_name for backward-compatible search behavior.
            # The full multi-method design is deferred to results/search redesign.
            method.method_name = method_tokens[0]
            if len(method_tokens) > 1:
                self.logger.warning(
                    'multiple v2 model_method sections present; using first '
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
    def _mark_compatibility_section(section):
        """Mark a generated section without changing user-facing quantities."""
        section.m_annotations[V2_COMPATIBILITY_ANNOTATION] = {'generated': True}
        return section

    @staticmethod
    def _is_compatibility_section(section) -> bool:
        try:
            if V2_COMPATIBILITY_ANNOTATION in section.m_annotations:
                return True

            # Backward-compatible cleanup for archives produced by the initial
            # visible-marker implementation. New sections never use these fields.
            raw_id = getattr(section, 'raw_id', None)
            if raw_id == _LEGACY_V2_COMPATIBILITY_RUN_ID:
                return True
            label = getattr(section, 'label', None)
            if isinstance(label, str) and label.startswith(
                _LEGACY_V2_COMPATIBILITY_LABEL
            ):
                return True
            provenance = getattr(section, 'provenance', None)
            provenance_label = getattr(provenance, 'label', None)
            return isinstance(provenance_label, str) and provenance_label.startswith(
                _LEGACY_V2_COMPATIBILITY_LABEL
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
                        if not self._is_compatibility_section(section)
                    ],
                )

        spectroscopic = properties.spectroscopic
        if spectroscopic is not None:
            spectroscopic.spectra = [
                section
                for section in spectroscopic.spectra or []
                if not self._is_compatibility_section(section)
            ]

        structural = properties.structural
        if structural is not None:
            structural.radius_of_gyration = [
                section
                for section in structural.radius_of_gyration or []
                if not self._is_compatibility_section(section)
            ]

        thermodynamic = properties.thermodynamic
        if thermodynamic is not None:
            thermodynamic.trajectory = [
                section
                for section in thermodynamic.trajectory or []
                if not self._is_compatibility_section(section)
            ]

        geometry_optimization = properties.geometry_optimization
        if self._is_compatibility_section(geometry_optimization):
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
                if self._is_compatibility_section(candidate)
            ),
            None,
        )
        if run is None:
            run = runschema.run.Run()
            self._mark_compatibility_section(run)
            archive.run.append(run)
        else:
            self._mark_compatibility_section(run)
            if getattr(run, 'raw_id', None) == _LEGACY_V2_COMPATIBILITY_RUN_ID:
                run.m_set('raw_id', None)

        calculations = run.calculation
        if calculations and len(calculations) > 0:
            calculation = calculations[0]
        else:
            calculation = runschema.calculation.Calculation()
            run.calculation.append(calculation)
        self._mark_compatibility_section(calculation)

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
        """Map v2 Simulation.outputs into results.properties (minimal slice)."""

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
                if not self._is_compatibility_section(compatibility_run):
                    continue
                self._mark_compatibility_section(compatibility_run)
                if (
                    getattr(compatibility_run, 'raw_id', None)
                    == _LEGACY_V2_COMPATIBILITY_RUN_ID
                ):
                    compatibility_run.m_set('raw_id', None)
                for compatibility_calculation in compatibility_run.calculation or []:
                    self._mark_compatibility_section(compatibility_calculation)
                    compatibility_calculation.dos_electronic = []
                    compatibility_calculation.band_structure_electronic = []

        # Geometry optimization workflow metadata is consumed directly by the
        # geometry optimization card and should be available even when there is
        # no electronic/spectroscopic payload in outputs.
        geometry_optimization = self.geometry_optimization()
        if geometry_optimization is not None:
            self._mark_compatibility_section(geometry_optimization)
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
        latest_dos_payload: list[dict] = []
        latest_band_structures: list[BandStructureElectronic] = []
        latest_greens_functions: list[GreensFunctionsElectronic] = []
        electronic_groups: dict[tuple[int | None, int | None], dict[str, Any]] = {}
        spectra_sections: list[Spectra] = []
        rg_sections: list[RadiusOfGyration] = []
        temperature_series: list[float] = []
        temperature_time: list[float] = []
        potential_energy_series: list[float] = []
        potential_energy_time: list[float] = []

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
                    mapped_spectrum.provenance = SpectraProvenance(
                        label=method_label
                    )
                if mapped_spectrum:
                    spectra_sections.append(mapped_spectrum)
            for xas in output.xas_spectra or []:
                mapped_spectrum = self._map_spectrum(xas, 'XAS')
                if mapped_spectrum and method_label:
                    mapped_spectrum.provenance = SpectraProvenance(
                        label=method_label
                    )
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

        if electronic_groups:
            # Prefer the representative system, then explicit provenance, then the
            # latest output group. All electronic properties come from this one group.
            def _group_score(group: dict[str, Any]) -> tuple[int, int, int, int]:
                return (
                    int(
                        representative_system is not None
                        and group['model_system_ref'] is representative_system
                    ),
                    int(group['model_method_ref'] is not None),
                    int(group['model_system_ref'] is not None),
                    group['last_index'],
                )

            selected_group = max(electronic_groups.values(), key=_group_score)
            latest_band_gaps = selected_group['band_gaps']
            latest_dos_sections = selected_group['dos_sections']
            latest_dos_payload = selected_group['dos_payload']
            latest_band_structures = selected_group['band_structures']
            latest_greens_functions = selected_group['greens_functions']

            if len(electronic_groups) > 1:
                self.logger.warning(
                    'discarding alternate electronic output source groups',
                    selected_method=getattr(
                        getattr(selected_group['model_method_ref'], 'm_def', None),
                        'name',
                        None,
                    ),
                    discarded_groups=len(electronic_groups) - 1,
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
                    'dos_electronic: could not build payload from v2 outputs.'
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
        if runschema and latest_dos_sections and latest_dos_payload:
            compatibility_target = self._ensure_legacy_run_calculation(archive)
            if compatibility_target is not None:
                calculation, run_index, calculation_index = compatibility_target
                calculation.dos_electronic = []
                run_total_refs: list[str] = []
                for idx, dos_entry in enumerate(latest_dos_payload):
                    legacy_dos = runschema.calculation.Dos()
                    legacy_dos.energies = dos_entry['energies_points']

                    legacy_total = runschema.calculation.DosValues()
                    legacy_total.value = dos_entry['values']
                    legacy_total.spin = int(dos_entry.get('spin_channel', 0) or 0)
                    legacy_dos.total.append(legacy_total)

                    calculation.dos_electronic.append(legacy_dos)
                    run_total_refs.append(
                        f'/run/{run_index}/calculation/{calculation_index}'
                        f'/dos_electronic/{idx}/total/0'
                    )

                latest_dos_sections[0].energies = (
                    f'/run/{run_index}/calculation/{calculation_index}'
                    '/dos_electronic/0/energies'
                )
                latest_dos_sections[0].total = run_total_refs

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
                self._mark_compatibility_section(band_gap)
                electronic.m_add_sub_section(ElectronicProperties.band_gap, band_gap)
            for dos in latest_dos_sections:
                self._mark_compatibility_section(dos)
                electronic.m_add_sub_section(ElectronicProperties.dos_electronic, dos)
            for band_structure in latest_band_structures:
                self._mark_compatibility_section(band_structure)
                electronic.m_add_sub_section(
                    ElectronicProperties.band_structure_electronic, band_structure
                )
            for greens in latest_greens_functions:
                self._mark_compatibility_section(greens)
                electronic.m_add_sub_section(
                    ElectronicProperties.greens_functions_electronic, greens
                )

        if spectra_sections:
            spectroscopic = properties.spectroscopic
            if spectroscopic is None:
                spectroscopic = properties.m_create(SpectroscopicProperties)
            for spectrum in spectra_sections:
                self._mark_compatibility_section(spectrum)
                spectroscopic.m_add_sub_section(
                    SpectroscopicProperties.spectra, spectrum
                )

        if rg_sections:
            structural = properties.structural
            if structural is None:
                structural = properties.m_create(StructuralProperties)
            for rg in rg_sections:
                self._mark_compatibility_section(rg)
                structural.m_add_sub_section(
                    StructuralProperties.radius_of_gyration, rg
                )

        if temperature_series or potential_energy_series:
            thermodynamic = properties.thermodynamic
            if thermodynamic is None:
                thermodynamic = properties.m_create(ThermodynamicProperties)
            trajectory = Trajectory()
            self._mark_compatibility_section(trajectory)
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
                'dos_electronic: could not build payload from v2 outputs.'
            )
            self.logger.warning(dos_warning)

        self._log_unmapped_output_groups(outputs)

    def _normalize_with_legacy(self, archive: EntryArchive, logger) -> None:
        """Normalization cascade for legacy schemas (v1 run schema, old data
        schemas, etc.).

        Delegates to the old ResultsNormalizer from nomad-FAIR which handles
        all legacy cases.
        """
        from nomad.normalizing.results import (
            ResultsNormalizer as LegacyResultsNormalizer,
        )

        legacy_normalizer = LegacyResultsNormalizer()
        legacy_normalizer.normalize(archive, logger)

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

    def normalize_run(self, logger=None) -> None:
        from nomad_topology_normalizer.normalizers.material import MaterialNormalizer

        # Fetch different information resources from which data is gathered
        repr_system = None
        for section in self.section_run.system:
            if section.is_representative:
                repr_system = section
                break
        try:
            optimade = self.entry_archive.metadata.optimade
        except Exception:
            optimade = None

        repr_symmetry = None
        if repr_system and repr_system.symmetry:
            repr_symmetry = repr_system.symmetry[0]

        # Create the section and populate the subsections
        results = self.entry_archive.results
        properties, conv_atoms, wyckoff_sets, spg_number = self.properties(
            repr_system, repr_symmetry
        )
        results.properties = properties
        results.material = MaterialNormalizer(
            self.entry_archive,
            repr_system,
            repr_symmetry,
            spg_number,
            conv_atoms,
            wyckoff_sets,
            properties,
            optimade,
            logger,
        ).material()

        results.method = MethodNormalizer(
            self.entry_archive, repr_system, results.material, logger
        ).method()

        # set entry type based on method and material
        workflow = self.entry_archive.workflow2
        if workflow is not None:
            workflow_name = workflow.name if workflow.name else workflow.m_def.name

            tag = ''
            if results.method.simulation:
                tag = 'simulation'
                entry_type = self.entry_archive.metadata.entry_type
                try:
                    if entry_type == workflow_name or not entry_type:
                        method_name = results.method.method_name
                        program_name = results.method.simulation.program_name
                        if workflow_name == 'SinglePoint' and method_name:
                            self.entry_archive.metadata.entry_type = (
                                f'{program_name} {method_name} {workflow_name}'
                            )
                        else:
                            self.entry_archive.metadata.entry_type = (
                                f'{program_name} {workflow_name}'
                            )
                except Exception:
                    if not entry_type:
                        self.entry_archive.metadata.entry_type = workflow_name
            type_tag = f'{self.entry_archive.metadata.entry_type} {tag}'

            # Populate entry_name
            entry_name = self.entry_archive.metadata.entry_name
            if not entry_name or entry_name == os.path.basename(
                self.entry_archive.metadata.mainfile
            ):
                material = results.material
                if material and material.chemical_formula_descriptive:
                    self.entry_archive.metadata.entry_name = (
                        f'{material.chemical_formula_descriptive} {type_tag}'
                    )
                else:
                    self.entry_archive.metadata.entry_name = f'{type_tag}'

    def resolve_band_gap(
        self, path: list[str] = ['run', 'calculation', 'band_gap']
    ) -> list[BandGap]:
        """Extract all band gaps from the given `path` and return them in a list along
        with their provenance.
        """
        bg_root: list[BandGap] = []
        if band_gaps := traverse_reversed(self.entry_archive, path):
            for bg in band_gaps:
                bg_results = BandGap()
                bg_results.index = bg.index
                bg_results.value = bg.value
                bg_results.type = bg.type
                bg_results.energy_highest_occupied = bg.energy_highest_occupied
                bg_results.energy_lowest_unoccupied = bg.energy_lowest_unoccupied
                bg_results.provenance = bg.provenance
                bg_root.insert(0, bg_results)
        return bg_root

    def resolve_band_structure(
        self, path: list[str] = ['run', 'calculation', 'band_structure_electronic']
    ) -> list[BandStructureElectronic]:
        """Returns a new section containing an electronic band structure. In
        the case of multiple valid band structures, only the latest one is
        considered.

        Band structure is reported only under the following conditions:
            - There is a non-empty array of kpoints.
            - There is a non-empty array of energies.
        """
        bs_root: list[BandStructureElectronic] = []
        if band_structures := traverse_reversed(self.entry_archive, path):
            for bs in band_structures:
                if not bs.segment:
                    continue
                valid = True
                for segment in bs.segment:
                    energies = segment.energies
                    k_points = segment.kpoints
                    if not valid_array(energies) or not valid_array(k_points):
                        valid = False
                        break
                if valid:
                    # Fill band structure data to the newer, improved data layout
                    bs_results = BandStructureElectronic()
                    bs_results.reciprocal_cell = bs.reciprocal_cell
                    bs_results.segment = [
                        segment.__class__().m_from_dict(segment.m_to_dict())
                        for segment in bs.segment
                    ]
                    bs_results.spin_polarized = (
                        bs_results.segment[0].energies.shape[0] > 1
                    )
                    bs_results.energy_fermi = bs.energy_fermi

                    for info in bs.band_gap:
                        info_new = BandGapDeprecated().m_from_dict(info.m_to_dict())
                        bs_results.m_add_sub_section(
                            BandStructureElectronic.band_gap, info_new
                        )
                    bs_root.insert(0, bs_results)
        return bs_root

    def resolve_dos_deprecated(
        self, path: list[str] = ['run', 'calculation', 'dos_electronic']
    ) -> list[DOSElectronic]:
        """Returns a reference to the section containing an electronic dos. In
        the case of multiple valid DOSes, only the latest one is reported.

        DOS is reported only under the following conditions:
            - There is a non-empty array of dos_values_normalized.
            - There is a non-empty array of dos_energies.

        NOTE: this function will be eventually deprecated. This is because
        DOSElectronic refers to an old schema which will be deleted. The new
        function `resolve_dos` should be the one which persists over time.
        """
        dos_sections = extract_section(self.entry_archive, path, full_list=True)
        # The old mapping does not work for the new spin-polarized schema
        if (
            not dos_sections or len(dos_sections) == 2
        ):  # ? shouldn't len(dos_sections) < 2 to pass
            return []
        dos = dos_sections[0]
        energies = dos.energies
        values = np.array([d.value.magnitude for d in dos.total])
        dos_results = None
        if valid_array(energies) and valid_array(values):
            dos_results = DOSElectronic()
            dos_results.energies = dos
            dos_results.total = dos.total
            dos_results.energy_fermi = dos.energy_fermi
        return [dos_results] if dos_results else []

    def resolve_dos(
        self, path: list[str] = ['run', 'calculation', 'dos_electronic']
    ) -> list[DOSElectronicNew]:
        """Returns a section containing the references for an electronic DOS.
        This section is then stored under
        `archive.results.properties.electronic.dos_electronic_new`.

        If the calculation is spin-polarized, inside this new section there is
        a list `data` of length 2 and a boolean `spin_polarized` set to true.
        It also reference the species-, atom-, and orbital-projected DOS, if
        these are present.

        This section is populated only when there are non-empty arrays for
        energies and DOS.total values.

        Args:
            path (list[str]): the path to the dos_electronic section to be
                extracted from the self.entry_archive.

        Returns:
            List[DOSElectronicNew]: the mapped DOS.
        """
        dos_result = None  # only instantiate `dos_results` if the tests below pass
        if dos_sections := extract_section(self.entry_archive, path, full_list=True):
            for dos_section in dos_sections:
                energies = dos_section.energies
                values = np.array([d.value.magnitude for d in dos_section.total])
                if valid_array(energies) and valid_array(values):
                    dos_result = DOSElectronicNew() if not dos_result else dos_result
                    dos_data = dos_result.m_create(DOSNew)
                    dos_data.energies = dos_section
                    dos_data.total = dos_section.total[-1]
                    dos_data.energy_fermi = dos_section.energy_fermi
                    dos_data.energy_ref = dos_section.energy_ref
                    # Storing deprecated BandGap info
                    for info in dos_section.band_gap:
                        info_new = BandGapDeprecated().m_from_dict(info.m_to_dict())
                        dos_data.m_add_sub_section(DOSNew.band_gap, info_new)
                    # Spin-polarized
                    dos_result.spin_polarized = len(dos_sections) == 2
                    dos_data.spin_channel = dos_section.spin_channel
                    # Projected DOS
                    has_projected = False
                    _projected_sections = {
                        key: value
                        for key, value in dos_section.m_def.all_sub_sections.items()
                        if 'projected' in key
                    }
                    _projected_data = {
                        key: value
                        for key, value in dos_data.m_def.all_quantities.items()
                        if 'projected' in key
                    }
                    for key, value in _projected_sections.items():
                        dos_projected = dos_section.m_get(value)
                        if dos_projected is not None and len(dos_projected) > 0:
                            dos_data.m_set(_projected_data.get(key), dos_projected)
                            has_projected = True
                    dos_result.has_projected = has_projected
        return [dos_result] if dos_result else []

    def resolve_greens_functions(
        self, path: list[str] = ['run', 'calculation', 'greens_functions']
    ) -> list[GreensFunctionsElectronic]:
        """Returns a section containing the references of the electronic Greens
        functions. This section is then stored under
        `archive.results.properties.electronic`.

        This section is only populated if there are non-zero values of the tau,
        Matsubara_freq, or frequencies, and its respective greens_function
        quantities.

        Args:
            path (list[str]): the path to the dos_electronic section to be
                extracted from the self.entry_archive.

        Returns:
            List[GreensFunctionsElectronic]: the mapped Greens functions.
        """
        gfs_root: list[GreensFunctionsElectronic] = []
        if greens_functions := traverse_reversed(self.entry_archive, path):
            for gfs in greens_functions:
                gfs_results = GreensFunctionsElectronic()
                # tau-axes quantities
                tau = gfs.tau
                if valid_array(tau):
                    gfs_results.tau = gfs
                    gfs_results.greens_function_tau = (
                        gfs if valid_array(gfs.greens_function_tau) else None
                    )
                # matsubara_freq-axes quantities
                matsubara_freq = gfs.matsubara_freq
                if valid_array(matsubara_freq):
                    gfs_results.matsubara_freq = gfs
                    gfs_results.greens_function_iw = (
                        gfs if valid_array(gfs.greens_function_iw) else None
                    )
                    gfs_results.self_energy_iw = (
                        gfs if valid_array(gfs.self_energy_iw) else None
                    )
                # frequencies-axes quantities
                frequencies = gfs.frequencies
                if valid_array(frequencies):
                    gfs_results.frequencies = gfs
                    gfs_results.greens_function_freq = (
                        gfs if valid_array(gfs.greens_function_freq) else None
                    )
                    gfs_results.self_energy_freq = (
                        gfs if valid_array(gfs.self_energy_freq) else None
                    )
                    gfs_results.hybridization_function_freq = (
                        gfs if valid_array(gfs.hybridization_function_freq) else None
                    )
                # Other GFs quantities
                gfs_results.orbital_occupations = (
                    gfs if valid_array(gfs.orbital_occupations) else None
                )
                gfs_results.quasiparticle_weights = (
                    gfs if valid_array(gfs.quasiparticle_weights) else None
                )
                if gfs.chemical_potential:
                    gfs_results.chemical_potential = gfs
                gfs_results.type = gfs.type if gfs.type else None
                gfs_root.append(gfs_results)
        return gfs_root

    def fetch_charge_density(
        self, path: list[str] = ['run', 'calculation', 'density_charge', 'value_hdf5']
    ) -> list[DensityCharge]:
        """Fetch charge density data.

        TODO: Implement charge density support for v2 data schema.
        Charge density is not yet available in nomad-simulations outputs.
        Once nomad-simulations adds DensityCharge property, update this method to:
        1. Check for archive.data.outputs[-1].density_charge or similar
        2. Map to results.properties.electronic.density_charge

        For now, this returns empty list for all schemas (legacy support removed).
        """
        # TODO: Uncomment and adapt once nomad-simulations has DensityCharge
        # return_list: list[DensityCharge] = []
        # if hdf5_wrappers := list(traverse_reversed(self.entry_archive, path)):
        #     for hdf5_wrapper in hdf5_wrappers:
        #         d = DensityCharge()
        #         d.m_set('value_hdf5', hdf5_wrapper.path)
        #         return_list.append(d)
        # return return_list
        pass
        return []

    def resolve_electric_field_gradient(
        self, path: list[str] = ['run', 'calculation', 'electric_field_gradient']
    ) -> list[ElectricFieldGradient]:
        """Returns a section containing the references for the Electric Field Gradient.
        This section is then stored under `archive.results.properties.electronic`.

        TODO: Implement EFG support for v2 data schema.
        Electric Field Gradient is not yet available in nomad-simulations outputs.
        Once nomad-simulations adds ElectricFieldGradient property, update
        this method to:
        1. Check for archive.data.outputs[-1].electric_field_gradients or similar
        2. Map to results.properties.electronic.electric_field_gradient

        For now, this returns empty list for all schemas (legacy support removed).

        Args:
            path (list[str]): the path to the electric field gradient section
                to be extracted from the self.entry_archive.

        Returns:
            list[ElectricFieldGradient]: the mapped Electric Field Gradient.
        """
        # TODO: Uncomment and adapt once nomad-simulations has ElectricFieldGradient
        # mapped_data: list[ElectricFieldGradient] = []
        # if stored_data := traverse_reversed(self.entry_archive, path):
        #     for data in stored_data:
        #         contribution = data.contribution
        #         value = data.value
        #         if valid_array(value):
        #             results_data = ElectricFieldGradient(
        #                 contribution=contribution, value=data
        #             )
        #             mapped_data.insert(0, results_data)
        # return mapped_data
        pass
        return []

    def resolve_spectra(self, path: list[str]) -> list[Spectra] | None:
        """Returns a section containing the references for a Spectra. This
        section is then stored under `archive.results.properties.spectroscopic`.

        This section is populated only when there are non-empty arrays for
        energies and intensities.

        Args:
            path (list[str]): the path to the spectra section to be extracted from the
                self.entry_archive.

        Returns:
            list[Spectra]: the mapped Spectra.
        """
        spectra = traverse_reversed(self.entry_archive, path)
        if not spectra:
            return None
        spectra_root: list[Spectra] = []
        for spectrum in spectra:
            n_energies = spectrum.n_energies
            if n_energies and n_energies > 0:
                spectra_results = Spectra(
                    type=spectrum.type, label='computation', n_energies=n_energies
                )
                provenance = spectra_results.m_create(SpectraProvenance)
                provenance.electronic_structure = spectrum.provenance
                energies = spectrum.excitation_energies
                intensities = spectrum.intensities
                if valid_array(energies) and valid_array(intensities):
                    spectra_results.energies = energies
                    spectra_results.intensities = intensities
                    if spectrum.intensities_units:
                        spectra_results.intensities_units = spectrum.intensities_units
                    spectra_root.insert(0, spectra_results)
        return spectra_root

    def resolve_magnetic_shielding(
        self, path: list[str]
    ) -> list[MagneticShielding] | None:
        """Returns a section containing the references for the (atomic) Magnetic
        Shielding. This section is then stored under
        `archive.results.properties.magnetic`.

        This section is populated only when there is a non empty array of
        `magnetic_shielding.value`.

        Args:
            path (list[str]): the path to the magnetic shielding section to be extracted
            from the self.entry_archive.

        Returns:
            List[MagneticShielding]: the mapped Magnetic Shielding.
        """
        stored_data = traverse_reversed(self.entry_archive, path)
        if not stored_data:
            return None
        mapped_data: list[MagneticShielding] = []
        for data in stored_data:
            value = data.value
            if valid_array(value):
                results_data = MagneticShielding(value=data)
                mapped_data.insert(0, results_data)
        return mapped_data

    def resolve_spin_spin_coupling(
        self, path: list[str]
    ) -> list[SpinSpinCoupling] | None:
        """Returns a section containing the references for the Spin Spin Coupling.
        This section is then stored under `archive.results.properties.magnetic`.

        This section is populated only when there is a non empty array of
        `spin_spin_coupling.value`.

        Args:
            path (list[str]): the path to the spin-spin coupling section to be extracted
            from the self.entry_archive.

        Returns:
            list[SpinSpinCoupling]: the mapped Spin Spin Coupling.
        """
        stored_data = traverse_reversed(self.entry_archive, path)
        if not stored_data:
            return None
        mapped_data: list[SpinSpinCoupling] = []
        for data in stored_data:
            contribution = data.contribution
            value = data.value
            reduced_value = data.reduced_value
            if valid_array(value) or valid_array(reduced_value):
                results_data = SpinSpinCoupling(
                    source='simulation',
                    contribution=contribution,
                    value=data if valid_array(value) else None,
                    reduced_value=data if valid_array(reduced_value) else None,
                )
                mapped_data.insert(0, results_data)
        return mapped_data

    def resolve_magnetic_susceptibility(
        self, path: list[str]
    ) -> list[MagneticSusceptibility] | None:
        """Returns a section containing the references for the Magnetic Susceptibility.
        This section is then stored under `archive.results.properties.magnetic`.

        This section is populated only when there is a non empty array of
        `magnetic_susceptibility.value`.

        Args:
            path (list[str]): the path to the magnetic susceptibility section
                to be extracted from the self.entry_archive.

        Returns:
            list[MagneticSusceptibility]: the mapped Magnetic Susceptibility.
        """
        stored_data = traverse_reversed(self.entry_archive, path)
        if not stored_data:
            return None
        mapped_data: list[MagneticSusceptibility] = []
        for data in stored_data:
            scale_dimension = data.scale_dimension
            value = data.value
            if valid_array(value):
                results_data = MagneticSusceptibility(
                    source='simulation', scale_dimension=scale_dimension, value=data
                )
                mapped_data.insert(0, results_data)
        return mapped_data

    def _resolve_workflow_gs_properties(
        self, methods: list[str], properties: list[str]
    ) -> None:
        """Resolves the ground state (gs) properties passed as a list
        `properties` (band_gap, band_structure, dos) for a given list of
        `methods` (dft, gw, tb, maxent).

        Args:
            methods (list[str]): the list of methods from which the
                properties are resolved.
            properties (list[str]): the list of properties to be resolved
                from `workflow2.results`.
        """
        properties_map = {
            'dos': 'dos_electronic_new',
            'band_structure': 'band_structure_electronic',
        }
        for method in methods:
            name = (
                'MaxEnt'
                if method == 'maxent'
                else 'FirstPrinciples'
                if method == 'first_principles'
                else method.upper()
            )
            for prop in properties:
                property_list = self.electronic_properties.get(
                    properties_map.get(prop, prop)
                )
                method_property_resolved = getattr(self, f'resolve_{prop}')(
                    ['workflow2', 'results', f'{method}_outputs', prop]
                )
                for item in method_property_resolved:
                    item.label = name
                    property_list.append(item)

    def get_gw_workflow_properties(self) -> None:
        """Gets the GW workflow (DFT+GW) properties and stores them in the
        self.electronic_properties dictionary.
        """
        properties = ['band_gap', 'band_structure', 'dos']
        methods = ['dft', 'gw']
        self._resolve_workflow_gs_properties(methods, properties)

    def get_tb_workflow_properties(self):
        """Gets the TB workflow (DFT+TB or GW+TB) properties and stores them
        in the self.electronic_properties dictionary.
        """
        properties = ['band_gap', 'band_structure', 'dos']
        methods = ['first_principles', 'tb']
        self._resolve_workflow_gs_properties(methods, properties)

    def get_dmft_workflow_properties(self) -> None:
        """Gets the DMFT workflow (DFT+TB+DMFT) properties and stores them in the
        self.electronic_properties dictionary.
        """
        properties = ['band_gap', 'band_structure', 'dos']
        methods = ['dft', 'tb']
        self._resolve_workflow_gs_properties(methods, properties)
        # Resolving DMFT Greens functions
        gfs_electronic: list[GreensFunctionsElectronic] = (
            self.electronic_properties.get('greens_functions_electronic')  # type: ignore
        )
        gfs_electronic_dmft = self.resolve_greens_functions(
            ['workflow2', 'results', 'dmft_outputs', 'greens_functions']
        )
        for item in gfs_electronic_dmft:
            item.label = 'DMFT'
            gfs_electronic.append(item)

    def get_maxent_workflow_properties(self) -> None:
        """Gets the MaxEnt workflow (DMFT+MaxEnt) properties and stores them
        in the self.electronic_properties dictionary.
        """
        properties = ['band_gap', 'dos']
        methods = ['maxent']
        self._resolve_workflow_gs_properties(methods, properties)
        # Resolving DMFT Greens functions
        gfs_electronic: list[GreensFunctionsElectronic] = (
            self.electronic_properties.get('greens_functions_electronic')  # type: ignore
        )
        for method in ['dmft', 'maxent']:
            name = 'MaxEnt' if method == 'maxent' else method.upper()
            gfs = self.resolve_greens_functions(
                [
                    'workflow2',
                    'results',
                    f'{method}_outputs',
                    'greens_functions',
                ]
            )
            for item in gfs:
                item.label = name
                gfs_electronic.append(item)

    def get_xs_workflow_properties(self, spectra: list[Spectra]) -> list[Spectra]:
        """Gets the XS workflow (DFT+GW+BSE) workflow properties and stores
        them in self.electronic_properties and in spectra. Then it returns the
        new Spectra section with the resolved data

        Args:
            spectra (Union[list[Spectra], None]): the input Spectra section
                resolved from `archive.run`.

        Returns:
            Union[List[Spectra], None]: the mapped Spectra from `workflow2.results`.
        """  # ! TODO: double-check typing
        properties = ['band_gap', 'band_structure', 'dos']
        methods = ['dft', 'gw']
        self._resolve_workflow_gs_properties(methods, properties)
        spct_electronic = spectra
        spectra = self.resolve_spectra(
            ['workflow2', 'results', 'spectra', 'spectrum_polarization']
        )
        if spectra:
            spct_electronic = spectra
        return spct_electronic

    def band_structure_phonon(self) -> BandStructurePhonon | None:
        """Returns a new section containing a phonon band structure. In
         the case of multiple valid band structures, only the latest one is
         considered.

        Band structure is reported only under the following conditions:
           - There is a non-empty array of kpoints.
           - There is a non-empty array of energies.
        """
        path = ['run', 'calculation', 'band_structure_phonon']
        for bs in traverse_reversed(self.entry_archive, path):
            if not bs.segment:
                continue
            valid = True
            for segment in bs.segment:
                energies = segment.energies
                k_points = segment.kpoints
                if not valid_array(energies) or not valid_array(k_points):
                    valid = False
                    break
            if valid:
                # Fill band structure data to the newer, improved data layout
                bs_new = BandStructurePhonon()
                bs_new.segment = [
                    segment.__class__().m_from_dict(segment.m_to_dict())
                    for segment in bs.segment
                ]
                return bs_new

        return None

    def dos_phonon(self) -> DOSPhonon | None:
        """Returns a section containing phonon dos data. In the case of
         multiple valid data sources, only the latest one is reported.

        DOS is reported only under the following conditions:
           - There is a non-empty array of values.
           - There is a non-empty array of energies.
        """
        path = ['run', 'calculation', 'dos_phonon']
        for dos in traverse_reversed(self.entry_archive, path):
            energies = dos.energies
            values = np.array([d.value.magnitude for d in dos.total])
            if valid_array(energies) and valid_array(values):
                dos_new = DOSPhonon()
                dos_new.energies = dos
                dos_new.total = dos.total
                return dos_new

        return None

    def energy_free_helmholtz(self) -> EnergyFreeHelmholtz | None:
        """Returns a section Helmholtz free energy data. In the case of
         multiple valid data sources, only the latest one is reported.

        Helmholtz free energy is reported only under the following conditions:
           - There is a non-empty array of temperatures.
           - There is a non-empty array of energies.
        """
        workflow = self.entry_archive.workflow2
        if workflow is None or not hasattr(workflow, 'results'):
            return None
        if not workflow.results or not hasattr(workflow.results, 'temperature'):
            return None

        path = ['workflow2', 'results']

        for thermo_prop in traverse_reversed(self.entry_archive, path):
            temperatures = thermo_prop.temperature
            energies = thermo_prop.vibrational_free_energy_at_constant_volume
            if valid_array(temperatures) and valid_array(energies):
                energy_free = EnergyFreeHelmholtz()
                energy_free.energies = thermo_prop
                energy_free.temperatures = thermo_prop
                return energy_free

        return None

    def heat_capacity_constant_volume(self) -> HeatCapacityConstantVolume | None:
        """Returns a section containing heat capacity data. In the case of
         multiple valid data sources, only the latest one is reported.

        Heat capacity is reported only under the following conditions:
           - There is a non-empty array of temperatures.
           - There is a non-empty array of energies.
        """
        workflow = self.entry_archive.workflow2
        if workflow is None or not hasattr(workflow, 'results'):
            return None
        if not workflow.results or not hasattr(workflow.results, 'temperature'):
            return None

        path = ['workflow2', 'results']
        for thermo_prop in traverse_reversed(self.entry_archive, path):
            temperatures = thermo_prop.temperature
            heat_capacities = thermo_prop.heat_capacity_c_v
            if valid_array(temperatures) and valid_array(heat_capacities):
                heat_cap = HeatCapacityConstantVolume()
                heat_cap.heat_capacities = thermo_prop
                heat_cap.temperatures = thermo_prop
                return heat_cap

        return None

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

    def get_md_provenance(self, workflow: Workflow) -> MolecularDynamics | None:
        """Retrieves the MD provenance from the given workflow."""
        md = None
        if workflow.m_def.name == 'MolecularDynamics':
            try:
                md = MolecularDynamics()
                md.time_step = workflow.method.integration_timestep
                md.ensemble_type = workflow.method.thermodynamic_ensemble
            except Exception:
                pass
        return md

    def trajectory(self) -> list[Trajectory]:  # noqa: PLR0912, PLR0915
        """Returns a list of trajectories."""
        path = ['workflow2']
        trajs = []
        for workflow in traverse_reversed(self.entry_archive, path):
            # Check validity
            if workflow.m_def.name == 'MolecularDynamics':
                traj = Trajectory()
                md = self.get_md_provenance(workflow)
                if md:
                    traj.provenance = MDProvenance(molecular_dynamics=md)

                # Loop through calculations, gather thermodynamics directly
                # from each step in the workflow.
                volume = []
                volume_time = []
                pressure = []
                pressure_time = []
                temperature = []
                temperature_time = []
                potential_energy = []
                potential_energy_time = []

                calculations_ref = []
                if workflow.results and workflow.results.calculations_ref:
                    calculations_ref = workflow.results.calculations_ref
                for calc in calculations_ref:
                    time = calc.time
                    if time is not None:
                        time = time.magnitude
                        if calc.volume is not None:
                            volume.append(calc.volume.magnitude)
                            volume_time.append(time)
                        if calc.pressure is not None:
                            pressure.append(calc.pressure.magnitude)
                            pressure_time.append(time)
                        if calc.temperature is not None:
                            temperature.append(calc.temperature.magnitude)
                            temperature_time.append(time)
                        if calc.energy:
                            if calc.energy.potential is not None:
                                potential_energy.append(
                                    calc.energy.potential.value.magnitude
                                )
                                potential_energy_time.append(time)

                available_properties = []
                if volume:
                    traj.volume = VolumeDynamic(value=volume, time=volume_time)
                    available_properties.append('volume')
                if pressure:
                    traj.pressure = PressureDynamic(value=pressure, time=pressure_time)
                    available_properties.append('pressure')
                if temperature:
                    traj.temperature = TemperatureDynamic(
                        value=temperature, time=temperature_time
                    )
                    available_properties.append('temperature')
                if potential_energy:
                    traj.energy_potential = EnergyDynamic(
                        value=potential_energy, time=potential_energy_time
                    )
                    available_properties.append('energy_potential')
                if available_properties:
                    traj.available_properties = available_properties
                trajs.append(traj)
        return trajs

    def rdf(self) -> list[RadialDistributionFunction]:
        """Returns a list of radial distribution functions."""
        workflow = self.entry_archive.workflow2
        if workflow is None or workflow.m_def.name != 'MolecularDynamics':
            return None

        path = ['workflow2', 'results', 'radial_distribution_functions']
        rdfs = []
        for rdf_workflow in traverse_reversed(self.entry_archive, path):
            rdf_values = rdf_workflow.radial_distribution_function_values
            if rdf_values is not None:
                for rdf_value in rdf_values or []:
                    rdf = RadialDistributionFunction()
                    try:
                        rdf.bins = rdf_value.bins
                        rdf.n_bins = rdf_value.n_bins
                        rdf.value = rdf_value.value
                        rdf.label = rdf_value.label
                        rdf.frame_start = rdf_value.frame_start
                        rdf.frame_end = rdf_value.frame_end
                        rdf.type = rdf_workflow.type
                        md = self.get_md_provenance(
                            rdf_workflow.m_parent.m_parent.m_parent
                        )
                        if md:
                            rdf.provenance = MDProvenance(molecular_dynamics=md)
                    except Exception as e:
                        self.logger.error(
                            'error in resolving radial distribution data', exc_info=e
                        )
                    else:
                        rdfs.append(rdf)

        return rdfs

    def rg(self) -> list[RadiusOfGyration]:
        """Returns a list of Radius of gyration trajectories."""
        path_workflow = ['workflow2']
        rgs: list[RadiusOfGyration] = []
        for workflow in traverse_reversed(self.entry_archive, path_workflow):
            # Check validity
            if workflow.m_def.name == 'MolecularDynamics' and workflow.results:
                results = workflow.results
                md = self.get_md_provenance(workflow)
                if (
                    results.calculations_ref
                    and results.calculations_ref[0].radius_of_gyration
                ):
                    for rg_index, rg in enumerate(
                        results.calculations_ref[0].radius_of_gyration
                    ):
                        for rg_values_index, __ in enumerate(
                            rg.radius_of_gyration_values
                        ):
                            rg_results = RadiusOfGyration()
                            rg_value = []
                            rg_time = []
                            if md:
                                rg_results.provenance = MDProvenance(
                                    molecular_dynamics=md
                                )
                            for calc in results.calculations_ref:
                                if not calc.system_ref:
                                    continue
                                sec_rg = calc.radius_of_gyration[rg_index]
                                rg_results.kind = sec_rg.kind
                                time = calc.time
                                if time is not None:
                                    time = time.magnitude
                                sec_rg_values = sec_rg.radius_of_gyration_values[
                                    rg_values_index
                                ]
                                rg_results.label = sec_rg_values.label
                                rg_results.atomsgroup_ref = sec_rg_values.atomsgroup_ref
                                rg_time.append(time)
                                rg_value.append(sec_rg_values.value.magnitude)
                            rg_results.time = rg_time
                            rg_results.value = rg_value
                    rgs.append(rg_results)
        return rgs

    def msd(self) -> list[MeanSquaredDisplacement]:
        """Returns a list of mean squared displacements."""
        workflow = self.entry_archive.workflow2
        if workflow is None or workflow.m_def.name != 'MolecularDynamics':
            return None

        path = ['workflow2', 'results', 'mean_squared_displacements']
        msds = []
        for msd_workflow in traverse_reversed(self.entry_archive, path):
            msd_values = msd_workflow.mean_squared_displacement_values
            if msd_values is not None:
                for msd_value in msd_values or []:
                    msd = MeanSquaredDisplacement()
                    try:
                        msd.times = msd_value.times
                        msd.n_times = msd_value.n_times
                        msd.value = msd_value.value
                        msd.label = msd_value.label
                        msd.errors = msd_value.errors
                        msd.type = msd_workflow.type
                        msd.direction = msd_workflow.direction
                        msd.error_type = msd_workflow.error_type
                        diffusion_constant = msd_value.diffusion_constant
                        if diffusion_constant is not None:
                            msd.diffusion_constant_value = diffusion_constant.value
                            msd.diffusion_constant_error_type = (
                                diffusion_constant.error_type
                            )
                            msd.diffusion_constant_errors = (
                                diffusion_constant.errors
                                if isinstance(
                                    diffusion_constant.errors, list | np.ndarray
                                )
                                else [diffusion_constant.errors]
                            )

                        md = self.get_md_provenance(
                            msd_workflow.m_parent.m_parent.m_parent
                        )
                        if md:
                            msd.provenance = MDProvenance(molecular_dynamics=md)
                    except Exception as e:
                        self.logger.error(
                            'error in resolving mean squared displacement data',
                            exc_info=e,
                        )
                    else:
                        msds.append(msd)

        return msds

    def properties(  # noqa: PLR0912, PLR0915
        self, repr_system: ArchiveSection, repr_symmetry: ArchiveSection
    ) -> tuple:
        """Returns a populated Properties subsection."""
        properties = Properties()

        # Structures
        conv_atoms = None
        wyckoff_sets = None
        spg_number = None
        if repr_system:
            original_atoms = repr_system.m_cache.get('representative_atoms')
            if original_atoms:
                structural_type = repr_system.type
                if structural_type == 'bulk':
                    conv_atoms, _, wyckoff_sets, spg_number = self.structures_bulk(
                        repr_symmetry
                    )
                elif structural_type == '2D':
                    conv_atoms, _, wyckoff_sets, spg_number = structures_2d(
                        original_atoms
                    )
                elif structural_type == '1D':
                    conv_atoms, _ = self.structures_1d(original_atoms)

        self.electronic_properties: dict[str, list[Any]] = {
            'band_gap': self.resolve_band_gap(),
            'dos_electronic': self.resolve_dos_deprecated(),
            'dos_electronic_new': self.resolve_dos(),
            'band_structure_electronic': self.resolve_band_structure(),
            'greens_functions_electronic': self.resolve_greens_functions(),
            'density_charge': self.fetch_charge_density(),
            'electric_field_gradient': self.resolve_electric_field_gradient(),
        }
        #   spectroscopic properties list
        spectra = self.resolve_spectra(['run', 'calculation', 'spectra'])
        # Resolving GW, XS workflow properties
        workflow = self.entry_archive.workflow2
        if workflow:
            workflow_name = workflow.name if workflow.name else workflow.m_def.name
            if workflow_name == 'DFT+GW':
                self.get_gw_workflow_properties()
            elif workflow_name == 'FirstPrinciples+TB':
                self.get_tb_workflow_properties()
            elif workflow_name in ['DFT+TB+DMFT', 'DFT+DMFT', 'TB+DMFT']:
                self.get_dmft_workflow_properties()
            elif workflow_name == 'DMFT+MaxEnt':
                self.get_maxent_workflow_properties()
            elif workflow_name in ['PhotonPolarization', 'BSE']:
                spectra = self.resolve_spectra(
                    ['workflow2', 'results', 'spectrum_polarization']
                )
            elif workflow_name == 'XS':
                spectra = self.get_xs_workflow_properties(spectra)

        # check if a property in `ElectronicProperties` is present
        if any(len(value) > 0 for value in self.electronic_properties.values()):
            electronic = ElectronicProperties()
            for (
                property_name,
                electronic_property,
            ) in self.electronic_properties.items():
                if len(electronic_property) > 0:
                    for prop in electronic_property:
                        electronic.m_append(property_name, prop)
            properties.electronic = electronic

        # Spectroscopic
        if spectra:
            spectroscopic = SpectroscopicProperties()
            spectroscopic.spectra = spectra
            properties.spectroscopic = spectroscopic

        # Magnetic
        magnetic_shielding = self.resolve_magnetic_shielding(
            ['run', 'calculation', 'magnetic_shielding']
        )
        spin_spin_coupling = self.resolve_spin_spin_coupling(
            ['run', 'calculation', 'spin_spin_coupling']
        )
        magnetic_susceptibility = self.resolve_magnetic_susceptibility(
            ['run', 'calculation', 'magnetic_susceptibility']
        )
        if magnetic_shielding or spin_spin_coupling or magnetic_susceptibility:
            magnetic = MagneticProperties()
            if magnetic_shielding:
                magnetic.magnetic_shielding = magnetic_shielding
            if spin_spin_coupling:
                magnetic.spin_spin_coupling = spin_spin_coupling
            if magnetic_susceptibility:
                magnetic.magnetic_susceptibility = magnetic_susceptibility
            properties.magnetic = magnetic

        # Vibrational
        bs_phonon = self.band_structure_phonon()
        dos_phonon = self.dos_phonon()
        energy_free = self.energy_free_helmholtz()
        heat_cap = self.heat_capacity_constant_volume()
        if bs_phonon or dos_phonon or energy_free or heat_cap:
            vibrational = VibrationalProperties()
            if dos_phonon:
                vibrational.dos_phonon = dos_phonon
            if bs_phonon:
                vibrational.band_structure_phonon = bs_phonon
            if energy_free:
                vibrational.energy_free_helmholtz = energy_free
            if heat_cap:
                vibrational.heat_capacity_constant_volume = heat_cap
            properties.vibrational = vibrational

        # Mechanical
        energy_volume_curves = self.energy_volume_curves()
        bulk_modulus = self.bulk_modulus()
        shear_modulus = self.shear_modulus()
        geometry_optimization = self.geometry_optimization()
        if (
            energy_volume_curves
            or bulk_modulus
            or shear_modulus
            or geometry_optimization
        ):
            mechanical = MechanicalProperties()
            for ev in energy_volume_curves:
                mechanical.m_add_sub_section(
                    MechanicalProperties.energy_volume_curve, ev
                )
            for bm in bulk_modulus:
                mechanical.m_add_sub_section(MechanicalProperties.bulk_modulus, bm)
            for sm in shear_modulus:
                mechanical.m_add_sub_section(MechanicalProperties.shear_modulus, sm)
            properties.mechanical = mechanical

        # Geometry optimization
        properties.geometry_optimization = self.geometry_optimization()

        # Thermodynamic
        trajectory = self.trajectory()
        if trajectory:
            thermodynamic = ThermodynamicProperties()
            thermodynamic.trajectory = trajectory
            properties.thermodynamic = thermodynamic

        # Structural
        rdf = self.rdf()
        rg = self.rg()
        if rdf or rg:
            structural = StructuralProperties()
            structural.radial_distribution_function = rdf
            structural.radius_of_gyration = rg
            properties.structural = structural

        # Dynamical
        msd = self.msd()
        if msd:
            dynamical = DynamicalProperties()
            dynamical.mean_squared_displacement = msd
            properties.dynamical = dynamical

        try:
            n_calc = len(self.section_run.calculation)
        except Exception:
            n_calc = 0
        properties.n_calculations = n_calc

        return properties, conv_atoms, wyckoff_sets, spg_number

    def structures_bulk(self, repr_symmetry):
        """The symmetry of bulk structures has already been analyzed. Here we
        use the cached results.
        """
        conv_atoms = None
        prim_atoms = None
        wyckoff_sets = None
        spg_number = None
        if repr_symmetry:
            symmetry_analyzer = repr_symmetry.m_cache.get('symmetry_analyzer')
            if symmetry_analyzer:
                spg_number = symmetry_analyzer.get_space_group_number()
                conv_atoms = symmetry_analyzer.get_conventional_system()
                prim_atoms = symmetry_analyzer.get_primitive_system()

                # For some reason MatID seems to drop the periodicity,
                # reintroduce it here.
                conv_atoms.set_pbc(True)
                prim_atoms.set_pbc(True)
                try:
                    wyckoff_sets = symmetry_analyzer.get_wyckoff_sets_conventional(
                        return_parameters=True
                    )
                except Exception:
                    self.logger.error('Error resolving Wyckoff sets.')
                    wyckoff_sets = []

        return conv_atoms, prim_atoms, wyckoff_sets, spg_number

    def structures_1d(self, original_atoms):
        conv_atoms = None
        prim_atoms = None
        try:
            # First get a symmetry analyzer and the primitive system
            symm_system = original_atoms.copy()
            symm_system.set_pbc(True)
            symmetry_analyzer = SymmetryAnalyzer(
                symm_system,
                config.normalize.symmetry_tolerance,
                config.normalize.flat_dim_threshold,
            )
            prim_atoms = symmetry_analyzer.get_primitive_system()
            prim_atoms.set_pbc(True)

            # Get dimension of system by also taking into account the covalent radii
            dimensions = matid.geometry.get_dimensions(prim_atoms, [True, True, True])
            basis_dimensions = np.linalg.norm(prim_atoms.get_cell(), axis=1)
            gaps = basis_dimensions - dimensions
            periodicity = gaps <= config.normalize.cluster_threshold

            # If one axis is not periodic, return. This only happens if the vacuum
            # gap is not aligned with a cell vector.
            if sum(periodicity) != 1:
                self.logger.warning(
                    'could not detect the periodic dimensions in a 1D system'
                )
                return conv_atoms, prim_atoms

            # Translate to center of mass
            conv_atoms = prim_atoms.copy()
            pbc_cm = matid.geometry.get_center_of_mass(prim_atoms)
            cell_center = 0.5 * np.sum(conv_atoms.get_cell(), axis=0)
            translation = cell_center - pbc_cm
            translation[periodicity] = 0
            conv_atoms.translate(translation)
            conv_atoms.wrap()
            conv_atoms.set_pbc(periodicity)

            # Reduce cell size to just fit the system in the non-periodic dimensions.
            conv_atoms = atomutils.get_minimized_structure(conv_atoms)

            # Swap the cell axes so that the periodic one is always the first
            # basis (=a)
            swap_dim = 0
            for i, periodic in enumerate(periodicity):
                if periodic:
                    periodic_dim = i
                    break
            if periodic_dim != swap_dim:
                atomutils.swap_basis(conv_atoms, periodic_dim, swap_dim)

            prim_atoms = conv_atoms
        except Exception as e:
            self.logger.error(
                'could not construct a conventional system for a 1D material',
                exc_info=e,
            )
        return conv_atoms, prim_atoms

    def energy_volume_curves(self) -> list[EnergyVolumeCurve]:
        """Returns a list containing the found EnergyVolumeCurves."""
        workflow = self.entry_archive.workflow2
        ev_curves: list[EnergyVolumeCurve] = []
        # workflow must be equation of state
        if (
            workflow is None
            or workflow.m_def.name != 'EquationOfState'
            or workflow.results is None
        ):
            return ev_curves

        # Volumes must be present
        volumes = workflow.results.volumes
        if not valid_array(volumes):
            self.logger.warning('missing eos volumes')
            return ev_curves

        # Raw EV curve
        energies_raw = workflow.results.energies
        if valid_array(energies_raw):
            ev_curves.append(
                EnergyVolumeCurve(
                    type='raw',
                    volumes=workflow.results,
                    energies_raw=workflow.results,
                )
            )
        else:
            self.logger.warning('missing eos energies')

        # Fitted EV curves
        fits = workflow.results.eos_fit
        if not fits:
            return ev_curves
        for fit in fits:
            energies_fitted = fit.fitted_energies
            function_name = fit.function_name
            if valid_array(energies_fitted):
                ev_curves.append(
                    EnergyVolumeCurve(
                        type=function_name,
                        volumes=workflow.results,
                        energies_fit=fit,
                    )
                )

        return ev_curves

    def bulk_modulus(self) -> list[BulkModulus]:
        """Returns a list containing the found BulkModulus."""
        workflow = self.entry_archive.workflow2
        bulk_modulus: list[BulkModulus] = []
        if (
            workflow is None
            or not hasattr(workflow, 'results')
            or workflow.results is None
        ):
            return bulk_modulus

        if workflow.m_def.name == 'Elastic':
            bulk_modulus_vrh = workflow.results.bulk_modulus_hill
            if bulk_modulus_vrh:
                bulk_modulus.append(
                    BulkModulus(
                        type='voigt_reuss_hill_average',
                        value=bulk_modulus_vrh,
                    )
                )
            bulk_modulus_voigt = workflow.results.bulk_modulus_voigt
            if bulk_modulus_voigt:
                bulk_modulus.append(
                    BulkModulus(
                        type='voigt_average',
                        value=bulk_modulus_voigt,
                    )
                )
            bulk_modulus_reuss = workflow.results.bulk_modulus_reuss
            if bulk_modulus_reuss:
                bulk_modulus.append(
                    BulkModulus(
                        type='reuss_average',
                        value=bulk_modulus_reuss,
                    )
                )

        if workflow.m_def.name == 'EquationOfState':
            fits = workflow.results.eos_fit
            if not fits:
                return bulk_modulus

            for fit in fits:
                modulus = fit.bulk_modulus
                function_name = fit.function_name
                if modulus is not None and function_name:
                    bulk_modulus.append(
                        BulkModulus(
                            type=function_name,
                            value=modulus,
                        )
                    )
                else:
                    self.logger.warning(
                        'missing eos fitted energies and/or function name'
                    )

        return bulk_modulus

    def shear_modulus(self) -> list[ShearModulus]:
        """Returns a list containing the found ShearModulus."""
        workflow = self.entry_archive.workflow2
        shear_modulus: list[ShearModulus] = []
        if (
            workflow is None
            or not hasattr(workflow, 'results')
            or workflow.results is None
        ):
            return shear_modulus

        if workflow.m_def.name != 'Elastic':
            return shear_modulus

        shear_modulus_vrh = workflow.results.shear_modulus_hill
        if shear_modulus_vrh:
            shear_modulus.append(
                ShearModulus(
                    type='voigt_reuss_hill_average',
                    value=shear_modulus_vrh,
                )
            )
        shear_modulus_voigt = workflow.results.shear_modulus_voigt
        if shear_modulus_voigt:
            shear_modulus.append(
                ShearModulus(
                    type='voigt_average',
                    value=shear_modulus_voigt,
                )
            )
        shear_modulus_reuss = workflow.results.shear_modulus_reuss
        if shear_modulus_reuss:
            shear_modulus.append(
                ShearModulus(
                    type='reuss_average',
                    value=shear_modulus_reuss,
                )
            )

        return shear_modulus
