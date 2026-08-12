from __future__ import annotations

from typing import Any

from ase.data import chemical_symbols
from nomad.datamodel.results import WyckoffSet, bravais_lattices, crystal_systems

# TODO(migration): Keep adapter local to results-normalizer during migration.
# TODO(migration): Re-evaluate elevation only after nomad-simulations/nomad-FAIR
# symmetry field stability is confirmed and cross-plugin parity fixtures exist.


def empty_symmetry_data() -> dict[str, Any]:
    """Create an empty legacy-equivalent symmetry data dictionary."""
    return {
        'hall_number': None,
        'hall_symbol': None,
        'bravais_lattice': None,
        'crystal_system': None,
        'space_group_number': None,
        'space_group_symbol': None,
        'point_group': None,
        'strukturbericht_designation': None,
        'prototype_formula': None,
        'prototype_aflow_id': None,
        'origin_shift': None,
        'transformation_matrix': None,
        'wyckoff_letters': None,
        'equivalent_atoms': None,
        'site_multiplicities': None,
    }


def is_symmetry_data_minimally_complete(symmetry_data: dict[str, Any]) -> bool:
    """Return True when core crystallographic identifiers are available."""
    return bool(
        symmetry_data.get('space_group_number') is not None
        and symmetry_data.get('space_group_symbol')
        and symmetry_data.get('point_group')
    )


def _crystal_system_from_lattice_type(lattice_type: Any) -> str | None:
    """Map a v2 `lattice_type` onto the legacy `crystal_system` enum.

    `GlobalCrystalSymmetry` has no `crystal_system` quantity; it carries the
    crystal family in Pearson-style `'<code> - <name>'` form (e.g. `'c - cubic'`).
    Only the seven 3D names have a legacy-equivalent target, so the 2D/1D lattice
    types (`'mp - oblique'`, `'hp - hexagonal 2D'`, ...) stay unmapped.
    """
    if not isinstance(lattice_type, str):
        return None
    crystal_system = lattice_type.split(' - ')[-1].strip().lower()
    return crystal_system if crystal_system in crystal_systems else None


def _legacy_bravais_lattice(bravais_lattice: Any) -> str | None:
    """Keep only Pearson symbols that the legacy `bravais_lattice` enum accepts.

    The v2 `bravais_lattice` property also reconstructs 2D/1D symbols such as
    `'mpp'` or `'ocp'`, which are not enum members and would raise on assignment.
    """
    if not isinstance(bravais_lattice, str):
        return None
    return bravais_lattice if bravais_lattice in bravais_lattices else None


def from_legacy_repr_symmetry(repr_symmetry: Any) -> dict[str, Any]:
    """Normalize legacy `repr_system.symmetry[0]` into symmetry data."""
    if repr_symmetry is None:
        return empty_symmetry_data()

    symmetry_data = empty_symmetry_data()
    symmetry_data.update(
        hall_number=getattr(repr_symmetry, 'hall_number', None),
        hall_symbol=getattr(repr_symmetry, 'hall_symbol', None),
        bravais_lattice=getattr(repr_symmetry, 'bravais_lattice', None),
        crystal_system=getattr(repr_symmetry, 'crystal_system', None),
        space_group_number=getattr(repr_symmetry, 'space_group_number', None),
        space_group_symbol=getattr(
            repr_symmetry,
            'international_short_symbol',
            getattr(repr_symmetry, 'space_group_symbol', None),
        ),
        point_group=getattr(repr_symmetry, 'point_group', None),
        strukturbericht_designation=getattr(
            repr_symmetry, 'strukturbericht_designation', None
        ),
        prototype_formula=getattr(repr_symmetry, 'prototype_formula', None),
        prototype_aflow_id=getattr(repr_symmetry, 'prototype_aflow_id', None),
        origin_shift=getattr(repr_symmetry, 'origin_shift', None),
        transformation_matrix=getattr(repr_symmetry, 'transformation_matrix', None),
    )
    return symmetry_data


def from_model_system(model_system: Any) -> dict[str, Any]:
    """Normalize v2 ModelSystem symmetry/local_symmetry into symmetry data."""
    if model_system is None:
        return empty_symmetry_data()

    symmetry = getattr(model_system, 'symmetry', None)
    local_symmetry = getattr(model_system, 'local_symmetry', None)
    symmetry_data = empty_symmetry_data()
    if symmetry is None:
        symmetry_data.update(
            wyckoff_letters=getattr(local_symmetry, 'wyckoff_letters', None),
            equivalent_atoms=getattr(local_symmetry, 'equivalent_atoms', None),
            site_multiplicities=getattr(local_symmetry, 'site_multiplicities', None),
        )
        return symmetry_data

    symmetry_data.update(
        hall_number=getattr(symmetry, 'hall_number', None),
        hall_symbol=getattr(symmetry, 'hall_symbol', None),
        bravais_lattice=_legacy_bravais_lattice(
            getattr(symmetry, 'bravais_lattice', None)
        ),
        crystal_system=getattr(symmetry, 'crystal_system', None)
        or _crystal_system_from_lattice_type(getattr(symmetry, 'lattice_type', None)),
        space_group_number=getattr(symmetry, 'space_group_number', None),
        space_group_symbol=getattr(symmetry, 'space_group_symbol', None),
        point_group=getattr(symmetry, 'point_group_symbol', None),
        strukturbericht_designation=getattr(
            symmetry, 'strukturbericht_designation', None
        ),
        prototype_formula=getattr(symmetry, 'prototype_formula', None),
        prototype_aflow_id=getattr(symmetry, 'prototype_aflow_id', None),
        origin_shift=getattr(symmetry, 'analysis_origin_shift', None),
        transformation_matrix=getattr(symmetry, 'analysis_transformation_matrix', None),
        wyckoff_letters=getattr(local_symmetry, 'wyckoff_letters', None),
        equivalent_atoms=getattr(local_symmetry, 'equivalent_atoms', None),
        site_multiplicities=getattr(local_symmetry, 'site_multiplicities', None),
    )
    return symmetry_data


def find_model_system_representation(model_system: Any, cell_type: str) -> Any | None:
    """Return an existing analyzed representation for the requested cell type."""
    if model_system is None:
        return None
    requested = cell_type.lower()
    for representation in getattr(model_system, 'representations', None) or []:
        representation_type = getattr(representation, 'crystal_cell_type', None)
        representation_name = getattr(representation, 'name', None)
        if any(
            isinstance(value, str) and value.lower() == requested
            for value in (representation_type, representation_name)
        ):
            return representation
    return None


def wyckoff_sets_from_model_system(model_system: Any) -> list[WyckoffSet] | None:
    """Build legacy material-id inputs from normalized v2 local symmetry.

    ``site_multiplicities`` already contains the conventional-cell orbit size.
    The returned indices are therefore count carriers for the unchanged legacy
    ``material_id_bulk`` hashing function; no crystallographic analysis is repeated.
    """
    if model_system is None:
        return None

    local_symmetry = getattr(model_system, 'local_symmetry', None)
    letters = getattr(local_symmetry, 'wyckoff_letters', None)
    equivalent_atoms = getattr(local_symmetry, 'equivalent_atoms', None)
    multiplicities = getattr(local_symmetry, 'site_multiplicities', None)
    if letters is None or equivalent_atoms is None or multiplicities is None:
        return None

    try:
        labels = list(model_system.get_symbols())
    except Exception:
        labels = []
        for particle_state in getattr(model_system, 'particle_states', None) or []:
            label = getattr(particle_state, 'chemical_symbol', None)
            if label is None:
                atomic_number = getattr(particle_state, 'atomic_number', None)
                try:
                    label = chemical_symbols[int(atomic_number)]
                except Exception:
                    return None
            labels.append(label)

    try:
        if not labels or not (
            len(labels) == len(letters) == len(equivalent_atoms) == len(multiplicities)
        ):
            return None
    except TypeError:
        return None

    grouped_indices: dict[int, list[int]] = {}
    for index, equivalent_index in enumerate(equivalent_atoms):
        grouped_indices.setdefault(int(equivalent_index), []).append(index)

    wyckoff_sets: list[WyckoffSet] = []
    for indices in grouped_indices.values():
        elements = {str(labels[index]) for index in indices}
        group_letters = {str(letters[index]) for index in indices}
        group_multiplicities = {int(multiplicities[index]) for index in indices}
        if (
            len(elements) != 1
            or len(group_letters) != 1
            or len(group_multiplicities) != 1
        ):
            return None

        multiplicity = group_multiplicities.pop()
        if multiplicity <= 0:
            return None
        wyckoff_sets.append(
            WyckoffSet(
                element=elements.pop(),
                wyckoff_letter=group_letters.pop(),
                indices=list(range(multiplicity)),
            )
        )

    return wyckoff_sets or None


def has_complete_model_system_symmetry(model_system: Any) -> bool:
    """Whether v2 carries every input needed for bulk results and material ID."""
    symmetry_data = from_model_system(model_system)
    conventional = find_model_system_representation(model_system, 'conventional')
    return bool(
        is_symmetry_data_minimally_complete(symmetry_data)
        and wyckoff_sets_from_model_system(model_system)
        and conventional is not None
        and getattr(conventional, 'lattice_vectors', None) is not None
    )


def apply_symmetry_data_to_results_symmetry(
    target_symmetry: Any, symmetry_data: dict[str, Any]
) -> None:
    """Apply symmetry data values to a results Symmetry-like target section."""
    if target_symmetry is None or symmetry_data is None:
        return

    # `results.material.symmetry` (Symmetry) and topology `System.symmetry`
    # (SymmetryNew) name the AFLOW prototype id differently, so each source field
    # lists every legacy-equivalent target name and the first declared one wins.
    # Without this the prototype id is silently dropped on topology nodes.
    field_map = {
        'hall_number': ('hall_number',),
        'hall_symbol': ('hall_symbol',),
        'bravais_lattice': ('bravais_lattice',),
        'crystal_system': ('crystal_system',),
        'space_group_number': ('space_group_number',),
        'space_group_symbol': ('space_group_symbol',),
        'point_group': ('point_group',),
        'strukturbericht_designation': ('strukturbericht_designation',),
        # `SymmetryNew.prototype_name` carries the structure name (e.g. 'wurtzite'),
        # not the prototype formula, so it is deliberately not a target here.
        'prototype_formula': ('prototype_formula',),
        'prototype_aflow_id': ('prototype_aflow_id', 'prototype_label_aflow'),
        'origin_shift': ('origin_shift',),
        'transformation_matrix': ('transformation_matrix',),
    }
    for source_name, target_names in field_map.items():
        value = symmetry_data.get(source_name)
        if value is None:
            continue
        for target_name in target_names:
            if hasattr(target_symmetry, target_name):
                setattr(target_symmetry, target_name, value)
                break
