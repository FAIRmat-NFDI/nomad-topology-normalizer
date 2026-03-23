from __future__ import annotations

from typing import Any

# TODO(migration): Keep adapter local to topology-normalizer during v2 migration.
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
        bravais_lattice=getattr(symmetry, 'bravais_lattice', None),
        crystal_system=getattr(symmetry, 'crystal_system', None),
        space_group_number=getattr(symmetry, 'space_group_number', None),
        space_group_symbol=getattr(symmetry, 'space_group_symbol', None),
        point_group=getattr(symmetry, 'point_group_symbol', None),
        strukturbericht_designation=getattr(
            symmetry, 'strukturbericht_designation', None
        ),
        prototype_formula=getattr(symmetry, 'prototype_formula', None),
        prototype_aflow_id=getattr(symmetry, 'prototype_aflow_id', None),
        origin_shift=getattr(symmetry, 'analysis_origin_shift', None),
        transformation_matrix=getattr(
            symmetry, 'analysis_transformation_matrix', None
        ),
        wyckoff_letters=getattr(local_symmetry, 'wyckoff_letters', None),
        equivalent_atoms=getattr(local_symmetry, 'equivalent_atoms', None),
        site_multiplicities=getattr(local_symmetry, 'site_multiplicities', None),
    )
    return symmetry_data


def apply_symmetry_data_to_results_symmetry(
    target_symmetry: Any, symmetry_data: dict[str, Any]
) -> None:
    """Apply symmetry data values to a results Symmetry-like target section."""
    if target_symmetry is None or symmetry_data is None:
        return

    field_map = {
        'hall_number': 'hall_number',
        'hall_symbol': 'hall_symbol',
        'bravais_lattice': 'bravais_lattice',
        'crystal_system': 'crystal_system',
        'space_group_number': 'space_group_number',
        'space_group_symbol': 'space_group_symbol',
        'point_group': 'point_group',
        'strukturbericht_designation': 'strukturbericht_designation',
        'prototype_formula': 'prototype_formula',
        'prototype_aflow_id': 'prototype_aflow_id',
        'origin_shift': 'origin_shift',
        'transformation_matrix': 'transformation_matrix',
    }
    for source_name, target_name in field_map.items():
        value = symmetry_data.get(source_name)
        if value is not None and hasattr(target_symmetry, target_name):
            setattr(target_symmetry, target_name, value)
