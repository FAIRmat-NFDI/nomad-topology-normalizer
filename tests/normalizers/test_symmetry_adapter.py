from types import SimpleNamespace

from nomad.datamodel.results import Symmetry

from nomad_topology_normalizer.normalizers.symmetry_adapter import (
    apply_symmetry_data_to_results_symmetry,
    from_legacy_repr_symmetry,
    from_model_system,
    is_symmetry_data_minimally_complete,
)


def test_from_legacy_repr_symmetry_maps_core_fields():
    repr_symmetry = SimpleNamespace(
        hall_number=123,
        hall_symbol='P 1',
        bravais_lattice='cP',
        crystal_system='cubic',
        space_group_number=221,
        international_short_symbol='Pm-3m',
        point_group='m-3m',
        strukturbericht_designation='B2',
        prototype_formula='AB',
        prototype_aflow_id='AB_cP2_221_a_b',
        origin_shift=[0.0, 0.0, 0.0],
        transformation_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )

    symmetry_data = from_legacy_repr_symmetry(repr_symmetry)

    assert symmetry_data['space_group_number'] == 221
    assert symmetry_data['space_group_symbol'] == 'Pm-3m'
    assert symmetry_data['point_group'] == 'm-3m'
    assert symmetry_data['prototype_aflow_id'] == 'AB_cP2_221_a_b'
    assert is_symmetry_data_minimally_complete(symmetry_data)


def test_from_model_system_maps_global_and_local_symmetry():
    symmetry = SimpleNamespace(
        hall_number=523,
        hall_symbol='-F 4 2 3',
        bravais_lattice='cF',
        crystal_system='cubic',
        space_group_number=216,
        space_group_symbol='F-43m',
        point_group_symbol='-43m',
        analysis_origin_shift=[0.0, 0.0, 0.0],
        analysis_transformation_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        prototype_formula='ABC',
        prototype_aflow_id='ABC_cF12_216_c_a_b',
    )
    local_symmetry = SimpleNamespace(
        wyckoff_letters=['a', 'c', 'b'],
        equivalent_atoms=[0, 1, 2],
        site_multiplicities=[4, 4, 4],
    )
    model_system = SimpleNamespace(symmetry=symmetry, local_symmetry=local_symmetry)

    symmetry_data = from_model_system(model_system)

    assert symmetry_data['space_group_number'] == 216
    assert symmetry_data['space_group_symbol'] == 'F-43m'
    assert symmetry_data['point_group'] == '-43m'
    assert symmetry_data['wyckoff_letters'] == ['a', 'c', 'b']
    assert symmetry_data['site_multiplicities'] == [4, 4, 4]


def test_apply_symmetry_data_to_results_symmetry_sets_available_fields():
    symmetry_data = {
        'hall_number': 123,
        'hall_symbol': 'P 1',
        'bravais_lattice': 'cP',
        'crystal_system': 'cubic',
        'space_group_number': 221,
        'space_group_symbol': 'Pm-3m',
        'point_group': 'm-3m',
        'prototype_aflow_id': 'AB_cP2_221_a_b',
        'prototype_formula': 'AB',
    }
    target = Symmetry()

    apply_symmetry_data_to_results_symmetry(target, symmetry_data)

    assert target.hall_number == 123
    assert target.space_group_number == 221
    assert target.space_group_symbol == 'Pm-3m'
    assert target.point_group == 'm-3m'
    assert target.prototype_aflow_id == 'AB_cP2_221_a_b'
