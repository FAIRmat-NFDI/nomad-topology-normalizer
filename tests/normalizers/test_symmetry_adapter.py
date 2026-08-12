from types import SimpleNamespace

from nomad.datamodel.results import Symmetry, SymmetryNew
from nomad_simulations.schema_packages.model_system import GlobalCrystalSymmetry

from nomad_results_normalizer.normalizers.symmetry_adapter import (
    apply_symmetry_data_to_results_symmetry,
    find_model_system_representation,
    from_legacy_repr_symmetry,
    from_model_system,
    has_complete_model_system_symmetry,
    is_symmetry_data_minimally_complete,
    wyckoff_sets_from_model_system,
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


def test_complete_model_system_symmetry_provides_legacy_material_id_inputs():
    symmetry = SimpleNamespace(
        space_group_number=166,
        space_group_symbol='R-3m',
        point_group_symbol='-3m',
    )
    local_symmetry = SimpleNamespace(
        wyckoff_letters=['c', 'c'],
        equivalent_atoms=[0, 0],
        site_multiplicities=[6, 6],
    )
    conventional = SimpleNamespace(
        name='conventional', lattice_vectors=[[8.0, 0.0, 0.0]] * 3
    )
    model_system = SimpleNamespace(
        symmetry=symmetry,
        local_symmetry=local_symmetry,
        particle_states=[
            SimpleNamespace(chemical_symbol='Si', atomic_number=14),
            SimpleNamespace(chemical_symbol='Si', atomic_number=14),
        ],
        representations=[conventional],
    )

    wyckoff_sets = wyckoff_sets_from_model_system(model_system)

    assert len(wyckoff_sets) == 1
    assert wyckoff_sets[0].element == 'Si'
    assert wyckoff_sets[0].wyckoff_letter == 'c'
    assert len(wyckoff_sets[0].indices) == 6
    assert (
        find_model_system_representation(model_system, 'conventional') is conventional
    )
    assert has_complete_model_system_symmetry(model_system)


def test_crystal_system_is_derived_from_v2_lattice_type():
    """The real v2 section has no `crystal_system`; it must come from `lattice_type`.

    Built from `GlobalCrystalSymmetry` rather than a namespace so that a schema
    rename breaks this test instead of silently emptying a search field.
    """
    model_system = SimpleNamespace(
        symmetry=GlobalCrystalSymmetry(
            space_group_number=227,
            space_group_symbol='Fd-3m',
            point_group_symbol='m-3m',
            lattice_type='c - cubic',
            lattice_centering='F - all faces centred',
        ),
        local_symmetry=None,
    )

    symmetry_data = from_model_system(model_system)

    assert symmetry_data['crystal_system'] == 'cubic'
    assert symmetry_data['bravais_lattice'] == 'cF'

    target = Symmetry()
    apply_symmetry_data_to_results_symmetry(target, symmetry_data)
    assert target.crystal_system == 'cubic'
    assert target.bravais_lattice == 'cF'


def test_two_dimensional_lattice_type_has_no_legacy_equivalent():
    """2D Pearson symbols are not enum members and must not be assigned."""
    model_system = SimpleNamespace(
        symmetry=GlobalCrystalSymmetry(
            space_group_number=1,
            space_group_symbol='P1',
            point_group_symbol='1',
            lattice_type='mp - oblique',
            lattice_centering='p - primitive 2D/1D',
        ),
        local_symmetry=None,
    )

    symmetry_data = from_model_system(model_system)

    assert symmetry_data['crystal_system'] is None
    assert symmetry_data['bravais_lattice'] is None

    # Would raise on the MEnum quantity if an out-of-enum value leaked through.
    apply_symmetry_data_to_results_symmetry(Symmetry(), symmetry_data)


def test_prototype_id_reaches_topology_symmetry_section():
    """Topology nodes use `SymmetryNew`, which names the AFLOW id differently."""
    symmetry_data = from_model_system(
        SimpleNamespace(
            symmetry=GlobalCrystalSymmetry(
                space_group_number=225,
                space_group_symbol='Fm-3m',
                point_group_symbol='m-3m',
                prototype_aflow_id='AB_cF8_225_a_b',
            ),
            local_symmetry=None,
        )
    )

    topology_symmetry = SymmetryNew()
    apply_symmetry_data_to_results_symmetry(topology_symmetry, symmetry_data)
    assert topology_symmetry.prototype_label_aflow == 'AB_cF8_225_a_b'

    material_symmetry = Symmetry()
    apply_symmetry_data_to_results_symmetry(material_symmetry, symmetry_data)
    assert material_symmetry.prototype_aflow_id == 'AB_cF8_225_a_b'


def test_incomplete_local_symmetry_does_not_invent_wyckoff_sets():
    model_system = SimpleNamespace(
        local_symmetry=SimpleNamespace(
            wyckoff_letters=['a'],
            equivalent_atoms=None,
            site_multiplicities=[1],
        ),
        particle_states=[SimpleNamespace(chemical_symbol='Si', atomic_number=14)],
    )

    assert wyckoff_sets_from_model_system(model_system) is None
