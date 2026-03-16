from types import SimpleNamespace
from unittest.mock import Mock

from nomad_topology_normalizer.normalizers.normalizer import Normalizer


def _make_normalizer() -> Normalizer:
    normalizer = Normalizer()
    normalizer.logger = Mock()
    return normalizer


def test_representative_system_prefers_workflow_system_ref():
    normalizer = _make_normalizer()

    workflow_system = SimpleNamespace(name='workflow-system')
    model_system = SimpleNamespace(name='model-system', is_representative=True)
    archive = SimpleNamespace(
        workflow2=SimpleNamespace(
            results=SimpleNamespace(
                calculation_result_ref=SimpleNamespace(system_ref=workflow_system)
            )
        ),
        data=SimpleNamespace(model_system=[model_system], representative_system_index=0),
    )

    result = normalizer._representative_system(archive)

    assert result is workflow_system


def test_representative_system_uses_representative_system_index():
    normalizer = _make_normalizer()

    model_systems = [
        SimpleNamespace(name='system-0', is_representative=False),
        SimpleNamespace(name='system-1', is_representative=False),
    ]
    archive = SimpleNamespace(
        workflow2=None,
        data=SimpleNamespace(model_system=model_systems, representative_system_index=1),
    )

    result = normalizer._representative_system(archive)

    assert result is model_systems[1]


def test_representative_system_falls_back_to_flagged_system():
    normalizer = _make_normalizer()

    model_systems = [
        SimpleNamespace(name='system-0', is_representative=False),
        SimpleNamespace(name='system-1', is_representative=True),
    ]
    archive = SimpleNamespace(
        workflow2=None,
        data=SimpleNamespace(model_system=model_systems, representative_system_index=99),
    )

    result = normalizer._representative_system(archive)

    assert result is model_systems[1]


def test_representative_system_falls_back_to_last_system():
    normalizer = _make_normalizer()

    model_systems = [
        SimpleNamespace(name='system-0', is_representative=False),
        SimpleNamespace(name='system-1', is_representative=False),
    ]
    archive = SimpleNamespace(
        workflow2=None,
        data=SimpleNamespace(model_system=model_systems, representative_system_index=None),
    )

    result = normalizer._representative_system(archive)

    assert result is model_systems[-1]

