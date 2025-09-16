from nomad.config.models.plugins import NormalizerEntryPoint
from pydantic import Field


class NewNormalizerEntryPoint(NormalizerEntryPoint):
    parameter: int = Field(0, description='Custom configuration parameter')

    def load(self):
        from nomad_topology_normalizer.normalizers.normalizer import NewNormalizer

        return NewNormalizer(**self.model_dump())


normalizer_entry_point = NewNormalizerEntryPoint(
    name='NewNormalizer',
    description='New normalizer entry point configuration.',
)
