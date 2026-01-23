from nomad.config.models.plugins import NormalizerEntryPoint


class TopologyNormalizerEntryPoint(NormalizerEntryPoint):
    level: int = 3

    def load(self):
        # Import lazily to avoid circulars during module initialization
        from nomad_topology_normalizer.normalizers.topology import (
            TopologyNormalizer,
        )

        return TopologyNormalizer(**self.dict())


topology_normalizer_plugin = TopologyNormalizerEntryPoint(
    name='Topology ',
    description='New normalizer entry point configuration.',
)
