from nomad.config.models.plugins import NormalizerEntryPoint


class TopologyNormalizerEntryPoint(NormalizerEntryPoint):
    level: int = 3

    def load(self):
        # Import lazily to avoid circulars during module initialization
        from nomad_topology_normalizer.normalizers.topology import (
            TopologyNormalizer,
        )

        # Don't pass entry point config as __init__ args - Normalizer doesn't accept them
        return TopologyNormalizer()


topology_normalizer_plugin = TopologyNormalizerEntryPoint(
    name='Topology ',
    description='New normalizer entry point configuration.',
)
