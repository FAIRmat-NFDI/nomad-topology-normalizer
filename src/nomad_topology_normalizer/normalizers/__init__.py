from nomad.config.models.plugins import NormalizerEntryPoint


class TopologyNormalizerEntryPoint(NormalizerEntryPoint):
    normalizer_level = 3

    def load(self):
        from nomad_topology_normalizer.normalizers.normalizer import TopologyNormalizer

        return TopologyNormalizer(**self.dict())


system_normalizer_plugin = SystemNormalizerEntryPoint(
    name='System ',
    description='New normalizer entry point configuration.',
)
