from nomad.config.models.plugins import NormalizerEntryPoint


class TopologyNormalizerEntryPoint(NormalizerEntryPoint):
    level: int = 3

    def load(self):
        from nomad_topology_normalizer.normalizers.normalizer import TopologyNormalizer

        return TopologyNormalizer  # (**self.dict())


topology_normalizer_plugin = TopologyNormalizerEntryPoint(
    name='TopologyNormalizer',
    description='New normalizer entry point configuration.',
    python_package='nomad_topology_normalizer',
    normalizer_class_name='nomad_topology_normalizer.normalizers.normalizer.TopologyNormalizer',
)
