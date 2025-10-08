import warnings

from nomad.config.models.plugins import NormalizerEntryPoint


class TopologyNormalizerEntryPoint(NormalizerEntryPoint):
    level: int = 3

    def load(self):
        try:
            # Import lazily to avoid circulars during module initialization
            from .normalizer import TopologyNormalizer

            return TopologyNormalizer(**self.dict())
        except Exception as e:
            warnings.warn(
                f'TopologyNormalizer not ready during plugin scan ({e!r}); using No-Op normalizer.'
            )
            from nomad.normalizing import Normalizer

        return TopologyNormalizer(**self.dict())


topology_normalizer_plugin = TopologyNormalizerEntryPoint(
    name='TopologyNormalizer',
    description='New normalizer entry point configuration.',
    python_package='nomad_topology_normalizer',
    normalizer_class_name='nomad_topology_normalizer.normalizers.normalizer.TopologyNormalizer',
)
