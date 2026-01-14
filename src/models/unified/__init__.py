"""
Unified Models for CineSync v2
Comprehensive improvements to recommendation system
"""

from .cross_domain_embeddings import (
    UnifiedUserEmbedding,
    UnifiedItemEmbedding,
    CrossDomainRecommender
)

from .movie_ensemble_system import (
    MovieEnsembleRecommender,
    AdaptiveWeightingModule,
    UncertaintyEstimator
)

from .contrastive_learning import (
    InfoNCELoss,
    ContrastiveLearningModule,
    DataAugmentation,
    HardNegativeMiner
)

from .multimodal_features import (
    TextFeatureExtractor,
    VisualFeatureExtractor,
    AudioFeatureExtractor,
    MetadataEncoder,
    MultimodalFusion,
    CompleteMultimodalEncoder
)

from .context_aware import (
    TemporalContextEncoder,
    DeviceContextEncoder,
    SocialContextEncoder,
    MoodContextEncoder,
    ContextAwareRecommender
)

__all__ = [
    # Cross-domain
    'UnifiedUserEmbedding',
    'UnifiedItemEmbedding',
    'CrossDomainRecommender',

    # Ensemble
    'MovieEnsembleRecommender',
    'AdaptiveWeightingModule',
    'UncertaintyEstimator',

    # Contrastive learning
    'InfoNCELoss',
    'ContrastiveLearningModule',
    'DataAugmentation',
    'HardNegativeMiner',

    # Multimodal
    'TextFeatureExtractor',
    'VisualFeatureExtractor',
    'AudioFeatureExtractor',
    'MetadataEncoder',
    'MultimodalFusion',
    'CompleteMultimodalEncoder',

    # Context-aware
    'TemporalContextEncoder',
    'DeviceContextEncoder',
    'SocialContextEncoder',
    'MoodContextEncoder',
    'ContextAwareRecommender',
]
