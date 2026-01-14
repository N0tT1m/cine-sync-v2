"""
Complete Integration Example for CineSync v2
Demonstrates how to use all improvements together
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import all improvements
from src.models.unified.cross_domain_embeddings import CrossDomainRecommender
from src.models.unified.movie_ensemble_system import MovieEnsembleRecommender
from src.models.unified.contrastive_learning import (
    ContrastiveLearningModule, DataAugmentation
)
from src.models.unified.multimodal_features import (
    TextFeatureExtractor, MetadataEncoder, MultimodalFusion
)
from src.models.unified.context_aware import ContextAwareRecommender
from src.evaluation.comprehensive_metrics import ComprehensiveEvaluator
from src.training.advanced_trainer import AdvancedTrainer
from src.production.optimizations import (
    ModelQuantizer, EmbeddingCache, FastRecommender
)


class CineSyncV2System:
    """
    Complete CineSync v2 recommendation system.

    Integrates all improvements:
    - Unified cross-domain embeddings
    - Movie ensemble
    - Contrastive learning
    - Multimodal features
    - Context awareness
    - Production optimizations
    """

    def __init__(
        self,
        num_users: int,
        num_movies: int,
        num_tv_shows: int,
        num_genres: int = 20,
        device: torch.device = None,
        cache_dir: Path = Path('./cache'),
        checkpoint_dir: Path = Path('./checkpoints')
    ):
        self.num_users = num_users
        self.num_movies = num_movies
        self.num_tv_shows = num_tv_shows
        self.num_genres = num_genres

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir
        self.checkpoint_dir = checkpoint_dir

        # Create directories
        cache_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        logger.info("Initializing CineSync v2 system...")
        self._initialize_models()
        self._initialize_feature_extractors()
        self._initialize_production_components()

        logger.info("CineSync v2 system ready!")

    def _initialize_models(self):
        """Initialize all model components"""

        # 1. Unified cross-domain model
        logger.info("Creating unified cross-domain model...")
        self.unified_model = CrossDomainRecommender(
            num_users=self.num_users,
            num_movies=self.num_movies,
            num_tv_shows=self.num_tv_shows,
            embedding_dim=512,
            num_genres=self.num_genres
        ).to(self.device)

        # 2. Wrap with contrastive learning
        logger.info("Adding contrastive learning...")
        self.contrastive_model = ContrastiveLearningModule(
            base_model=self.unified_model,
            embed_dim=512,
            projection_dim=256,
            temperature=0.07,
            use_momentum_encoder=True,
            queue_size=65536
        ).to(self.device)

        # 3. Movie ensemble (combines all movie models)
        logger.info("Building movie ensemble...")
        self.movie_ensemble = MovieEnsembleRecommender(
            num_users=self.num_users,
            num_movies=self.num_movies,
            num_genres=self.num_genres,
            embed_dim=768,
            fusion_strategy='attention',
            enable_collaborative_models=True,
            enable_sequential_models=True,
            enable_transformer_models=True
        ).to(self.device)

        # 4. Context-aware wrapper
        logger.info("Adding context awareness...")
        self.context_aware_model = ContextAwareRecommender(
            base_recommender=self.unified_model,
            user_embed_dim=512,
            item_embed_dim=512,
            use_temporal=True,
            use_device=True,
            use_social=True,
            use_mood=True,
            context_fusion='attention'
        ).to(self.device)

    def _initialize_feature_extractors(self):
        """Initialize multimodal feature extractors"""

        logger.info("Setting up feature extractors...")

        # Text feature extractor
        self.text_extractor = TextFeatureExtractor(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            output_dim=512,
            freeze_base=False
        )

        # Metadata encoder
        self.metadata_encoder = MetadataEncoder(
            categorical_features={
                'genre': self.num_genres,
                'rating': 5,
                'language': 50
            },
            numerical_features=['year', 'popularity', 'runtime', 'vote_average'],
            embedding_dim=64,
            output_dim=512
        )

        # Multimodal fusion
        self.multimodal_fusion = MultimodalFusion(
            input_dims={
                'text': 512,
                'metadata': 512,
                'collaborative': 512
            },
            output_dim=768,
            fusion_type='attention'
        )

    def _initialize_production_components(self):
        """Initialize production optimization components"""

        logger.info("Setting up production components...")

        # Embedding cache
        self.embedding_cache = EmbeddingCache(
            cache_dir=self.cache_dir,
            embedding_dim=512,
            dtype=torch.float32
        )

        # Evaluator
        self.evaluator = ComprehensiveEvaluator(
            compute_accuracy=True,
            compute_diversity=True,
            compute_fairness=True,
            compute_business=True
        )

    def train(
        self,
        train_dataloader,
        val_dataloader,
        num_epochs: int = 10,
        use_contrastive: bool = True,
        use_multi_task: bool = True
    ):
        """
        Train the complete system.

        Args:
            train_dataloader: Training data
            val_dataloader: Validation data
            num_epochs: Number of epochs
            use_contrastive: Whether to use contrastive learning
            use_multi_task: Whether to use multi-task learning
        """

        logger.info(f"Starting training for {num_epochs} epochs...")

        # Choose model to train
        model_to_train = self.contrastive_model if use_contrastive else self.unified_model

        # Define tasks
        tasks = ['recommendation', 'rating', 'cross_domain']

        # Loss functions
        task_loss_fns = {
            'recommendation': nn.BCEWithLogitsLoss(),
            'rating': nn.MSELoss(),
            'cross_domain': nn.CrossEntropyLoss()
        }

        # Metric functions
        def compute_ndcg(preds, labels):
            from src.evaluation.comprehensive_metrics import AccuracyMetrics
            return AccuracyMetrics.ndcg(preds, labels, k=10)

        task_metric_fns = {
            'recommendation': compute_ndcg,
            'rating': lambda p, l: torch.sqrt(nn.MSELoss()(p, l)),
            'cross_domain': lambda p, l: (p.argmax(1) == l).float().mean()
        }

        # Create trainer
        trainer = AdvancedTrainer(
            model=model_to_train,
            tasks=tasks,
            device=self.device,
            output_dir=self.checkpoint_dir,
            base_lr=1e-3,
            weight_decay=0.01,
            use_curriculum=True,
            use_multi_task=use_multi_task,
            use_amp=True,
            gradient_accumulation_steps=4,
            max_grad_norm=1.0,
            warmup_steps=1000,
            early_stopping_patience=5
        )

        # Train
        trainer.train(
            train_dataloader=train_dataloader,
            eval_dataloader=val_dataloader,
            task_loss_fns=task_loss_fns,
            task_metric_fns=task_metric_fns,
            num_epochs=num_epochs
        )

        logger.info("Training complete!")

    def evaluate(
        self,
        test_dataloader,
        item_features: torch.Tensor,
        item_popularity: torch.Tensor,
        user_groups: Dict[int, int] = None
    ) -> Dict:
        """
        Comprehensive evaluation.

        Args:
            test_dataloader: Test data
            item_features: Item feature matrix
            item_popularity: Item popularity distribution
            user_groups: User demographic groups

        Returns:
            Dictionary of all metrics
        """

        logger.info("Running comprehensive evaluation...")

        # Get predictions
        all_predictions = []
        all_ground_truth = []

        self.unified_model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                outputs = self.unified_model(**batch)

                all_predictions.append(outputs)
                all_ground_truth.append(batch['labels'])

        predictions = torch.cat(all_predictions, dim=0)
        ground_truth = torch.cat(all_ground_truth, dim=0)

        # Evaluate
        results = self.evaluator.evaluate(
            predictions=predictions,
            ground_truth=ground_truth,
            item_features=item_features,
            item_popularity=item_popularity.cpu().numpy(),
            user_groups=user_groups,
            k_values=[5, 10, 20]
        )

        # Print results
        self.evaluator.print_results(results)

        return results

    def optimize_for_production(self):
        """
        Optimize models for production deployment.

        Returns:
            Optimized fast recommender
        """

        logger.info("Optimizing for production...")

        # 1. Quantize model
        logger.info("Quantizing model...")
        quantized_model = ModelQuantizer.dynamic_quantization(
            self.unified_model,
            dtype=torch.qint8
        )

        # 2. Create fast recommender with caching
        logger.info("Setting up fast recommender...")
        fast_recommender = FastRecommender(
            model=quantized_model,
            embedding_cache=self.embedding_cache,
            device=self.device,
            batch_size=32,
            use_ann=True  # Use FAISS for approximate nearest neighbor
        )

        logger.info("Production optimization complete!")

        return fast_recommender

    def recommend_movies(
        self,
        user_id: int,
        candidate_movie_ids: List[int],
        user_history: List[int] = None,
        context: Dict = None,
        top_k: int = 10,
        use_ensemble: bool = False,
        use_context: bool = True
    ) -> Dict:
        """
        Get movie recommendations for a user.

        Args:
            user_id: User ID
            candidate_movie_ids: Candidate movies to rank
            user_history: User's viewing history
            context: Context information (time, device, social, mood)
            top_k: Number of recommendations
            use_ensemble: Whether to use ensemble (slower but better)
            use_context: Whether to use context-aware model

        Returns:
            Dictionary with recommendations and metadata
        """

        candidates = torch.tensor(candidate_movie_ids).to(self.device)

        if use_ensemble:
            # Use ensemble (combines all models)
            logger.info("Using ensemble recommendation...")
            sequences = torch.tensor([user_history]).to(self.device) if user_history else None

            results = self.movie_ensemble.recommend_movies(
                user_id=user_id,
                candidate_movie_ids=candidates,
                sequences=sequences,
                top_k=top_k,
                min_confidence=0.5
            )

        elif use_context and context is not None:
            # Use context-aware model
            logger.info("Using context-aware recommendation...")

            top_items, scores = self.context_aware_model.recommend(
                user_id=user_id,
                candidate_items=candidates,
                context_data=context,
                top_k=top_k
            )

            results = {
                'recommended_ids': top_items.tolist(),
                'scores': scores.tolist()
            }

        else:
            # Use base unified model
            logger.info("Using unified model recommendation...")

            top_items, scores = self.unified_model.recommend(
                user_id=user_id,
                domain='movie',
                candidate_items=candidates,
                top_k=top_k,
                use_cross_domain=True  # Use TV history
            )

            results = {
                'recommended_ids': top_items.tolist(),
                'scores': scores.tolist()
            }

        return results

    def recommend_tv(
        self,
        user_id: int,
        candidate_tv_ids: List[int],
        context: Dict = None,
        top_k: int = 10
    ) -> Dict:
        """Get TV show recommendations"""

        candidates = torch.tensor(candidate_tv_ids).to(self.device)

        if context is not None:
            top_items, scores = self.context_aware_model.recommend(
                user_id=user_id,
                candidate_items=candidates,
                context_data=context,
                top_k=top_k
            )
        else:
            top_items, scores = self.unified_model.recommend(
                user_id=user_id,
                domain='tv',
                candidate_items=candidates,
                top_k=top_k,
                use_cross_domain=True  # Use movie history
            )

        return {
            'recommended_ids': top_items.tolist(),
            'scores': scores.tolist()
        }

    def save(self, path: Path):
        """Save complete system"""

        logger.info(f"Saving system to {path}...")

        torch.save({
            'unified_model': self.unified_model.state_dict(),
            'movie_ensemble': self.movie_ensemble.state_dict(),
            'context_aware_model': self.context_aware_model.state_dict(),
            'config': {
                'num_users': self.num_users,
                'num_movies': self.num_movies,
                'num_tv_shows': self.num_tv_shows,
                'num_genres': self.num_genres
            }
        }, path)

        logger.info("System saved!")

    def load(self, path: Path):
        """Load complete system"""

        logger.info(f"Loading system from {path}...")

        checkpoint = torch.load(path, map_location=self.device)

        self.unified_model.load_state_dict(checkpoint['unified_model'])
        self.movie_ensemble.load_state_dict(checkpoint['movie_ensemble'])
        self.context_aware_model.load_state_dict(checkpoint['context_aware_model'])

        logger.info("System loaded!")


# Example usage
def main():
    """Example of using the complete system"""

    # Initialize system
    system = CineSyncV2System(
        num_users=10000,
        num_movies=5000,
        num_tv_shows=3000,
        num_genres=20,
        cache_dir=Path('./cache'),
        checkpoint_dir=Path('./checkpoints')
    )

    # Example 1: Get movie recommendations with context
    logger.info("\n" + "="*80)
    logger.info("Example 1: Context-aware movie recommendations")
    logger.info("="*80)

    # Friday evening, watching on TV with family
    context = {
        'temporal': {
            'hour': torch.tensor([20]),  # 8 PM
            'day_of_week': torch.tensor([4]),  # Friday
            'month': torch.tensor([12])  # December
        },
        'device': {
            'device_type': torch.tensor([2]),  # TV
            'device_features': torch.tensor([[55.0, 0.0, 1.0, 1.0]])
        },
        'social': {
            'party_size': torch.tensor([4]),  # Family of 4
            'viewing_mode': torch.tensor([2])  # Family mode
        }
    }

    results = system.recommend_movies(
        user_id=42,
        candidate_movie_ids=list(range(100)),
        context=context,
        top_k=10,
        use_context=True
    )

    print(f"\nTop 10 movie recommendations:")
    print(f"Movie IDs: {results['recommended_ids']}")
    print(f"Scores: {results['scores']}")

    # Example 2: Get TV recommendations using movie history
    logger.info("\n" + "="*80)
    logger.info("Example 2: Cross-domain TV recommendations")
    logger.info("="*80)

    results = system.recommend_tv(
        user_id=42,
        candidate_tv_ids=list(range(100)),
        top_k=10
    )

    print(f"\nTop 10 TV recommendations (using movie history):")
    print(f"TV Show IDs: {results['recommended_ids']}")

    # Example 3: Ensemble recommendation
    logger.info("\n" + "="*80)
    logger.info("Example 3: Ensemble recommendation")
    logger.info("="*80)

    results = system.recommend_movies(
        user_id=42,
        candidate_movie_ids=list(range(100)),
        user_history=list(range(50)),  # Last 50 movies
        top_k=10,
        use_ensemble=True
    )

    print(f"\nEnsemble recommendations:")
    print(f"Movie IDs: {results['recommended_ids']}")
    print(f"Confidences: {results.get('confidences', 'N/A')}")

    # Save system
    system.save(Path('./checkpoints/complete_system.pt'))

    logger.info("\n" + "="*80)
    logger.info("Complete system demo finished!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
