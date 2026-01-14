"""
Comprehensive Evaluation Framework for CineSync v2
Measures accuracy, diversity, fairness, and business metrics
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class AccuracyMetrics:
    """
    Standard recommendation accuracy metrics.

    - Hit Rate (HR): Did we recommend at least one relevant item?
    - NDCG: Normalized Discounted Cumulative Gain (position-aware)
    - MRR: Mean Reciprocal Rank
    - Precision/Recall: Standard classification metrics
    - MAP: Mean Average Precision
    """

    @staticmethod
    def hit_rate(predictions: torch.Tensor, ground_truth: torch.Tensor, k: int = 10) -> float:
        """
        Hit Rate@K: Percentage of users with at least one relevant item in top-k.

        Args:
            predictions: Predicted rankings [num_users, num_items]
            ground_truth: Binary relevance [num_users, num_items]
            k: Top-k to consider

        Returns:
            Hit rate score
        """
        # Get top-k predictions
        _, top_k_indices = torch.topk(predictions, k, dim=1)

        # Check if any top-k items are relevant
        hits = torch.zeros(predictions.size(0))
        for i in range(predictions.size(0)):
            top_k_items = top_k_indices[i]
            relevant_items = torch.where(ground_truth[i] > 0)[0]
            hits[i] = float(len(set(top_k_items.tolist()) & set(relevant_items.tolist())) > 0)

        return hits.mean().item()

    @staticmethod
    def ndcg(predictions: torch.Tensor, ground_truth: torch.Tensor, k: int = 10) -> float:
        """
        Normalized Discounted Cumulative Gain@K.

        Measures ranking quality, giving more weight to relevant items at top positions.

        Args:
            predictions: Predicted rankings [num_users, num_items]
            ground_truth: Relevance scores [num_users, num_items]
            k: Top-k to consider

        Returns:
            NDCG score
        """
        # Get top-k predictions
        _, top_k_indices = torch.topk(predictions, k, dim=1)

        ndcg_scores = []
        for i in range(predictions.size(0)):
            # Relevance of predicted items
            predicted_relevance = ground_truth[i, top_k_indices[i]]

            # DCG (Discounted Cumulative Gain)
            dcg = torch.sum(
                predicted_relevance / torch.log2(torch.arange(2, k + 2, dtype=torch.float))
            )

            # Ideal DCG (if we ranked perfectly)
            ideal_relevance, _ = torch.sort(ground_truth[i], descending=True)
            idcg = torch.sum(
                ideal_relevance[:k] / torch.log2(torch.arange(2, k + 2, dtype=torch.float))
            )

            # NDCG
            if idcg > 0:
                ndcg_scores.append((dcg / idcg).item())
            else:
                ndcg_scores.append(0.0)

        return np.mean(ndcg_scores)

    @staticmethod
    def mrr(predictions: torch.Tensor, ground_truth: torch.Tensor) -> float:
        """
        Mean Reciprocal Rank.

        Average of 1/rank of first relevant item.

        Args:
            predictions: Predicted rankings [num_users, num_items]
            ground_truth: Binary relevance [num_users, num_items]

        Returns:
            MRR score
        """
        # Sort predictions
        _, sorted_indices = torch.sort(predictions, dim=1, descending=True)

        reciprocal_ranks = []
        for i in range(predictions.size(0)):
            relevant_items = torch.where(ground_truth[i] > 0)[0]

            if len(relevant_items) == 0:
                reciprocal_ranks.append(0.0)
                continue

            # Find rank of first relevant item
            for rank, item_idx in enumerate(sorted_indices[i], start=1):
                if item_idx in relevant_items:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)

        return np.mean(reciprocal_ranks)

    @staticmethod
    def precision_recall(
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
        k: int = 10
    ) -> Tuple[float, float]:
        """
        Precision and Recall@K.

        Args:
            predictions: Predicted rankings [num_users, num_items]
            ground_truth: Binary relevance [num_users, num_items]
            k: Top-k to consider

        Returns:
            (precision, recall)
        """
        _, top_k_indices = torch.topk(predictions, k, dim=1)

        precisions = []
        recalls = []

        for i in range(predictions.size(0)):
            top_k_items = set(top_k_indices[i].tolist())
            relevant_items = set(torch.where(ground_truth[i] > 0)[0].tolist())

            if len(relevant_items) == 0:
                continue

            # Precision: relevant items in top-k / k
            true_positives = len(top_k_items & relevant_items)
            precision = true_positives / k
            precisions.append(precision)

            # Recall: relevant items in top-k / total relevant
            recall = true_positives / len(relevant_items)
            recalls.append(recall)

        return np.mean(precisions), np.mean(recalls)

    @staticmethod
    def map_score(predictions: torch.Tensor, ground_truth: torch.Tensor, k: int = 10) -> float:
        """
        Mean Average Precision@K.

        Args:
            predictions: Predicted rankings [num_users, num_items]
            ground_truth: Binary relevance [num_users, num_items]
            k: Top-k to consider

        Returns:
            MAP score
        """
        _, top_k_indices = torch.topk(predictions, k, dim=1)

        average_precisions = []

        for i in range(predictions.size(0)):
            relevant_items = set(torch.where(ground_truth[i] > 0)[0].tolist())

            if len(relevant_items) == 0:
                continue

            # Calculate precision at each relevant item position
            precisions_at_k = []
            num_hits = 0

            for rank, item_idx in enumerate(top_k_indices[i].tolist(), start=1):
                if item_idx in relevant_items:
                    num_hits += 1
                    precision_at_rank = num_hits / rank
                    precisions_at_k.append(precision_at_rank)

            if precisions_at_k:
                average_precisions.append(np.mean(precisions_at_k))

        return np.mean(average_precisions) if average_precisions else 0.0


class DiversityMetrics:
    """
    Diversity metrics to prevent filter bubbles.

    - Catalog Coverage: What % of catalog gets recommended?
    - Diversity@K: How diverse are the recommendations?
    - Intra-list Distance: Average distance between recommended items
    - Novelty: How often do we recommend long-tail items?
    """

    @staticmethod
    def catalog_coverage(
        all_recommendations: List[List[int]],
        total_items: int
    ) -> float:
        """
        What percentage of the catalog gets recommended to at least one user?

        Args:
            all_recommendations: List of recommendation lists for all users
            total_items: Total number of items in catalog

        Returns:
            Coverage percentage
        """
        recommended_items = set()
        for recs in all_recommendations:
            recommended_items.update(recs)

        return len(recommended_items) / total_items

    @staticmethod
    def diversity_at_k(
        recommendations: torch.Tensor,
        item_features: torch.Tensor,
        k: int = 10
    ) -> float:
        """
        Average pairwise distance between items in recommendation list.

        Args:
            recommendations: Recommended item indices [num_users, k]
            item_features: Item feature matrix [num_items, feature_dim]
            k: Number of recommendations

        Returns:
            Average diversity
        """
        diversities = []

        for user_recs in recommendations:
            if len(user_recs) < 2:
                continue

            # Get features for recommended items
            rec_features = item_features[user_recs[:k]]

            # Compute pairwise distances
            distances = torch.cdist(rec_features, rec_features, p=2)

            # Average distance (excluding diagonal)
            mask = ~torch.eye(len(rec_features), dtype=torch.bool)
            avg_distance = distances[mask].mean().item()

            diversities.append(avg_distance)

        return np.mean(diversities)

    @staticmethod
    def novelty(
        recommendations: List[List[int]],
        item_popularity: np.ndarray
    ) -> float:
        """
        How novel are recommendations? (based on item popularity)

        Higher score = more long-tail recommendations

        Args:
            recommendations: List of recommendation lists
            item_popularity: Popularity score for each item

        Returns:
            Average novelty
        """
        novelties = []

        for recs in recommendations:
            # Novelty = -log2(popularity)
            rec_popularities = item_popularity[recs]
            rec_novelties = -np.log2(rec_popularities + 1e-10)
            novelties.append(rec_novelties.mean())

        return np.mean(novelties)


class FairnessMetrics:
    """
    Fairness metrics to ensure equitable recommendations.

    - Popularity Bias: Do we over-recommend popular items?
    - User Fairness: Do all user groups get good recommendations?
    - Item Fairness: Do all items get fair exposure?
    - Demographic Parity: Equal performance across demographics
    """

    @staticmethod
    def popularity_bias(
        recommendations: List[List[int]],
        item_popularity: np.ndarray
    ) -> float:
        """
        Measure bias toward popular items.

        Lower score = less bias (better)

        Args:
            recommendations: List of recommendation lists
            item_popularity: Popularity score for each item

        Returns:
            Popularity bias score
        """
        all_rec_items = []
        for recs in recommendations:
            all_rec_items.extend(recs)

        # Average popularity of recommended items
        rec_popularity = np.mean(item_popularity[all_rec_items])

        # Compare to average popularity of all items
        overall_popularity = np.mean(item_popularity)

        # Bias = how much we over-recommend popular items
        bias = rec_popularity / (overall_popularity + 1e-10)

        return bias

    @staticmethod
    def user_fairness(
        user_metrics: Dict[int, float],
        user_groups: Dict[int, int]
    ) -> float:
        """
        Measure performance disparity across user groups.

        Lower score = fairer (better)

        Args:
            user_metrics: Dict of {user_id: metric_value}
            user_groups: Dict of {user_id: group_id}

        Returns:
            Fairness score (coefficient of variation)
        """
        # Group metrics by user group
        group_metrics = defaultdict(list)
        for user_id, metric in user_metrics.items():
            group_id = user_groups.get(user_id, 0)
            group_metrics[group_id].append(metric)

        # Average metric per group
        group_averages = [np.mean(metrics) for metrics in group_metrics.values()]

        # Coefficient of variation (std / mean)
        if len(group_averages) > 1:
            cv = np.std(group_averages) / (np.mean(group_averages) + 1e-10)
            return cv
        else:
            return 0.0

    @staticmethod
    def gini_coefficient(values: np.ndarray) -> float:
        """
        Gini coefficient for measuring inequality.

        0 = perfect equality, 1 = perfect inequality

        Args:
            values: Array of values (e.g., item exposure counts)

        Returns:
            Gini coefficient
        """
        sorted_values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)

        return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n


class BusinessMetrics:
    """
    Business-oriented metrics.

    - CTR: Click-through rate
    - Engagement Time: Total watch time
    - Completion Rate: How often do users finish recommended content?
    - Retention: Do recommendations keep users coming back?
    - Revenue: Conversion to premium, rentals, etc.
    """

    @staticmethod
    def ctr(clicks: np.ndarray, impressions: np.ndarray) -> float:
        """Click-through rate"""
        return np.sum(clicks) / (np.sum(impressions) + 1e-10)

    @staticmethod
    def engagement_time(watch_times: np.ndarray) -> float:
        """Average engagement time per user"""
        return np.mean(watch_times)

    @staticmethod
    def completion_rate(
        watch_times: np.ndarray,
        content_durations: np.ndarray
    ) -> float:
        """What percentage of content do users watch?"""
        completion_rates = watch_times / (content_durations + 1e-10)
        completion_rates = np.clip(completion_rates, 0, 1)
        return np.mean(completion_rates)

    @staticmethod
    def retention_rate(
        user_active_days: Dict[int, List[int]],
        window_days: int = 7
    ) -> float:
        """
        What percentage of users return within N days?

        Args:
            user_active_days: Dict of {user_id: [list of active day indices]}
            window_days: Retention window

        Returns:
            Retention rate
        """
        retained_users = 0
        total_users = len(user_active_days)

        for user_id, active_days in user_active_days.items():
            if len(active_days) < 2:
                continue

            # Check if user returned within window
            gaps = np.diff(sorted(active_days))
            if np.any(gaps <= window_days):
                retained_users += 1

        return retained_users / (total_users + 1e-10)


class ComprehensiveEvaluator:
    """
    Complete evaluation framework that computes all metrics.
    """

    def __init__(
        self,
        compute_accuracy: bool = True,
        compute_diversity: bool = True,
        compute_fairness: bool = True,
        compute_business: bool = True
    ):
        self.compute_accuracy = compute_accuracy
        self.compute_diversity = compute_diversity
        self.compute_fairness = compute_fairness
        self.compute_business = compute_business

        self.accuracy_metrics = AccuracyMetrics()
        self.diversity_metrics = DiversityMetrics()
        self.fairness_metrics = FairnessMetrics()
        self.business_metrics = BusinessMetrics()

    def evaluate(
        self,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
        item_features: Optional[torch.Tensor] = None,
        item_popularity: Optional[np.ndarray] = None,
        user_groups: Optional[Dict] = None,
        business_data: Optional[Dict] = None,
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, Union[float, Dict]]:
        """
        Run comprehensive evaluation.

        Args:
            predictions: Model predictions [num_users, num_items]
            ground_truth: Ground truth relevance [num_users, num_items]
            item_features: Item features for diversity computation
            item_popularity: Item popularity for fairness/diversity
            user_groups: User group assignments for fairness
            business_data: Business metrics data
            k_values: List of k values to evaluate

        Returns:
            Dictionary of all metrics
        """
        results = {}

        # Accuracy metrics
        if self.compute_accuracy:
            accuracy_results = {}

            for k in k_values:
                accuracy_results[f'hit_rate@{k}'] = self.accuracy_metrics.hit_rate(
                    predictions, ground_truth, k
                )
                accuracy_results[f'ndcg@{k}'] = self.accuracy_metrics.ndcg(
                    predictions, ground_truth, k
                )
                precision, recall = self.accuracy_metrics.precision_recall(
                    predictions, ground_truth, k
                )
                accuracy_results[f'precision@{k}'] = precision
                accuracy_results[f'recall@{k}'] = recall
                accuracy_results[f'map@{k}'] = self.accuracy_metrics.map_score(
                    predictions, ground_truth, k
                )

            accuracy_results['mrr'] = self.accuracy_metrics.mrr(predictions, ground_truth)

            results['accuracy'] = accuracy_results

        # Diversity metrics
        if self.compute_diversity and item_features is not None:
            diversity_results = {}

            for k in k_values:
                # Get top-k recommendations
                _, top_k_indices = torch.topk(predictions, k, dim=1)

                diversity_results[f'diversity@{k}'] = self.diversity_metrics.diversity_at_k(
                    top_k_indices, item_features, k
                )

            # Catalog coverage
            all_recs = []
            for k in k_values:
                _, top_k = torch.topk(predictions, k, dim=1)
                all_recs.append([rec.tolist() for rec in top_k])

            diversity_results['catalog_coverage'] = self.diversity_metrics.catalog_coverage(
                all_recs[-1], predictions.size(1)
            )

            # Novelty
            if item_popularity is not None:
                diversity_results['novelty'] = self.diversity_metrics.novelty(
                    all_recs[-1], item_popularity
                )

            results['diversity'] = diversity_results

        # Fairness metrics
        if self.compute_fairness and item_popularity is not None:
            fairness_results = {}

            _, top_k = torch.topk(predictions, k_values[-1], dim=1)
            recs = [rec.tolist() for rec in top_k]

            fairness_results['popularity_bias'] = self.fairness_metrics.popularity_bias(
                recs, item_popularity
            )

            # Item exposure Gini
            item_counts = np.bincount(
                np.concatenate(recs),
                minlength=predictions.size(1)
            )
            fairness_results['exposure_gini'] = self.fairness_metrics.gini_coefficient(
                item_counts
            )

            # User fairness
            if user_groups is not None:
                user_ndcgs = {}
                for i in range(predictions.size(0)):
                    user_ndcgs[i] = self.accuracy_metrics.ndcg(
                        predictions[i:i+1], ground_truth[i:i+1], k_values[0]
                    )

                fairness_results['user_fairness'] = self.fairness_metrics.user_fairness(
                    user_ndcgs, user_groups
                )

            results['fairness'] = fairness_results

        # Business metrics
        if self.compute_business and business_data is not None:
            business_results = {}

            if 'clicks' in business_data and 'impressions' in business_data:
                business_results['ctr'] = self.business_metrics.ctr(
                    business_data['clicks'], business_data['impressions']
                )

            if 'watch_times' in business_data:
                business_results['engagement_time'] = self.business_metrics.engagement_time(
                    business_data['watch_times']
                )

            if 'watch_times' in business_data and 'durations' in business_data:
                business_results['completion_rate'] = self.business_metrics.completion_rate(
                    business_data['watch_times'], business_data['durations']
                )

            if 'user_active_days' in business_data:
                business_results['retention_rate'] = self.business_metrics.retention_rate(
                    business_data['user_active_days']
                )

            results['business'] = business_results

        return results

    def print_results(self, results: Dict):
        """Pretty print evaluation results"""
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*80)

        for category, metrics in results.items():
            print(f"\n{category.upper()} METRICS:")
            print("-" * 80)

            for metric_name, value in metrics.items():
                print(f"  {metric_name:.<50} {value:.4f}")

        print("\n" + "="*80)


# Example usage
if __name__ == "__main__":
    # Simulate data
    num_users = 100
    num_items = 1000

    # Predictions and ground truth
    predictions = torch.randn(num_users, num_items)
    ground_truth = torch.randint(0, 2, (num_users, num_items), dtype=torch.float)

    # Item features for diversity
    item_features = torch.randn(num_items, 128)

    # Item popularity
    item_popularity = np.random.exponential(1.0, num_items)
    item_popularity = item_popularity / item_popularity.sum()

    # User groups
    user_groups = {i: i % 5 for i in range(num_users)}

    # Business data
    business_data = {
        'clicks': np.random.binomial(1, 0.1, num_users),
        'impressions': np.ones(num_users) * 10,
        'watch_times': np.random.exponential(30, num_users),
        'durations': np.ones(num_users) * 90
    }

    # Evaluate
    evaluator = ComprehensiveEvaluator()
    results = evaluator.evaluate(
        predictions=predictions,
        ground_truth=ground_truth,
        item_features=item_features,
        item_popularity=item_popularity,
        user_groups=user_groups,
        business_data=business_data
    )

    evaluator.print_results(results)
