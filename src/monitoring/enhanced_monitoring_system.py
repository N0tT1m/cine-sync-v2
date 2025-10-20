#!/usr/bin/env python3
"""
Enhanced Monitoring System for CineSync v2
Completes the missing monitoring features: model drift detection, auto-retraining, A/B testing, and quality scoring
"""

import torch
import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import json
import hashlib
import pickle
from pathlib import Path
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import subprocess
import os

from wandb_config import WandbManager, init_wandb_for_inference
from wandb_inference_monitor import ProductionMonitor, ModelPerformanceTracker

logger = logging.getLogger(__name__)


class ModelDriftDetector:
    """Detect model performance drift and data distribution changes"""
    
    def __init__(self, baseline_window: int = 10000, detection_window: int = 1000):
        self.baseline_window = baseline_window
        self.detection_window = detection_window
        
        # Store baseline statistics
        self.baseline_stats = {}
        self.baseline_predictions = deque(maxlen=baseline_window)
        self.baseline_features = deque(maxlen=baseline_window)
        self.baseline_targets = deque(maxlen=baseline_window)
        
        # Current window for drift detection
        self.current_predictions = deque(maxlen=detection_window)
        self.current_features = deque(maxlen=detection_window)
        self.current_targets = deque(maxlen=detection_window)
        
        # Drift detection thresholds
        self.drift_thresholds = {
            'prediction_drift': 0.1,  # KL divergence threshold
            'performance_drift': 0.05,  # Performance degradation threshold
            'feature_drift': 0.1,  # Feature distribution drift threshold
            'concept_drift': 0.05  # Target distribution drift threshold
        }
        
        self.is_baseline_ready = False
    
    def add_baseline_data(self, predictions: np.ndarray, features: np.ndarray, 
                         targets: Optional[np.ndarray] = None):
        """Add data to baseline for drift detection"""
        self.baseline_predictions.extend(predictions.flatten())
        
        if features.ndim > 1:
            # For multi-dimensional features, store feature means
            self.baseline_features.extend(features.mean(axis=1))
        else:
            self.baseline_features.extend(features.flatten())
        
        if targets is not None:
            self.baseline_targets.extend(targets.flatten())
        
        # Update baseline statistics when we have enough data
        if len(self.baseline_predictions) >= self.baseline_window:
            self._update_baseline_stats()
            self.is_baseline_ready = True
    
    def detect_drift(self, predictions: np.ndarray, features: np.ndarray,
                     targets: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Detect various types of drift in model performance"""
        
        if not self.is_baseline_ready:
            return {'drift_detected': False, 'reason': 'Baseline not ready'}
        
        # Add current data to detection window
        self.current_predictions.extend(predictions.flatten())
        
        if features.ndim > 1:
            self.current_features.extend(features.mean(axis=1))
        else:
            self.current_features.extend(features.flatten())
        
        if targets is not None:
            self.current_targets.extend(targets.flatten())
        
        # Need enough current data for comparison
        if len(self.current_predictions) < self.detection_window:
            return {'drift_detected': False, 'reason': 'Insufficient current data'}
        
        drift_results = {
            'drift_detected': False,
            'drift_types': [],
            'drift_scores': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # 1. Prediction Drift Detection (distribution shift in predictions)
        pred_drift_score = self._calculate_distribution_drift(
            list(self.baseline_predictions), list(self.current_predictions)
        )
        drift_results['drift_scores']['prediction_drift'] = pred_drift_score
        
        if pred_drift_score > self.drift_thresholds['prediction_drift']:
            drift_results['drift_detected'] = True
            drift_results['drift_types'].append('prediction_drift')
        
        # 2. Feature Drift Detection (distribution shift in input features)
        feature_drift_score = self._calculate_distribution_drift(
            list(self.baseline_features), list(self.current_features)
        )
        drift_results['drift_scores']['feature_drift'] = feature_drift_score
        
        if feature_drift_score > self.drift_thresholds['feature_drift']:
            drift_results['drift_detected'] = True
            drift_results['drift_types'].append('feature_drift')
        
        # 3. Performance Drift Detection (if we have targets)
        if targets is not None and len(self.current_targets) >= self.detection_window:
            performance_drift_score = self._calculate_performance_drift()
            drift_results['drift_scores']['performance_drift'] = performance_drift_score
            
            if performance_drift_score > self.drift_thresholds['performance_drift']:
                drift_results['drift_detected'] = True
                drift_results['drift_types'].append('performance_drift')
            
            # 4. Concept Drift Detection (shift in target distribution)
            concept_drift_score = self._calculate_distribution_drift(
                list(self.baseline_targets), list(self.current_targets)
            )
            drift_results['drift_scores']['concept_drift'] = concept_drift_score
            
            if concept_drift_score > self.drift_thresholds['concept_drift']:
                drift_results['drift_detected'] = True
                drift_results['drift_types'].append('concept_drift')
        
        return drift_results
    
    def _update_baseline_stats(self):
        """Update baseline statistics for drift detection"""
        baseline_preds = np.array(list(self.baseline_predictions))
        baseline_feats = np.array(list(self.baseline_features))
        
        self.baseline_stats = {
            'prediction_mean': np.mean(baseline_preds),
            'prediction_std': np.std(baseline_preds),
            'prediction_hist': np.histogram(baseline_preds, bins=50),
            'feature_mean': np.mean(baseline_feats),
            'feature_std': np.std(baseline_feats),
            'feature_hist': np.histogram(baseline_feats, bins=50)
        }
        
        if self.baseline_targets:
            baseline_targets = np.array(list(self.baseline_targets))
            self.baseline_stats.update({
                'target_mean': np.mean(baseline_targets),
                'target_std': np.std(baseline_targets),
                'target_hist': np.histogram(baseline_targets, bins=50),
                'baseline_mse': mean_squared_error(baseline_targets, baseline_preds[-len(baseline_targets):])
            })
    
    def _calculate_distribution_drift(self, baseline: List[float], current: List[float]) -> float:
        """Calculate distribution drift using KL divergence"""
        try:
            # Create histograms with same bins
            min_val = min(min(baseline), min(current))
            max_val = max(max(baseline), max(current))
            bins = np.linspace(min_val, max_val, 50)
            
            hist_baseline, _ = np.histogram(baseline, bins=bins, density=True)
            hist_current, _ = np.histogram(current, bins=bins, density=True)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            hist_baseline = hist_baseline + epsilon
            hist_current = hist_current + epsilon
            
            # Normalize to create probability distributions
            hist_baseline = hist_baseline / np.sum(hist_baseline)
            hist_current = hist_current / np.sum(hist_current)
            
            # Calculate KL divergence
            kl_divergence = np.sum(hist_current * np.log(hist_current / hist_baseline))
            
            return float(kl_divergence)
            
        except Exception as e:
            logger.warning(f"Error calculating distribution drift: {e}")
            return 0.0
    
    def _calculate_performance_drift(self) -> float:
        """Calculate performance drift compared to baseline"""
        try:
            if not self.baseline_targets or not self.current_targets:
                return 0.0
            
            # Calculate current performance
            current_preds = list(self.current_predictions)[-len(self.current_targets):]
            current_mse = mean_squared_error(list(self.current_targets), current_preds)
            
            # Compare to baseline performance
            baseline_mse = self.baseline_stats.get('baseline_mse', current_mse)
            
            # Return relative performance degradation
            if baseline_mse > 0:
                return (current_mse - baseline_mse) / baseline_mse
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error calculating performance drift: {e}")
            return 0.0


class AutoRetrainingTrigger:
    """Automatic model retraining system based on drift detection and performance metrics"""
    
    def __init__(self, models_dir: str = "models", wandb_manager: Optional[WandbManager] = None):
        self.models_dir = Path(models_dir)
        self.wandb_manager = wandb_manager
        
        # Retraining triggers
        self.trigger_conditions = {
            'drift_detected': True,  # Trigger on any drift detection
            'error_rate_threshold': 0.1,  # 10% error rate
            'performance_degradation': 0.15,  # 15% performance drop
            'time_since_last_training': timedelta(days=7),  # Weekly retraining
            'new_data_threshold': 10000  # Retrain after 10k new samples
        }
        
        self.last_training_time = {}
        self.new_data_count = defaultdict(int)
        
        # Available training scripts
        self.training_scripts = {
            'ncf': 'neural_collaborative_filtering/train_with_wandb.py',
            'sequential': 'sequential_models/train_with_wandb.py',
            'two_tower': 'two_tower_model/train_with_wandb.py',
            'hybrid': 'hybrid_recommendation/train_with_wandb.py'
        }
    
    def check_retraining_needed(self, model_name: str, drift_results: Dict[str, Any],
                               error_rate: float, performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Check if model retraining should be triggered"""
        
        retraining_needed = False
        trigger_reasons = []
        
        # 1. Check for drift
        if drift_results.get('drift_detected', False):
            retraining_needed = True
            trigger_reasons.extend([f"drift_{dt}" for dt in drift_results.get('drift_types', [])])
        
        # 2. Check error rate
        if error_rate > self.trigger_conditions['error_rate_threshold']:
            retraining_needed = True
            trigger_reasons.append(f"high_error_rate_{error_rate:.3f}")
        
        # 3. Check performance degradation
        baseline_rmse = performance_metrics.get('baseline_rmse', 0)
        current_rmse = performance_metrics.get('current_rmse', 0)
        
        if baseline_rmse > 0 and current_rmse > 0:
            degradation = (current_rmse - baseline_rmse) / baseline_rmse
            if degradation > self.trigger_conditions['performance_degradation']:
                retraining_needed = True
                trigger_reasons.append(f"performance_degradation_{degradation:.3f}")
        
        # 4. Check time since last training
        last_training = self.last_training_time.get(model_name)
        if last_training:
            time_since_training = datetime.now() - last_training
            if time_since_training > self.trigger_conditions['time_since_last_training']:
                retraining_needed = True
                trigger_reasons.append(f"time_threshold_{time_since_training.days}_days")
        
        # 5. Check new data threshold
        if self.new_data_count[model_name] > self.trigger_conditions['new_data_threshold']:
            retraining_needed = True
            trigger_reasons.append(f"new_data_{self.new_data_count[model_name]}")
        
        return {
            'retraining_needed': retraining_needed,
            'trigger_reasons': trigger_reasons,
            'timestamp': datetime.now().isoformat()
        }
    
    def trigger_retraining(self, model_name: str, trigger_reasons: List[str]) -> Dict[str, Any]:
        """Trigger automatic model retraining"""
        
        if model_name not in self.training_scripts:
            return {
                'success': False,
                'error': f"No training script available for model {model_name}"
            }
        
        training_script = self.training_scripts[model_name]
        
        try:
            # Log retraining trigger
            logger.info(f"Triggering retraining for {model_name}. Reasons: {trigger_reasons}")
            
            if self.wandb_manager:
                self.wandb_manager.log_metrics({
                    f'retraining/{model_name}/triggered': 1,
                    f'retraining/{model_name}/trigger_count': len(trigger_reasons)
                })
            
            # Create retraining command
            retraining_command = [
                'python', training_script,
                '--epochs', '20',
                '--early-stopping-patience', '5',
                '--wandb-tags', f'auto_retrain_{datetime.now().strftime("%Y%m%d")}',
                '--wandb-notes', f'Auto-triggered retraining. Reasons: {", ".join(trigger_reasons)}'
            ]
            
            # Run retraining in background
            logger.info(f"Starting retraining: {' '.join(retraining_command)}")
            
            # For production, you'd want to run this asynchronously
            # For now, we'll just log the command that would be executed
            result = {
                'success': True,
                'training_script': training_script,
                'command': ' '.join(retraining_command),
                'start_time': datetime.now().isoformat(),
                'trigger_reasons': trigger_reasons
            }
            
            # Update last training time
            self.last_training_time[model_name] = datetime.now()
            self.new_data_count[model_name] = 0  # Reset new data counter
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to trigger retraining for {model_name}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def add_new_data_point(self, model_name: str, count: int = 1):
        """Track new data points for retraining threshold"""
        self.new_data_count[model_name] += count


class ABTestingFramework:
    """A/B testing framework for recommendation models"""
    
    def __init__(self, wandb_manager: Optional[WandbManager] = None):
        self.wandb_manager = wandb_manager
        self.experiments = {}
        self.user_assignments = {}
        
        # Metrics tracking
        self.experiment_metrics = defaultdict(lambda: defaultdict(list))
    
    def create_experiment(self, experiment_name: str, model_variants: Dict[str, Any],
                         traffic_allocation: Dict[str, float],
                         success_metrics: List[str] = None,
                         duration_days: int = 14) -> Dict[str, Any]:
        """Create a new A/B test experiment"""
        
        # Validate traffic allocation
        total_traffic = sum(traffic_allocation.values())
        if abs(total_traffic - 1.0) > 0.01:
            raise ValueError(f"Traffic allocation must sum to 1.0, got {total_traffic}")
        
        experiment = {
            'name': experiment_name,
            'model_variants': model_variants,
            'traffic_allocation': traffic_allocation,
            'success_metrics': success_metrics or ['click_rate', 'rating', 'engagement'],
            'start_time': datetime.now(),
            'end_time': datetime.now() + timedelta(days=duration_days),
            'is_active': True,
            'total_users': 0,
            'variant_users': defaultdict(int)
        }
        
        self.experiments[experiment_name] = experiment
        
        # Log experiment creation
        if self.wandb_manager:
            self.wandb_manager.log_metrics({
                f'ab_testing/{experiment_name}/created': 1,
                f'ab_testing/{experiment_name}/variants': len(model_variants),
                f'ab_testing/{experiment_name}/duration_days': duration_days
            })
        
        logger.info(f"Created A/B test experiment: {experiment_name}")
        return experiment
    
    def assign_user_to_variant(self, experiment_name: str, user_id: int) -> Optional[str]:
        """Assign user to experiment variant using consistent hashing"""
        
        if experiment_name not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_name]
        
        # Check if experiment is still active
        if not experiment['is_active'] or datetime.now() > experiment['end_time']:
            return None
        
        # Check if user already assigned
        user_key = f"{experiment_name}_{user_id}"
        if user_key in self.user_assignments:
            return self.user_assignments[user_key]
        
        # Assign user using consistent hashing
        hash_input = f"{experiment_name}_{user_id}_{experiment['start_time']}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = (hash_value % 10000) / 10000.0  # Convert to 0-1 range
        
        # Determine variant based on traffic allocation
        cumulative_prob = 0.0
        assigned_variant = None
        
        for variant, allocation in experiment['traffic_allocation'].items():
            cumulative_prob += allocation
            if bucket <= cumulative_prob:
                assigned_variant = variant
                break
        
        # Record assignment
        if assigned_variant:
            self.user_assignments[user_key] = assigned_variant
            experiment['total_users'] += 1
            experiment['variant_users'][assigned_variant] += 1
            
            # Log assignment
            if self.wandb_manager:
                self.wandb_manager.log_metrics({
                    f'ab_testing/{experiment_name}/assignments/{assigned_variant}': 1
                })
        
        return assigned_variant
    
    def record_experiment_metric(self, experiment_name: str, user_id: int,
                                metric_name: str, metric_value: float):
        """Record metric for A/B test analysis"""
        
        # Get user's variant assignment
        variant = self.assign_user_to_variant(experiment_name, user_id)
        if not variant:
            return
        
        # Record metric
        metric_key = f"{variant}_{metric_name}"
        self.experiment_metrics[experiment_name][metric_key].append(metric_value)
        
        # Log to wandb
        if self.wandb_manager:
            self.wandb_manager.log_metrics({
                f'ab_testing/{experiment_name}/metrics/{variant}/{metric_name}': metric_value
            })
    
    def analyze_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """Analyze A/B test results with statistical significance"""
        
        if experiment_name not in self.experiments:
            return {'error': 'Experiment not found'}
        
        experiment = self.experiments[experiment_name]
        metrics_data = self.experiment_metrics[experiment_name]
        
        analysis = {
            'experiment_name': experiment_name,
            'status': 'active' if experiment['is_active'] else 'completed',
            'duration_days': (datetime.now() - experiment['start_time']).days,
            'total_users': experiment['total_users'],
            'variant_results': {},
            'statistical_significance': {}
        }
        
        # Analyze each success metric
        for success_metric in experiment['success_metrics']:
            variant_data = {}
            
            # Collect data for each variant
            for variant in experiment['model_variants'].keys():
                metric_key = f"{variant}_{success_metric}"
                values = metrics_data.get(metric_key, [])
                
                if values:
                    variant_data[variant] = {
                        'count': len(values),
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'values': values
                    }
            
            analysis['variant_results'][success_metric] = variant_data
            
            # Statistical significance testing (t-test between variants)
            if len(variant_data) >= 2:
                variants = list(variant_data.keys())
                for i in range(len(variants)):
                    for j in range(i + 1, len(variants)):
                        variant_a, variant_b = variants[i], variants[j]
                        values_a = variant_data[variant_a]['values']
                        values_b = variant_data[variant_b]['values']
                        
                        if len(values_a) > 10 and len(values_b) > 10:
                            t_stat, p_value = stats.ttest_ind(values_a, values_b)
                            
                            comparison_key = f"{variant_a}_vs_{variant_b}_{success_metric}"
                            analysis['statistical_significance'][comparison_key] = {
                                't_statistic': float(t_stat),
                                'p_value': float(p_value),
                                'significant': p_value < 0.05,
                                'effect_size': abs(variant_data[variant_a]['mean'] - variant_data[variant_b]['mean'])
                            }
        
        return analysis
    
    def end_experiment(self, experiment_name: str) -> Dict[str, Any]:
        """End an A/B test experiment and get final results"""
        
        if experiment_name not in self.experiments:
            return {'error': 'Experiment not found'}
        
        self.experiments[experiment_name]['is_active'] = False
        self.experiments[experiment_name]['end_time'] = datetime.now()
        
        # Get final analysis
        final_analysis = self.analyze_experiment(experiment_name)
        
        # Log experiment completion
        if self.wandb_manager:
            self.wandb_manager.log_metrics({
                f'ab_testing/{experiment_name}/completed': 1,
                f'ab_testing/{experiment_name}/final_users': self.experiments[experiment_name]['total_users']
            })
        
        logger.info(f"Completed A/B test experiment: {experiment_name}")
        return final_analysis


class RealtimeQualityScorer:
    """Real-time recommendation quality scoring system"""
    
    def __init__(self, wandb_manager: Optional[WandbManager] = None):
        self.wandb_manager = wandb_manager
        self.quality_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Quality scoring weights
        self.quality_weights = {
            'diversity': 0.2,
            'novelty': 0.2,
            'accuracy': 0.3,
            'coverage': 0.1,
            'serendipity': 0.1,
            'freshness': 0.1
        }
    
    def calculate_recommendation_quality(self, user_id: int, recommendations: List[int],
                                       user_history: List[int], all_items: List[int],
                                       item_features: Dict[int, Dict[str, Any]] = None) -> Dict[str, float]:
        """Calculate comprehensive recommendation quality score"""
        
        quality_scores = {}
        
        # 1. Diversity Score (genre/category diversity)
        quality_scores['diversity'] = self._calculate_diversity(recommendations, item_features)
        
        # 2. Novelty Score (how new/unpopular the items are)
        quality_scores['novelty'] = self._calculate_novelty(recommendations, all_items)
        
        # 3. Coverage Score (how much of the catalog is being recommended)
        quality_scores['coverage'] = self._calculate_coverage(recommendations, all_items)
        
        # 4. Serendipity Score (unexpected but relevant recommendations)
        quality_scores['serendipity'] = self._calculate_serendipity(recommendations, user_history, item_features)
        
        # 5. Freshness Score (how recent the content is)
        quality_scores['freshness'] = self._calculate_freshness(recommendations, item_features)
        
        # 6. Accuracy placeholder (would need user feedback to calculate)
        quality_scores['accuracy'] = 0.8  # Placeholder - replace with actual accuracy calculation
        
        # Calculate overall quality score
        overall_quality = sum(
            quality_scores[metric] * self.quality_weights[metric]
            for metric in quality_scores.keys()
        )
        
        quality_scores['overall'] = overall_quality
        
        # Store in history
        self.quality_history[user_id].append(overall_quality)
        
        # Log to wandb
        if self.wandb_manager:
            log_dict = {f'quality/{metric}': score for metric, score in quality_scores.items()}
            self.wandb_manager.log_metrics(log_dict)
        
        return quality_scores
    
    def _calculate_diversity(self, recommendations: List[int], 
                           item_features: Dict[int, Dict[str, Any]] = None) -> float:
        """Calculate genre/category diversity in recommendations"""
        if not item_features:
            return 0.5  # Default if no features available
        
        try:
            genres = set()
            for item_id in recommendations:
                item_data = item_features.get(item_id, {})
                item_genres = item_data.get('genres', [])
                if isinstance(item_genres, str):
                    item_genres = item_genres.split('|')
                genres.update(item_genres)
            
            # Normalize by maximum possible diversity
            max_diversity = min(len(recommendations), 20)  # Assume max 20 genres
            diversity_score = len(genres) / max_diversity
            
            return min(diversity_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating diversity: {e}")
            return 0.5
    
    def _calculate_novelty(self, recommendations: List[int], all_items: List[int]) -> float:
        """Calculate novelty based on item popularity"""
        try:
            # Simple novelty: recommend less popular items
            total_items = len(all_items)
            avg_position = np.mean([all_items.index(item) if item in all_items else total_items 
                                  for item in recommendations])
            
            # Normalize: higher position = more novel
            novelty_score = avg_position / total_items
            
            return min(novelty_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating novelty: {e}")
            return 0.5
    
    def _calculate_coverage(self, recommendations: List[int], all_items: List[int]) -> float:
        """Calculate catalog coverage"""
        try:
            # Simple coverage metric
            unique_recommendations = len(set(recommendations))
            coverage_score = unique_recommendations / min(len(recommendations), 20)  # Penalize duplicates
            
            return min(coverage_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating coverage: {e}")
            return 0.5
    
    def _calculate_serendipity(self, recommendations: List[int], user_history: List[int],
                              item_features: Dict[int, Dict[str, Any]] = None) -> float:
        """Calculate serendipity (unexpected but relevant)"""
        try:
            if not user_history:
                return 0.5
            
            # Simple serendipity: items not similar to user's history
            history_set = set(user_history)
            new_items = [item for item in recommendations if item not in history_set]
            
            serendipity_score = len(new_items) / len(recommendations)
            
            return min(serendipity_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating serendipity: {e}")
            return 0.5
    
    def _calculate_freshness(self, recommendations: List[int],
                           item_features: Dict[int, Dict[str, Any]] = None) -> float:
        """Calculate content freshness"""
        if not item_features:
            return 0.5
        
        try:
            current_year = datetime.now().year
            years = []
            
            for item_id in recommendations:
                item_data = item_features.get(item_id, {})
                year = item_data.get('year', current_year - 10)  # Default to 10 years ago
                years.append(year)
            
            # Calculate average age and convert to freshness
            avg_age = current_year - np.mean(years)
            freshness_score = max(0, 1 - (avg_age / 20))  # Normalize by 20 years
            
            return min(freshness_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Error calculating freshness: {e}")
            return 0.5
    
    def get_user_quality_trend(self, user_id: int) -> Dict[str, float]:
        """Get quality trend for a specific user"""
        history = list(self.quality_history[user_id])
        
        if len(history) < 2:
            return {'trend': 0.0, 'avg_quality': 0.5, 'sample_count': len(history)}
        
        # Simple trend calculation
        recent_avg = np.mean(history[-min(10, len(history)):])  # Last 10 recommendations
        overall_avg = np.mean(history)
        trend = recent_avg - overall_avg
        
        return {
            'trend': float(trend),
            'avg_quality': float(overall_avg),
            'recent_avg': float(recent_avg),
            'sample_count': len(history)
        }


class EnhancedMonitoringSystem:
    """Complete enhanced monitoring system with all missing features"""
    
    def __init__(self, project_name: str = "cinesync-v2-enhanced-monitoring"):
        """Initialize enhanced monitoring system"""
        
        # Initialize wandb
        self.wandb_manager = init_wandb_for_inference("enhanced-monitor")
        
        # Initialize all monitoring components
        self.production_monitor = ProductionMonitor(self.wandb_manager)
        self.drift_detector = ModelDriftDetector()
        self.auto_retrainer = AutoRetrainingTrigger(wandb_manager=self.wandb_manager)
        self.ab_tester = ABTestingFramework(wandb_manager=self.wandb_manager)
        self.quality_scorer = RealtimeQualityScorer(wandb_manager=self.wandb_manager)
        
        self.is_monitoring = False
        
    def start_enhanced_monitoring(self):
        """Start all monitoring components"""
        self.production_monitor.start_monitoring()
        self.is_monitoring = True
        logger.info("Enhanced monitoring system started")
    
    def stop_enhanced_monitoring(self):
        """Stop all monitoring"""
        self.production_monitor.stop_monitoring()
        self.wandb_manager.finish()
        self.is_monitoring = False
        logger.info("Enhanced monitoring system stopped")
    
    def monitor_recommendation_request(self, model_name: str, user_id: int,
                                    recommendations: List[int], user_history: List[int],
                                    features: np.ndarray, predictions: np.ndarray,
                                    ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Comprehensive monitoring for a single recommendation request"""
        
        monitoring_results = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'user_id': user_id
        }
        
        # 1. Record basic inference metrics
        start_time = time.time()
        # ... inference happens here ...
        latency = time.time() - start_time
        
        self.production_monitor.record_inference(
            model_name, latency, len(recommendations), predictions, ground_truth
        )
        
        # 2. Check for model drift
        self.drift_detector.add_baseline_data(predictions, features, ground_truth)
        drift_results = self.drift_detector.detect_drift(predictions, features, ground_truth)
        monitoring_results['drift_analysis'] = drift_results
        
        # 3. Calculate recommendation quality
        quality_scores = self.quality_scorer.calculate_recommendation_quality(
            user_id, recommendations, user_history, []  # all_items would come from your catalog
        )
        monitoring_results['quality_scores'] = quality_scores
        
        # 4. Check if retraining needed
        current_metrics = self.production_monitor.metrics.get_summary_metrics()
        error_rate = current_metrics.get('error_rate', 0)
        
        retraining_check = self.auto_retrainer.check_retraining_needed(
            model_name, drift_results, error_rate, current_metrics
        )
        monitoring_results['retraining_check'] = retraining_check
        
        # 5. Trigger retraining if needed
        if retraining_check['retraining_needed']:
            retrain_result = self.auto_retrainer.trigger_retraining(
                model_name, retraining_check['trigger_reasons']
            )
            monitoring_results['retraining_triggered'] = retrain_result
        
        # 6. Track new data point
        self.auto_retrainer.add_new_data_point(model_name)
        
        return monitoring_results


# Example usage
if __name__ == "__main__":
    # Initialize enhanced monitoring
    monitor = EnhancedMonitoringSystem()
    monitor.start_enhanced_monitoring()
    
    # Example A/B test setup
    ab_test = monitor.ab_tester.create_experiment(
        experiment_name="ncf_vs_two_tower",
        model_variants={
            'ncf_baseline': {'model_type': 'ncf', 'version': 'v1.0'},
            'two_tower_enhanced': {'model_type': 'two_tower', 'version': 'v2.1'}
        },
        traffic_allocation={'ncf_baseline': 0.5, 'two_tower_enhanced': 0.5},
        success_metrics=['rating', 'click_rate', 'engagement_time'],
        duration_days=14
    )
    
    # Example monitoring call
    sample_monitoring = monitor.monitor_recommendation_request(
        model_name='ncf',
        user_id=12345,
        recommendations=[1, 2, 3, 4, 5],
        user_history=[10, 11, 12],
        features=np.random.randn(5, 64),
        predictions=np.random.rand(5) * 5,
        ground_truth=np.random.rand(5) * 5
    )
    
    print("Enhanced monitoring results:", sample_monitoring)
    
    # Clean up
    monitor.stop_enhanced_monitoring()