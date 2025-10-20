#!/usr/bin/env python3
"""
Wandb Inference Monitoring for CineSync v2
Production monitoring for recommendation models in real-time
"""

import torch
import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import json
from pathlib import Path

from wandb_config import WandbManager, init_wandb_for_inference

logger = logging.getLogger(__name__)


class InferenceMetrics:
    """Track inference metrics for production monitoring"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.batch_sizes = deque(maxlen=window_size)
        self.error_count = 0
        self.total_requests = 0
        self.start_time = datetime.now()
        
        # Accuracy tracking
        self.predictions = deque(maxlen=window_size)
        self.ground_truth = deque(maxlen=window_size)
        
        # Model-specific metrics
        self.model_metrics = defaultdict(lambda: {
            'requests': 0,
            'latencies': deque(maxlen=window_size),
            'errors': 0
        })
    
    def record_inference(self, model_name: str, latency: float, batch_size: int, 
                        predictions: Optional[np.ndarray] = None, 
                        ground_truth: Optional[np.ndarray] = None):
        """Record a single inference request"""
        self.latencies.append(latency)
        self.batch_sizes.append(batch_size)
        self.total_requests += 1
        
        # Model-specific tracking
        self.model_metrics[model_name]['requests'] += 1
        self.model_metrics[model_name]['latencies'].append(latency)
        
        # Accuracy tracking if available
        if predictions is not None and ground_truth is not None:
            self.predictions.extend(predictions)
            self.ground_truth.extend(ground_truth)
    
    def record_error(self, model_name: str):
        """Record an inference error"""
        self.error_count += 1
        self.model_metrics[model_name]['errors'] += 1
    
    def get_summary_metrics(self) -> Dict[str, float]:
        """Get summary metrics for all models"""
        if not self.latencies:
            return {}
        
        uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        
        metrics = {
            'total_requests': self.total_requests,
            'error_rate': self.error_count / max(self.total_requests, 1),
            'avg_latency_ms': np.mean(self.latencies) * 1000,
            'p50_latency_ms': np.percentile(self.latencies, 50) * 1000,
            'p95_latency_ms': np.percentile(self.latencies, 95) * 1000,
            'p99_latency_ms': np.percentile(self.latencies, 99) * 1000,
            'throughput_per_hour': self.total_requests / max(uptime_hours, 0.01),
            'avg_batch_size': np.mean(self.batch_sizes),
            'uptime_hours': uptime_hours
        }
        
        # Add accuracy metrics if available
        if self.predictions and self.ground_truth:
            predictions = np.array(list(self.predictions))
            ground_truth = np.array(list(self.ground_truth))
            
            mse = np.mean((predictions - ground_truth) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - ground_truth))
            
            metrics.update({
                'rmse': rmse,
                'mae': mae,
                'mse': mse
            })
        
        return metrics
    
    def get_model_metrics(self, model_name: str) -> Dict[str, float]:
        """Get metrics for a specific model"""
        model_data = self.model_metrics[model_name]
        
        if not model_data['latencies']:
            return {'requests': model_data['requests'], 'errors': model_data['errors']}
        
        return {
            'requests': model_data['requests'],
            'errors': model_data['errors'],
            'error_rate': model_data['errors'] / max(model_data['requests'], 1),
            'avg_latency_ms': np.mean(model_data['latencies']) * 1000,
            'p95_latency_ms': np.percentile(model_data['latencies'], 95) * 1000
        }


class ProductionMonitor:
    """Production monitoring for recommendation inference"""
    
    def __init__(self, wandb_manager: WandbManager, log_interval: int = 300):
        """
        Initialize production monitor
        
        Args:
            wandb_manager: Initialized wandb manager
            log_interval: Interval in seconds to log metrics
        """
        self.wandb_manager = wandb_manager
        self.log_interval = log_interval
        self.metrics = InferenceMetrics()
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Alert thresholds
        self.alert_thresholds = {
            'error_rate': 0.05,  # 5% error rate
            'p95_latency_ms': 1000,  # 1 second
            'throughput_drop': 0.5  # 50% throughput drop
        }
        
        self.last_throughput = None
    
    def start_monitoring(self):
        """Start background monitoring thread"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Production monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Production monitoring stopped")
    
    def record_inference(self, model_name: str, latency: float, batch_size: int,
                        predictions: Optional[np.ndarray] = None,
                        ground_truth: Optional[np.ndarray] = None):
        """Record inference metrics"""
        self.metrics.record_inference(model_name, latency, batch_size, predictions, ground_truth)
    
    def record_error(self, model_name: str, error_message: str = None):
        """Record inference error"""
        self.metrics.record_error(model_name)
        
        if error_message:
            logger.error(f"Inference error in {model_name}: {error_message}")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            try:
                # Get current metrics
                summary_metrics = self.metrics.get_summary_metrics()
                
                if summary_metrics:
                    # Log to wandb
                    log_dict = {}
                    for key, value in summary_metrics.items():
                        log_dict[f'production/{key}'] = value
                    
                    # Add model-specific metrics
                    for model_name in self.metrics.model_metrics:
                        model_metrics = self.metrics.get_model_metrics(model_name)
                        for metric_name, metric_value in model_metrics.items():
                            log_dict[f'production/{model_name}/{metric_name}'] = metric_value
                    
                    self.wandb_manager.log_metrics(log_dict)
                    
                    # Check for alerts
                    self._check_alerts(summary_metrics)
                
                # Sleep until next log interval
                time.sleep(self.log_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Short sleep on error
    
    def _check_alerts(self, metrics: Dict[str, float]):
        """Check for alert conditions"""
        alerts = []
        
        # Error rate alert
        if metrics.get('error_rate', 0) > self.alert_thresholds['error_rate']:
            alerts.append(f"High error rate: {metrics['error_rate']:.3f}")
        
        # Latency alert
        if metrics.get('p95_latency_ms', 0) > self.alert_thresholds['p95_latency_ms']:
            alerts.append(f"High latency: {metrics['p95_latency_ms']:.1f}ms")
        
        # Throughput drop alert
        current_throughput = metrics.get('throughput_per_hour', 0)
        if self.last_throughput and current_throughput < self.last_throughput * self.alert_thresholds['throughput_drop']:
            alerts.append(f"Throughput drop: {current_throughput:.1f} (was {self.last_throughput:.1f})")
        
        self.last_throughput = current_throughput
        
        # Log alerts
        if alerts:
            alert_message = "; ".join(alerts)
            logger.warning(f"Production alerts: {alert_message}")
            
            # Log alert to wandb
            self.wandb_manager.log_metrics({
                'production/alert': 1,
                'production/alert_message': alert_message
            })


class ModelPerformanceTracker:
    """Track model performance across different contexts"""
    
    def __init__(self, wandb_manager: WandbManager):
        self.wandb_manager = wandb_manager
        self.performance_data = defaultdict(lambda: defaultdict(list))
        
        # Context categories
        self.contexts = {
            'user_type': ['new', 'returning', 'power_user'],
            'content_type': ['movie', 'tv', 'mixed'],
            'time_of_day': ['morning', 'afternoon', 'evening', 'night'],
            'recommendation_type': ['popular', 'similar', 'personalized', 'cross_content']
        }
    
    def record_recommendation_feedback(self, model_name: str, user_context: Dict[str, str],
                                     recommendation_quality: float, user_satisfaction: float):
        """Record feedback for recommendations"""
        
        # Store performance by context
        context_key = self._create_context_key(user_context)
        
        self.performance_data[model_name][context_key].extend([
            recommendation_quality, user_satisfaction
        ])
        
        # Log to wandb if we have enough data points
        if len(self.performance_data[model_name][context_key]) >= 10:
            self._log_context_performance(model_name, context_key)
    
    def _create_context_key(self, user_context: Dict[str, str]) -> str:
        """Create context key from user context"""
        context_parts = []
        for category, value in user_context.items():
            if category in self.contexts and value in self.contexts[category]:
                context_parts.append(f"{category}_{value}")
        
        return "_".join(sorted(context_parts)) if context_parts else "default"
    
    def _log_context_performance(self, model_name: str, context_key: str):
        """Log performance metrics for specific context"""
        data = self.performance_data[model_name][context_key]
        
        if len(data) < 2:
            return
        
        # Calculate metrics
        quality_scores = data[::2]  # Every even index
        satisfaction_scores = data[1::2]  # Every odd index
        
        metrics = {
            f'context_performance/{model_name}/{context_key}/avg_quality': np.mean(quality_scores),
            f'context_performance/{model_name}/{context_key}/avg_satisfaction': np.mean(satisfaction_scores),
            f'context_performance/{model_name}/{context_key}/quality_std': np.std(quality_scores),
            f'context_performance/{model_name}/{context_key}/satisfaction_std': np.std(satisfaction_scores),
            f'context_performance/{model_name}/{context_key}/sample_count': len(quality_scores)
        }
        
        self.wandb_manager.log_metrics(metrics)


class RecommendationMonitor:
    """Complete monitoring solution for recommendation systems"""
    
    def __init__(self, project_name: str = "cinesync-v2-production"):
        """Initialize recommendation monitoring"""
        
        # Initialize wandb for production monitoring
        self.wandb_manager = init_wandb_for_inference("production-monitor")
        
        # Initialize monitoring components
        self.production_monitor = ProductionMonitor(self.wandb_manager)
        self.performance_tracker = ModelPerformanceTracker(self.wandb_manager)
        
        # Model registry
        self.active_models = {}
        
    def register_model(self, model_name: str, model: torch.nn.Module, 
                      model_metadata: Dict[str, Any] = None):
        """Register a model for monitoring"""
        self.active_models[model_name] = {
            'model': model,
            'metadata': model_metadata or {},
            'registration_time': datetime.now()
        }
        
        # Log model registration
        self.wandb_manager.log_metrics({
            f'models/{model_name}/registered': 1,
            f'models/{model_name}/parameters': sum(p.numel() for p in model.parameters())
        })
        
        logger.info(f"Registered model {model_name} for monitoring")
    
    def start_monitoring(self):
        """Start production monitoring"""
        self.production_monitor.start_monitoring()
        logger.info("Recommendation monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.production_monitor.stop_monitoring()
        self.wandb_manager.finish()
        logger.info("Recommendation monitoring stopped")
    
    def monitor_inference(self, model_name: str, inference_function: callable):
        """Decorator to monitor inference functions"""
        
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                # Run inference
                result = inference_function(*args, **kwargs)
                
                # Calculate metrics
                latency = time.time() - start_time
                batch_size = self._estimate_batch_size(args, kwargs)
                
                # Record successful inference
                self.production_monitor.record_inference(
                    model_name, latency, batch_size
                )
                
                return result
                
            except Exception as e:
                # Record error
                self.production_monitor.record_error(model_name, str(e))
                raise
        
        return wrapper
    
    def _estimate_batch_size(self, args: tuple, kwargs: dict) -> int:
        """Estimate batch size from function arguments"""
        # Look for common batch size indicators
        for arg in args:
            if isinstance(arg, (list, tuple, np.ndarray)):
                return len(arg)
            elif isinstance(arg, torch.Tensor):
                return arg.size(0)
        
        for value in kwargs.values():
            if isinstance(value, (list, tuple, np.ndarray)):
                return len(value)
            elif isinstance(value, torch.Tensor):
                return value.size(0)
        
        return 1  # Default to single item
    
    def log_recommendation_feedback(self, model_name: str, user_id: int,
                                  recommended_items: List[int], user_rating: float,
                                  user_context: Dict[str, str] = None):
        """Log user feedback for recommendations"""
        
        # Calculate recommendation quality (placeholder - implement based on your metrics)
        recommendation_quality = self._calculate_recommendation_quality(
            user_id, recommended_items
        )
        
        # Record performance
        self.performance_tracker.record_recommendation_feedback(
            model_name, user_context or {}, recommendation_quality, user_rating
        )
        
        # Log individual feedback
        self.wandb_manager.log_metrics({
            f'feedback/{model_name}/user_rating': user_rating,
            f'feedback/{model_name}/recommendation_quality': recommendation_quality,
            f'feedback/{model_name}/num_items': len(recommended_items)
        })
    
    def _calculate_recommendation_quality(self, user_id: int, 
                                        recommended_items: List[int]) -> float:
        """Calculate recommendation quality score (implement based on your metrics)"""
        # Placeholder implementation
        # In practice, this would use metrics like diversity, novelty, coverage, etc.
        return np.random.uniform(0.5, 1.0)  # Replace with actual calculation
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        summary = {
            'production_metrics': self.production_monitor.metrics.get_summary_metrics(),
            'active_models': list(self.active_models.keys()),
            'monitoring_status': self.production_monitor.is_monitoring,
            'uptime_hours': (datetime.now() - self.production_monitor.metrics.start_time).total_seconds() / 3600
        }
        
        return summary


# Example usage context manager
class WandbInferenceContext:
    """Context manager for monitoring inference with wandb"""
    
    def __init__(self, monitor: RecommendationMonitor, model_name: str):
        self.monitor = monitor
        self.model_name = model_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Successful inference
            latency = time.time() - self.start_time
            self.monitor.production_monitor.record_inference(self.model_name, latency, 1)
        else:
            # Error occurred
            self.monitor.production_monitor.record_error(self.model_name, str(exc_val))


# Example usage
if __name__ == "__main__":
    # Initialize monitoring
    monitor = RecommendationMonitor()
    
    # Register a model (example)
    # monitor.register_model("ncf", ncf_model)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Example inference monitoring
    with WandbInferenceContext(monitor, "ncf"):
        # Your inference code here
        time.sleep(0.1)  # Simulated inference
        pass
    
    # Example feedback logging
    monitor.log_recommendation_feedback(
        "ncf", 
        user_id=123, 
        recommended_items=[1, 2, 3, 4, 5],
        user_rating=4.5,
        user_context={"user_type": "returning", "content_type": "movie"}
    )
    
    # Get summary
    summary = monitor.get_monitoring_summary()
    print(f"Monitoring summary: {summary}")
    
    # Stop monitoring
    monitor.stop_monitoring()