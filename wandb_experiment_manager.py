#!/usr/bin/env python3
"""
Wandb Experiment Management for CineSync v2
Comprehensive experiment tracking, comparison, and management utilities
"""

import wandb
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import yaml

from wandb_config import WandbManager, WandbConfig

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    name: str
    model_type: str
    hyperparameters: Dict[str, Any]
    dataset_config: Dict[str, Any]
    tags: List[str]
    notes: Optional[str] = None
    parent_experiment: Optional[str] = None


class ExperimentManager:
    """Manage and track experiments across multiple models"""
    
    def __init__(self, project_name: str = "cinesync-v2-experiments", 
                 entity: Optional[str] = None):
        """
        Initialize experiment manager
        
        Args:
            project_name: Wandb project name
            entity: Wandb entity (team/user)
        """
        self.project_name = project_name
        self.entity = entity
        self.api = wandb.Api()
        
        # Experiment registry
        self.experiments = {}
        self.active_sweeps = {}
        
        # Results cache
        self._results_cache = {}
        self._cache_timestamp = {}
    
    def create_experiment(self, config: ExperimentConfig, 
                         auto_start: bool = True) -> str:
        """
        Create and optionally start a new experiment
        
        Args:
            config: Experiment configuration
            auto_start: Whether to start the experiment immediately
            
        Returns:
            Experiment ID
        """
        
        # Create wandb configuration
        wandb_config = WandbConfig(
            project=self.project_name,
            entity=self.entity,
            name=config.name,
            tags=config.tags + [config.model_type],
            notes=config.notes,
            config={
                'model_type': config.model_type,
                'hyperparameters': config.hyperparameters,
                'dataset_config': config.dataset_config,
                'parent_experiment': config.parent_experiment
            }
        )
        
        # Store experiment config
        experiment_id = f"{config.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiments[experiment_id] = {
            'config': config,
            'wandb_config': wandb_config,
            'status': 'created',
            'created_at': datetime.now(),
            'run_id': None
        }
        
        logger.info(f"Created experiment: {experiment_id}")
        
        if auto_start:
            return self.start_experiment(experiment_id)
        
        return experiment_id
    
    def start_experiment(self, experiment_id: str) -> str:
        """Start an experiment"""
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        # Initialize wandb
        wandb_manager = WandbManager(experiment['wandb_config'])
        run = wandb_manager.init()
        
        if run:
            experiment['status'] = 'running'
            experiment['run_id'] = run.id
            experiment['wandb_manager'] = wandb_manager
            
            logger.info(f"Started experiment: {experiment_id} (run: {run.id})")
            return run.id
        else:
            experiment['status'] = 'failed'
            logger.error(f"Failed to start experiment: {experiment_id}")
            return None
    
    def finish_experiment(self, experiment_id: str, 
                         final_metrics: Optional[Dict[str, float]] = None):
        """Finish an experiment"""
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        if 'wandb_manager' in experiment:
            # Log final metrics
            if final_metrics:
                experiment['wandb_manager'].log_metrics({
                    f'final/{k}': v for k, v in final_metrics.items()
                })
            
            # Finish wandb run
            experiment['wandb_manager'].finish()
            
            experiment['status'] = 'completed'
            experiment['completed_at'] = datetime.now()
            
            logger.info(f"Finished experiment: {experiment_id}")
    
    def create_sweep(self, model_type: str, sweep_config: Dict[str, Any],
                    count: int = 20) -> str:
        """
        Create hyperparameter sweep
        
        Args:
            model_type: Type of model to sweep
            sweep_config: Wandb sweep configuration
            count: Number of runs in sweep
            
        Returns:
            Sweep ID
        """
        
        # Create sweep
        sweep_id = wandb.sweep(
            sweep_config,
            project=f"{self.project_name}-{model_type}-sweep",
            entity=self.entity
        )
        
        # Store sweep info
        self.active_sweeps[sweep_id] = {
            'model_type': model_type,
            'config': sweep_config,
            'count': count,
            'created_at': datetime.now(),
            'status': 'created'
        }
        
        logger.info(f"Created sweep: {sweep_id} for {model_type}")
        return sweep_id
    
    def start_sweep_agent(self, sweep_id: str, train_function: callable):
        """Start sweep agent"""
        
        if sweep_id not in self.active_sweeps:
            raise ValueError(f"Sweep {sweep_id} not found")
        
        sweep_info = self.active_sweeps[sweep_id]
        sweep_info['status'] = 'running'
        
        # Start agent
        wandb.agent(
            sweep_id,
            train_function,
            count=sweep_info['count'],
            project=f"{self.project_name}-{sweep_info['model_type']}-sweep",
            entity=self.entity
        )
        
        sweep_info['status'] = 'completed'
        logger.info(f"Completed sweep: {sweep_id}")
    
    def get_experiment_results(self, experiment_id: str, 
                              force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """Get results for a specific experiment"""
        
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        run_id = experiment.get('run_id')
        
        if not run_id:
            return None
        
        # Check cache
        cache_key = f"{self.project_name}/{run_id}"
        if not force_refresh and cache_key in self._results_cache:
            cache_time = self._cache_timestamp.get(cache_key, datetime.min)
            if datetime.now() - cache_time < timedelta(minutes=5):
                return self._results_cache[cache_key]
        
        try:
            # Fetch from wandb
            run = self.api.run(f"{self.entity}/{self.project_name}/{run_id}")
            
            results = {
                'id': run_id,
                'name': run.name,
                'state': run.state,
                'config': dict(run.config),
                'summary': dict(run.summary),
                'history': run.history().to_dict('records'),
                'created_at': run.created_at,
                'updated_at': run.updated_at,
                'runtime_seconds': (run.updated_at - run.created_at).total_seconds() if run.updated_at else None
            }
            
            # Cache results
            self._results_cache[cache_key] = results
            self._cache_timestamp[cache_key] = datetime.now()
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to fetch results for {experiment_id}: {e}")
            return None
    
    def compare_experiments(self, experiment_ids: List[str], 
                           metrics: List[str] = None) -> pd.DataFrame:
        """
        Compare multiple experiments
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metrics: List of metrics to compare (None for all)
            
        Returns:
            DataFrame with comparison results
        """
        
        if metrics is None:
            metrics = ['final/val_loss', 'final/val_rmse', 'final/train_loss', 'final/train_rmse']
        
        comparison_data = []
        
        for exp_id in experiment_ids:
            results = self.get_experiment_results(exp_id)
            
            if not results:
                continue
            
            row = {
                'experiment_id': exp_id,
                'name': results['name'],
                'state': results['state'],
                'runtime_minutes': results.get('runtime_seconds', 0) / 60,
                'model_type': results['config'].get('model_type', 'unknown')
            }
            
            # Add requested metrics
            for metric in metrics:
                # Try to get from summary first, then from final history
                value = results['summary'].get(metric)
                if value is None and results['history']:
                    # Look for metric in history
                    history_df = pd.DataFrame(results['history'])
                    if metric in history_df.columns:
                        value = history_df[metric].dropna().iloc[-1] if not history_df[metric].dropna().empty else None
                
                row[metric] = value
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def get_best_experiment(self, model_type: Optional[str] = None,
                           metric: str = 'final/val_rmse',
                           minimize: bool = True) -> Optional[str]:
        """
        Find best experiment by metric
        
        Args:
            model_type: Filter by model type (None for all)
            metric: Metric to optimize
            minimize: Whether to minimize or maximize metric
            
        Returns:
            Best experiment ID
        """
        
        # Get all experiments
        experiment_ids = list(self.experiments.keys())
        
        if model_type:
            experiment_ids = [
                exp_id for exp_id in experiment_ids
                if self.experiments[exp_id]['config'].model_type == model_type
            ]
        
        if not experiment_ids:
            return None
        
        # Compare experiments
        comparison_df = self.compare_experiments(experiment_ids, [metric])
        
        if comparison_df.empty or metric not in comparison_df.columns:
            return None
        
        # Filter out missing values
        valid_df = comparison_df.dropna(subset=[metric])
        
        if valid_df.empty:
            return None
        
        # Find best
        if minimize:
            best_idx = valid_df[metric].idxmin()
        else:
            best_idx = valid_df[metric].idxmax()
        
        return valid_df.loc[best_idx, 'experiment_id']
    
    def create_experiment_report(self, experiment_ids: List[str],
                               output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create comprehensive experiment report
        
        Args:
            experiment_ids: Experiments to include in report
            output_path: Path to save report (optional)
            
        Returns:
            Report dictionary
        """
        
        # Get comparison data
        comparison_df = self.compare_experiments(experiment_ids)
        
        # Calculate statistics
        report = {
            'summary': {
                'total_experiments': len(experiment_ids),
                'completed_experiments': len(comparison_df[comparison_df['state'] == 'finished']),
                'failed_experiments': len(comparison_df[comparison_df['state'] == 'failed']),
                'running_experiments': len(comparison_df[comparison_df['state'] == 'running']),
                'total_runtime_hours': comparison_df['runtime_minutes'].sum() / 60,
                'avg_runtime_minutes': comparison_df['runtime_minutes'].mean()
            },
            'model_breakdown': comparison_df.groupby('model_type').agg({
                'experiment_id': 'count',
                'runtime_minutes': ['mean', 'sum'],
                'final/val_rmse': ['mean', 'min', 'max'],
                'final/val_loss': ['mean', 'min', 'max']
            }).to_dict(),
            'best_experiments': {},
            'detailed_results': comparison_df.to_dict('records'),
            'generated_at': datetime.now().isoformat()
        }
        
        # Find best experiments by model type
        for model_type in comparison_df['model_type'].unique():
            best_exp = self.get_best_experiment(model_type, 'final/val_rmse', minimize=True)
            if best_exp:
                report['best_experiments'][model_type] = best_exp
        
        # Save report if requested
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report
    
    def visualize_experiment_comparison(self, experiment_ids: List[str],
                                      metrics: List[str] = None,
                                      save_path: Optional[str] = None):
        """Create visualization comparing experiments"""
        
        if metrics is None:
            metrics = ['final/val_rmse', 'final/train_rmse', 'runtime_minutes']
        
        # Get comparison data
        comparison_df = self.compare_experiments(experiment_ids, metrics)
        
        if comparison_df.empty:
            logger.warning("No data to visualize")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Experiment Comparison', fontsize=16)
        
        # Plot 1: Validation RMSE by model type
        if 'final/val_rmse' in comparison_df.columns:
            sns.boxplot(data=comparison_df, x='model_type', y='final/val_rmse', ax=axes[0, 0])
            axes[0, 0].set_title('Validation RMSE by Model Type')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Training vs Validation RMSE
        if 'final/train_rmse' in comparison_df.columns and 'final/val_rmse' in comparison_df.columns:
            scatter = axes[0, 1].scatter(
                comparison_df['final/train_rmse'], 
                comparison_df['final/val_rmse'],
                c=comparison_df['model_type'].astype('category').cat.codes,
                alpha=0.7
            )
            axes[0, 1].set_xlabel('Training RMSE')
            axes[0, 1].set_ylabel('Validation RMSE')
            axes[0, 1].set_title('Training vs Validation RMSE')
            axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        # Plot 3: Runtime by model type
        if 'runtime_minutes' in comparison_df.columns:
            sns.barplot(data=comparison_df, x='model_type', y='runtime_minutes', ax=axes[1, 0])
            axes[1, 0].set_title('Average Runtime by Model Type')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Performance vs Runtime
        if 'final/val_rmse' in comparison_df.columns and 'runtime_minutes' in comparison_df.columns:
            scatter = axes[1, 1].scatter(
                comparison_df['runtime_minutes'],
                comparison_df['final/val_rmse'],
                c=comparison_df['model_type'].astype('category').cat.codes,
                alpha=0.7
            )
            axes[1, 1].set_xlabel('Runtime (minutes)')
            axes[1, 1].set_ylabel('Validation RMSE')
            axes[1, 1].set_title('Performance vs Runtime')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def export_experiments(self, experiment_ids: List[str],
                          format: str = 'csv',
                          output_path: str = 'experiments_export') -> str:
        """
        Export experiment results
        
        Args:
            experiment_ids: Experiments to export
            format: Export format ('csv', 'json', 'yaml')
            output_path: Output file path (without extension)
            
        Returns:
            Path to exported file
        """
        
        # Get comparison data
        comparison_df = self.compare_experiments(experiment_ids)
        
        # Add full file extension
        if format == 'csv':
            output_file = f"{output_path}.csv"
            comparison_df.to_csv(output_file, index=False)
        elif format == 'json':
            output_file = f"{output_path}.json"
            comparison_df.to_json(output_file, orient='records', indent=2)
        elif format == 'yaml':
            output_file = f"{output_path}.yaml"
            data = comparison_df.to_dict('records')
            with open(output_file, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported {len(experiment_ids)} experiments to {output_file}")
        return output_file
    
    def cleanup_failed_experiments(self, days_old: int = 7):
        """Clean up failed experiments older than specified days"""
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        to_remove = []
        for exp_id, exp_data in self.experiments.items():
            if (exp_data.get('status') == 'failed' and 
                exp_data.get('created_at', datetime.now()) < cutoff_date):
                to_remove.append(exp_id)
        
        for exp_id in to_remove:
            del self.experiments[exp_id]
            logger.info(f"Cleaned up failed experiment: {exp_id}")
        
        logger.info(f"Cleaned up {len(to_remove)} failed experiments")


# Utility functions for common experiment patterns
def run_model_comparison_experiment(models: Dict[str, Any], 
                                  train_function: callable,
                                  base_config: Dict[str, Any]) -> ExperimentManager:
    """Run comparison experiment across multiple models"""
    
    manager = ExperimentManager()
    experiment_ids = []
    
    for model_name, model_config in models.items():
        # Create experiment config
        config = ExperimentConfig(
            name=f"{model_name}_comparison_{datetime.now().strftime('%Y%m%d')}",
            model_type=model_name,
            hyperparameters={**base_config, **model_config},
            dataset_config=base_config.get('dataset_config', {}),
            tags=['comparison', 'baseline']
        )
        
        # Create and start experiment
        exp_id = manager.create_experiment(config, auto_start=False)
        experiment_ids.append(exp_id)
        
        # Run training
        manager.start_experiment(exp_id)
        train_function(config.hyperparameters)
        manager.finish_experiment(exp_id)
    
    # Create comparison report
    report = manager.create_experiment_report(experiment_ids)
    
    logger.info(f"Model comparison completed. Best overall: {report['best_experiments']}")
    
    return manager


if __name__ == "__main__":
    # Example usage
    manager = ExperimentManager()
    
    # Create experiment
    config = ExperimentConfig(
        name="ncf_baseline_test",
        model_type="ncf",
        hyperparameters={
            'embedding_dim': 64,
            'hidden_layers': [128, 64],
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 10
        },
        dataset_config={'train_size': 0.8, 'val_size': 0.2},
        tags=['baseline', 'test']
    )
    
    exp_id = manager.create_experiment(config, auto_start=False)
    print(f"Created experiment: {exp_id}")
    
    # Example of getting experiment results
    # results = manager.get_experiment_results(exp_id)
    # print(f"Results: {results}")