#!/usr/bin/env python3
"""
Discord Notifications for CineSync v2 Training
Provides comprehensive training event notifications with rich embeds and statistics
"""

import requests
import time
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Configuration for training notifications."""
    model_name: str
    dataset_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str = "Adam"

class CineSyncDiscordAlerter:
    """Discord webhook notifications for CineSync training events."""
    
    def __init__(self, webhook_url: str = None, phone_number: str = None):
        self.webhook_url = webhook_url
        self.phone_number = phone_number
        self.discord_enabled = webhook_url is not None
        self.sms_enabled = phone_number is not None
        self.enabled = self.discord_enabled or self.sms_enabled
        
    def send_sms(self, message: str):
        """Send SMS notification using TextBelt service."""
        if not self.sms_enabled:
            return
            
        phone = ''.join(c for c in self.phone_number if c.isdigit() or c == '+')
        
        try:
            data = {
                'phone': phone,
                'message': f"CineSync: {message}",
                'key': 'textbelt'
            }
            
            response = requests.post('https://textbelt.com/text', data=data, timeout=10)
            if response.json().get('success'):
                print(f"📱 SMS sent: {message[:50]}...")
                return True
        except Exception as e:
            print(f"⚠️  SMS failed: {e}")
            
        return False
    
    def send_notification(self, title: str, description: str, color: int = 0x00ff00, fields: list = None, sms_message: str = None):
        """Send both Discord and SMS notifications."""
        if not self.enabled:
            return
        
        # Send Discord notification
        if self.discord_enabled:
            try:
                embed = {
                    "title": title,
                    "description": description,
                    "color": color,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    "footer": {"text": "CineSync v2 Training Bot"}
                }
                
                if fields:
                    embed["fields"] = fields
                
                payload = {
                    "embeds": [embed],
                    "username": "CineSync Training"
                }
                
                response = requests.post(self.webhook_url, json=payload, timeout=10)
                response.raise_for_status()
                print(f"🔔 Discord notification sent: {title}")
                
            except Exception as e:
                print(f"⚠️  Discord notification failed: {e}")
        
        # Send SMS notification
        if self.sms_enabled:
            if sms_message is None:
                sms_text = f"{title}: {description}"
                if fields and len(fields) > 0:
                    key_info = ", ".join([f"{f['name']}: {f['value']}" for f in fields[:2]])
                    sms_text += f" ({key_info})"
            else:
                sms_text = sms_message
                
            if len(sms_text) > 160:
                sms_text = sms_text[:157] + "..."
                
            self.send_sms(sms_text)
    
    def training_started(self, config: TrainingConfig, dataset_info: Dict[str, Any]):
        """Notify training start with comprehensive information."""
        fields = [
            {"name": "Model", "value": config.model_name, "inline": True},
            {"name": "Dataset", "value": config.dataset_name, "inline": True},
            {"name": "Epochs", "value": str(config.epochs), "inline": True},
            {"name": "Batch Size", "value": str(config.batch_size), "inline": True},
            {"name": "Learning Rate", "value": f"{config.learning_rate:.2e}", "inline": True},
            {"name": "Optimizer", "value": config.optimizer, "inline": True},
        ]
        
        # Add dataset statistics
        if dataset_info:
            total_samples = dataset_info.get('total_samples', 0)
            movie_samples = dataset_info.get('movie_samples', 0)
            tv_samples = dataset_info.get('tv_samples', 0)
            
            fields.extend([
                {"name": "Total Samples", "value": f"{total_samples:,}", "inline": True},
                {"name": "Movie Ratings", "value": f"{movie_samples:,}", "inline": True},
                {"name": "TV Show Ratings", "value": f"{tv_samples:,}", "inline": True},
            ])
        
        sms_msg = f"Training started: {config.model_name}, {config.epochs} epochs, {total_samples:,} samples"
        
        self.send_notification(
            title="🚀 CineSync Training Started",
            description="Recommendation model training has begun!",
            color=0x0099ff,
            fields=fields,
            sms_message=sms_msg
        )
    
    def training_progress(self, epoch: int, total_epochs: int, metrics: Dict[str, float], 
                         dataset_stats: Dict[str, Any] = None):
        """Notify training progress with metrics."""
        progress = (epoch / total_epochs) * 100
        
        fields = [
            {"name": "Progress", "value": f"{progress:.1f}% ({epoch}/{total_epochs})", "inline": True},
            {"name": "Train Loss", "value": f"{metrics.get('train_loss', 0):.4f}", "inline": True},
            {"name": "Val Loss", "value": f"{metrics.get('val_loss', 0):.4f}", "inline": True},
        ]
        
        # Add additional metrics if available
        if 'mse' in metrics:
            fields.append({"name": "MSE", "value": f"{metrics['mse']:.4f}", "inline": True})
        if 'mae' in metrics:
            fields.append({"name": "MAE", "value": f"{metrics['mae']:.4f}", "inline": True})
        if 'r2_score' in metrics:
            fields.append({"name": "R² Score", "value": f"{metrics['r2_score']:.4f}", "inline": True})
        
        # Add data pipeline performance
        if dataset_stats:
            samples_per_sec = dataset_stats.get('samples_per_sec', 0)
            if samples_per_sec > 0:
                fields.append({"name": "Data Speed", "value": f"{samples_per_sec:.1f} samples/sec", "inline": True})
        
        sms_msg = f"Training progress: {progress:.1f}% ({epoch}/{total_epochs}), Loss: {metrics.get('train_loss', 0):.4f}"
        
        self.send_notification(
            title="📊 Training Progress",
            description=f"Epoch {epoch} completed",
            color=0xffaa00,
            fields=fields,
            sms_message=sms_msg
        )
    
    def training_completed(self, duration_hours: float, final_metrics: Dict[str, float], 
                          model_path: str, best_metrics: Dict[str, float] = None):
        """Notify training completion with final results."""
        fields = [
            {"name": "Duration", "value": f"{duration_hours:.2f} hours", "inline": True},
            {"name": "Model Saved", "value": model_path, "inline": False},
        ]
        
        # Add final metrics
        if final_metrics:
            for metric, value in final_metrics.items():
                fields.append({
                    "name": metric.replace("_", " ").title(),
                    "value": f"{value:.4f}" if isinstance(value, float) else str(value),
                    "inline": True
                })
        
        # Add best metrics if available
        if best_metrics:
            fields.append({"name": "Best Validation Loss", "value": f"{best_metrics.get('best_val_loss', 0):.4f}", "inline": True})
            if 'best_r2_score' in best_metrics:
                fields.append({"name": "Best R² Score", "value": f"{best_metrics['best_r2_score']:.4f}", "inline": True})
        
        sms_msg = f"Training completed! Duration: {duration_hours:.2f}h"
        if final_metrics and 'train_loss' in final_metrics:
            sms_msg += f", Final loss: {final_metrics['train_loss']:.4f}"
        
        self.send_notification(
            title="✅ CineSync Training Completed",
            description="Recommendation model training finished successfully!",
            color=0x00ff00,
            fields=fields,
            sms_message=sms_msg
        )
    
    def training_error(self, error_msg: str, epoch: int = None, traceback: str = None):
        """Notify training error with details."""
        fields = []
        if epoch is not None:
            fields.append({"name": "Failed at Epoch", "value": str(epoch), "inline": True})
        
        # Truncate error message for Discord
        error_display = error_msg[:1000] + "..." if len(error_msg) > 1000 else error_msg
        fields.append({"name": "Error", "value": error_display, "inline": False})
        
        if traceback and len(traceback) < 500:
            fields.append({"name": "Traceback", "value": f"```{traceback}```", "inline": False})
        
        sms_msg = f"Training error"
        if epoch is not None:
            sms_msg += f" at epoch {epoch}"
        sms_msg += f": {error_msg[:100]}"
        
        self.send_notification(
            title="❌ Training Error",
            description="Training encountered an error!",
            color=0xff0000,
            fields=fields,
            sms_message=sms_msg
        )
    
    def data_pipeline_performance(self, stats: Dict[str, Any]):
        """Notify about data pipeline performance metrics."""
        samples_per_sec = stats.get('samples_per_sec', 0)
        
        # Only notify if performance is exceptional or problematic
        if samples_per_sec > 500:  # High performance
            color = 0x00ff88
            title = "🚀 High Performance Data Loading"
            description = f"Rust data loader achieving {samples_per_sec:.1f} samples/sec!"
        elif samples_per_sec < 10:  # Poor performance
            color = 0xff9900
            title = "⚠️ Slow Data Loading Detected"
            description = f"Data loading only {samples_per_sec:.1f} samples/sec - consider optimization"
        else:
            return  # Normal performance, no notification needed
        
        fields = [
            {"name": "Data Speed", "value": f"{samples_per_sec:.1f} samples/sec", "inline": True},
            {"name": "Total Samples", "value": f"{stats.get('total_samples', 0):,}", "inline": True},
            {"name": "Backend", "value": stats.get('backend', 'Unknown'), "inline": True},
        ]
        
        if 'movie_samples' in stats:
            fields.extend([
                {"name": "Movie Ratings", "value": f"{stats['movie_samples']:,}", "inline": True},
                {"name": "TV Show Ratings", "value": f"{stats.get('tv_samples', 0):,}", "inline": True},
            ])
        
        self.send_notification(
            title=title,
            description=description,
            color=color,
            fields=fields
        )
    
    def hardware_warning(self, warning_msg: str, metrics: Dict[str, float] = None):
        """Notify hardware warnings with system metrics."""
        fields = []
        
        if metrics:
            if 'gpu_memory_percent' in metrics:
                fields.append({"name": "GPU Memory", "value": f"{metrics['gpu_memory_percent']:.1f}%", "inline": True})
            if 'cpu_percent' in metrics:
                fields.append({"name": "CPU Usage", "value": f"{metrics['cpu_percent']:.1f}%", "inline": True})
            if 'memory_percent' in metrics:
                fields.append({"name": "RAM Usage", "value": f"{metrics['memory_percent']:.1f}%", "inline": True})
            if 'gpu_temperature' in metrics:
                fields.append({"name": "GPU Temp", "value": f"{metrics['gpu_temperature']:.1f}°C", "inline": True})
        
        sms_msg = f"Hardware warning: {warning_msg[:120]}"
        
        self.send_notification(
            title="⚠️ Hardware Warning",
            description=warning_msg,
            color=0xff9900,
            fields=fields,
            sms_message=sms_msg
        )
    
    def model_checkpoint_saved(self, epoch: int, metrics: Dict[str, float], checkpoint_path: str):
        """Notify when a model checkpoint is saved."""
        fields = [
            {"name": "Epoch", "value": str(epoch), "inline": True},
            {"name": "Checkpoint Path", "value": checkpoint_path, "inline": False},
        ]
        
        if metrics:
            for metric, value in list(metrics.items())[:4]:  # Limit to 4 metrics
                fields.append({
                    "name": metric.replace("_", " ").title(),
                    "value": f"{value:.4f}" if isinstance(value, float) else str(value),
                    "inline": True
                })
        
        # Only send SMS for significant checkpoints (every 10 epochs or best model)
        send_sms = epoch % 10 == 0 or 'best' in checkpoint_path.lower()
        sms_msg = f"Checkpoint saved at epoch {epoch}: {checkpoint_path}" if send_sms else None
        
        self.send_notification(
            title="💾 Model Checkpoint Saved",
            description=f"Model checkpoint saved at epoch {epoch}",
            color=0x9966ff,
            fields=fields,
            sms_message=sms_msg
        )
    
    def early_stopping_triggered(self, epoch: int, patience: int, best_metric: float, metric_name: str):
        """Notify when early stopping is triggered."""
        fields = [
            {"name": "Stopped at Epoch", "value": str(epoch), "inline": True},
            {"name": "Patience", "value": str(patience), "inline": True},
            {"name": f"Best {metric_name.title()}", "value": f"{best_metric:.4f}", "inline": True},
        ]
        
        sms_msg = f"Early stopping at epoch {epoch}. Best {metric_name}: {best_metric:.4f}"
        
        self.send_notification(
            title="🛑 Early Stopping Triggered",
            description=f"Training stopped early due to no improvement in {metric_name} for {patience} epochs",
            color=0xff6600,
            fields=fields,
            sms_message=sms_msg
        )

def test_notifications():
    """Test Discord notifications for CineSync."""
    # Replace with your Discord webhook URL
    webhook_url = "YOUR_DISCORD_WEBHOOK_URL_HERE"
    
    if webhook_url == "YOUR_DISCORD_WEBHOOK_URL_HERE":
        print("⚠️  Please set a valid Discord webhook URL in the test_notifications function")
        return
    
    alerter = CineSyncDiscordAlerter(webhook_url=webhook_url)
    
    # Test training started
    config = TrainingConfig(
        model_name="HybridRecommender",
        dataset_name="MovieLens-32M + TMDB TV",
        epochs=50,
        batch_size=256,
        learning_rate=0.001,
        optimizer="AdamW"
    )
    
    dataset_info = {
        'total_samples': 25000263,
        'movie_samples': 20000263,
        'tv_samples': 5000000,
    }
    
    print("Testing training started notification...")
    alerter.training_started(config, dataset_info)
    
    time.sleep(2)
    
    # Test training progress
    print("Testing training progress notification...")
    metrics = {
        'train_loss': 0.8234,
        'val_loss': 0.8567,
        'mse': 0.6789,
        'mae': 0.7123,
        'r2_score': 0.3456
    }
    
    dataset_stats = {'samples_per_sec': 342.5}
    alerter.training_progress(5, 50, metrics, dataset_stats)
    
    time.sleep(2)
    
    # Test training completed
    print("Testing training completed notification...")
    final_metrics = {
        'final_train_loss': 0.4123,
        'final_val_loss': 0.4356,
        'final_mse': 0.3456,
        'final_mae': 0.4567,
        'final_r2_score': 0.7890
    }
    
    best_metrics = {
        'best_val_loss': 0.4123,
        'best_r2_score': 0.8012
    }
    
    alerter.training_completed(
        duration_hours=12.5,
        final_metrics=final_metrics,
        model_path="./models/hybrid_recommender_v2",
        best_metrics=best_metrics
    )
    
    print("✅ Test notifications completed!")

if __name__ == "__main__":
    test_notifications()