"""
Meta-Learning TV Adapter (MLTA)
Fast adaptation to new genres, platforms, and user preferences using MAML
Optimized for RTX 4090
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import copy

class MetaLearningBlock(nn.Module):
    """Meta-learning block with fast adaptation capabilities"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_adaptation_steps: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_adaptation_steps = num_adaptation_steps
        
        # Main network
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Meta-parameters for fast adaptation
        self.meta_lr = nn.Parameter(torch.tensor(0.01))
        
    def forward(self, x: torch.Tensor, adaptation_data: Optional[Dict] = None) -> torch.Tensor:
        """Forward pass with optional adaptation"""
        if adaptation_data is not None:
            return self.fast_forward(x, adaptation_data)
        else:
            return self.network(x)
    
    def fast_forward(self, x: torch.Tensor, adaptation_data: Dict) -> torch.Tensor:
        """Fast forward pass with meta-learned parameters"""
        # Clone parameters for adaptation
        adapted_params = self.get_adapted_parameters(adaptation_data)
        
        # Forward pass with adapted parameters
        return self.functional_forward(x, adapted_params)
    
    def get_adapted_parameters(self, adaptation_data: Dict) -> Dict:
        """Get adapted parameters using gradient-based meta-learning"""
        support_x = adaptation_data['support_x']
        support_y = adaptation_data['support_y']
        
        # Clone current parameters
        adapted_params = {}
        for name, param in self.named_parameters():
            if 'network' in name:
                adapted_params[name] = param.clone()
        
        # Perform adaptation steps
        for step in range(self.num_adaptation_steps):
            # Forward pass with current adapted parameters
            predictions = self.functional_forward(support_x, adapted_params)
            loss = F.mse_loss(predictions, support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, adapted_params.values(), 
                                      create_graph=True, retain_graph=True)
            
            # Update adapted parameters
            for (name, param), grad in zip(adapted_params.items(), grads):
                adapted_params[name] = param - self.meta_lr * grad
        
        return adapted_params
    
    def functional_forward(self, x: torch.Tensor, params: Dict) -> torch.Tensor:
        """Functional forward pass using provided parameters"""
        # This is a simplified functional forward
        # In practice, you'd need to implement functional versions of all layers
        
        # For now, use the standard network (this would be replaced with functional implementation)
        return self.network(x)

class GenreAdapter(nn.Module):
    """Adapter for new genres with few-shot learning capabilities"""
    
    def __init__(self, base_dim: int = 512, adapter_dim: int = 128, num_genres: int = 50):
        super().__init__()
        self.base_dim = base_dim
        self.adapter_dim = adapter_dim
        
        # Base genre embeddings
        self.base_genre_embeddings = nn.Embedding(num_genres, base_dim, padding_idx=0)
        
        # Adapter network for new genres
        self.genre_adapter = nn.Sequential(
            nn.Linear(base_dim, adapter_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(adapter_dim, adapter_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(adapter_dim, base_dim)
        )
        
        # Meta-learning components
        self.meta_net = MetaLearningBlock(base_dim, adapter_dim, base_dim)
        
        # Genre similarity network
        self.similarity_net = nn.Sequential(
            nn.Linear(base_dim * 2, adapter_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(adapter_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, genre_ids: torch.Tensor, 
                adaptation_data: Optional[Dict] = None) -> torch.Tensor:
        """Forward pass with optional genre adaptation"""
        
        # Base embeddings
        base_embeddings = self.base_genre_embeddings(genre_ids)
        
        if adaptation_data is not None:
            # Adapt to new genres
            adapted_embeddings = self.meta_net(base_embeddings, adaptation_data)
            
            # Apply adapter
            adapter_output = self.genre_adapter(adapted_embeddings)
            
            # Combine base and adapted embeddings
            combined = base_embeddings + adapter_output
        else:
            combined = base_embeddings
        
        return combined
    
    def compute_genre_similarity(self, genre1: torch.Tensor, genre2: torch.Tensor) -> torch.Tensor:
        """Compute similarity between genres"""
        combined = torch.cat([genre1, genre2], dim=-1)
        return self.similarity_net(combined)

class PlatformAdapter(nn.Module):
    """Adapter for different streaming platforms"""
    
    def __init__(self, base_dim: int = 512, num_platforms: int = 20):
        super().__init__()
        self.base_dim = base_dim
        self.num_platforms = num_platforms
        
        # Platform-specific adapters
        self.platform_adapters = nn.ModuleDict({
            f'platform_{i}': nn.Sequential(
                nn.Linear(base_dim, base_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(base_dim // 2, base_dim),
                nn.LayerNorm(base_dim)
            ) for i in range(num_platforms)
        })
        
        # Meta-adapter for new platforms
        self.meta_adapter = MetaLearningBlock(base_dim, base_dim // 2, base_dim)
        
        # Platform feature extractor
        self.platform_features = nn.Sequential(
            nn.Linear(10, base_dim // 4),  # Platform statistics
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(base_dim // 4, base_dim)
        )
        
    def forward(self, features: torch.Tensor, platform_id: int,
                platform_stats: Optional[torch.Tensor] = None,
                adaptation_data: Optional[Dict] = None) -> torch.Tensor:
        """Adapt features for specific platform"""
        
        if platform_id < self.num_platforms and f'platform_{platform_id}' in self.platform_adapters:
            # Use pre-trained platform adapter
            adapted_features = self.platform_adapters[f'platform_{platform_id}'](features)
        else:
            # Use meta-adapter for new platform
            adapted_features = self.meta_adapter(features, adaptation_data)
        
        # Incorporate platform-specific statistics
        if platform_stats is not None:
            platform_embed = self.platform_features(platform_stats)
            adapted_features = adapted_features + platform_embed
        
        return adapted_features

class UserPreferenceAdapter(nn.Module):
    """Adapter for personalized user preferences"""
    
    def __init__(self, base_dim: int = 512, user_dim: int = 128):
        super().__init__()
        self.base_dim = base_dim
        self.user_dim = user_dim
        
        # User preference encoder
        self.preference_encoder = nn.Sequential(
            nn.Linear(20, user_dim),  # User preference features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(user_dim, user_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(user_dim, base_dim)
        )
        
        # Attention mechanism for preference weighting
        self.preference_attention = nn.MultiheadAttention(
            base_dim, num_heads=4, dropout=0.1, batch_first=True
        )
        
        # Meta-learning for quick adaptation to new users
        self.user_meta_net = MetaLearningBlock(base_dim, user_dim, base_dim)
        
        # Preference dynamics (temporal changes)
        self.preference_dynamics = nn.GRU(
            base_dim, user_dim, batch_first=True
        )
        
    def forward(self, features: torch.Tensor,
                user_preferences: torch.Tensor,
                user_history: Optional[torch.Tensor] = None,
                adaptation_data: Optional[Dict] = None) -> torch.Tensor:
        """Adapt features based on user preferences"""
        
        # Encode user preferences
        preference_embed = self.preference_encoder(user_preferences)
        
        # Handle temporal preference changes
        if user_history is not None:
            preference_dynamics, _ = self.preference_dynamics(user_history.unsqueeze(1))
            preference_embed = preference_embed + preference_dynamics.squeeze(1)
        
        # Apply attention to weight features by preferences
        features_expanded = features.unsqueeze(1)
        preference_expanded = preference_embed.unsqueeze(1)
        
        attended_features, attention_weights = self.preference_attention(
            features_expanded, preference_expanded, preference_expanded
        )
        
        adapted_features = attended_features.squeeze(1)
        
        # Meta-learning adaptation for new users
        if adaptation_data is not None:
            adapted_features = self.user_meta_net(adapted_features, adaptation_data)
        
        return adapted_features

class MetaLearningTVAdapter(nn.Module):
    """
    Complete Meta-Learning TV Adapter (MLTA)
    
    Features:
    - Fast adaptation to new genres using few-shot learning
    - Platform-specific optimization
    - Personalized user preference adaptation
    - Domain adaptation for different content types
    - Continual learning capabilities
    """
    
    def __init__(self,
                 base_feature_dim: int = 1024,
                 adapter_dim: int = 256,
                 num_genres: int = 50,
                 num_platforms: int = 20,
                 num_adaptation_steps: int = 5,
                 meta_lr: float = 0.01):
        super().__init__()
        
        self.base_feature_dim = base_feature_dim
        self.adapter_dim = adapter_dim
        self.num_adaptation_steps = num_adaptation_steps
        
        # Core adapters
        self.genre_adapter = GenreAdapter(
            base_feature_dim, adapter_dim, num_genres
        )
        
        self.platform_adapter = PlatformAdapter(
            base_feature_dim, num_platforms
        )
        
        self.user_preference_adapter = UserPreferenceAdapter(
            base_feature_dim, adapter_dim
        )
        
        # Meta-learning controller
        self.meta_controller = nn.Sequential(
            nn.Linear(base_feature_dim, adapter_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(adapter_dim, 3),  # Weights for different adapters
            nn.Softmax(dim=-1)
        )
        
        # Domain adaptation network
        self.domain_adapter = nn.Sequential(
            nn.Linear(base_feature_dim, adapter_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(adapter_dim, adapter_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(adapter_dim, base_feature_dim)
        )
        
        # Feature fusion network
        self.feature_fusion = nn.Sequential(
            nn.Linear(base_feature_dim * 2, base_feature_dim),
            nn.LayerNorm(base_feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(base_feature_dim, base_feature_dim)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(base_feature_dim, adapter_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(adapter_dim, base_feature_dim),
            nn.LayerNorm(base_feature_dim)
        )
        
        # Continual learning components
        self.memory_bank = nn.Parameter(torch.randn(1000, base_feature_dim))  # Episodic memory
        self.memory_attention = nn.MultiheadAttention(
            base_feature_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        
    def forward(self,
                base_features: torch.Tensor,
                genre_ids: Optional[torch.Tensor] = None,
                platform_id: Optional[int] = None,
                platform_stats: Optional[torch.Tensor] = None,
                user_preferences: Optional[torch.Tensor] = None,
                user_history: Optional[torch.Tensor] = None,
                adaptation_data: Optional[Dict] = None,
                domain_context: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with meta-learning adaptation
        
        Args:
            base_features: Base TV show features
            genre_ids: Genre IDs for adaptation
            platform_id: Platform identifier
            platform_stats: Platform-specific statistics
            user_preferences: User preference features
            user_history: User viewing history
            adaptation_data: Few-shot adaptation data
            domain_context: Domain context (e.g., 'new_genre', 'new_platform')
        """
        
        batch_size = base_features.size(0)
        adapted_features_list = []
        
        # Genre adaptation
        if genre_ids is not None:
            genre_adapted = self.genre_adapter(
                genre_ids, adaptation_data
            )
            # Aggregate genre features (mean pooling)
            if genre_adapted.dim() == 3:
                genre_adapted = genre_adapted.mean(dim=1)
            adapted_features_list.append(genre_adapted)
        
        # Platform adaptation
        if platform_id is not None:
            platform_adapted = self.platform_adapter(
                base_features, platform_id, platform_stats, adaptation_data
            )
            adapted_features_list.append(platform_adapted)
        
        # User preference adaptation
        if user_preferences is not None:
            user_adapted = self.user_preference_adapter(
                base_features, user_preferences, user_history, adaptation_data
            )
            adapted_features_list.append(user_adapted)
        
        # Domain adaptation
        domain_adapted = self.domain_adapter(base_features)
        
        # Meta-controller to weight different adaptations
        if adapted_features_list:
            # Stack adapted features
            stacked_features = torch.stack(adapted_features_list, dim=1)  # [batch, num_adapters, dim]
            
            # Get adapter weights from meta-controller
            controller_weights = self.meta_controller(base_features)  # [batch, num_adapters]
            
            # Weight and combine adapted features
            if controller_weights.size(1) != stacked_features.size(1):
                # Pad weights if needed
                weight_padding = stacked_features.size(1) - controller_weights.size(1)
                controller_weights = F.pad(controller_weights, (0, weight_padding), value=1.0/stacked_features.size(1))
                controller_weights = F.softmax(controller_weights, dim=-1)
            
            weighted_features = torch.sum(
                stacked_features * controller_weights.unsqueeze(-1), dim=1
            )
        else:
            weighted_features = base_features
        
        # Combine with domain adaptation
        combined_features = torch.cat([weighted_features, domain_adapted], dim=-1)
        fused_features = self.feature_fusion(combined_features)
        
        # Memory-based adaptation (continual learning)
        memory_features, memory_attention_weights = self.memory_attention(
            fused_features.unsqueeze(1),
            self.memory_bank.unsqueeze(0).expand(batch_size, -1, -1),
            self.memory_bank.unsqueeze(0).expand(batch_size, -1, -1)
        )
        
        # Final feature combination
        final_features = fused_features + memory_features.squeeze(1)
        
        # Output projection
        output_features = self.output_projection(final_features)
        
        return {
            'adapted_features': output_features,
            'base_features': base_features,
            'controller_weights': controller_weights if 'controller_weights' in locals() else None,
            'memory_attention': memory_attention_weights,
            'individual_adaptations': {
                'genre': adapted_features_list[0] if len(adapted_features_list) > 0 else None,
                'platform': adapted_features_list[1] if len(adapted_features_list) > 1 else None,
                'user': adapted_features_list[2] if len(adapted_features_list) > 2 else None,
                'domain': domain_adapted
            }
        }
    
    def fast_adapt(self, 
                   support_data: Dict[str, torch.Tensor],
                   query_data: Dict[str, torch.Tensor],
                   adaptation_type: str = 'genre') -> torch.Tensor:
        """Fast adaptation using few-shot learning"""
        
        # Prepare adaptation data
        adaptation_data = {
            'support_x': support_data['features'],
            'support_y': support_data['targets']
        }
        
        # Forward pass on query data with adaptation
        outputs = self.forward(
            query_data['features'],
            genre_ids=query_data.get('genre_ids'),
            platform_id=query_data.get('platform_id'),
            user_preferences=query_data.get('user_preferences'),
            adaptation_data=adaptation_data
        )
        
        return outputs['adapted_features']
    
    def update_memory(self, new_features: torch.Tensor, importance_weights: Optional[torch.Tensor] = None):
        """Update episodic memory with new features"""
        
        with torch.no_grad():
            if importance_weights is None:
                importance_weights = torch.ones(new_features.size(0), device=new_features.device)
            
            # Simple FIFO memory update
            # In practice, you'd implement more sophisticated memory management
            num_new = new_features.size(0)
            if num_new < self.memory_bank.size(0):
                # Shift existing memories
                self.memory_bank[num_new:] = self.memory_bank[:-num_new].clone()
                # Add new memories
                self.memory_bank[:num_new] = new_features
    
    def get_adaptation_loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """Compute adaptation-specific loss"""
        
        # Main adaptation loss
        main_loss = F.mse_loss(outputs['adapted_features'], targets)
        
        # Regularization losses
        reg_loss = 0
        
        # Encourage diversity in controller weights
        if outputs['controller_weights'] is not None:
            entropy = -torch.sum(outputs['controller_weights'] * torch.log(outputs['controller_weights'] + 1e-8), dim=1)
            reg_loss += 0.01 * entropy.mean()
        
        # Memory sparsity regularization
        if outputs['memory_attention'] is not None:
            sparsity_loss = torch.mean(torch.sum(outputs['memory_attention'] ** 2, dim=-1))
            reg_loss += 0.01 * sparsity_loss
        
        return main_loss + reg_loss

class MAMLTrainer:
    """Model-Agnostic Meta-Learning trainer for the TV adapter"""
    
    def __init__(self, model: MetaLearningTVAdapter, meta_lr: float = 0.001, inner_lr: float = 0.01):
        self.model = model
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
    
    def meta_train_step(self, task_batch: List[Dict]) -> float:
        """Perform one meta-training step"""
        
        meta_loss = 0
        
        for task in task_batch:
            # Clone model for inner loop
            task_model = copy.deepcopy(self.model)
            task_optimizer = torch.optim.SGD(task_model.parameters(), lr=self.inner_lr)
            
            # Inner loop: adapt to task
            support_data = task['support']
            for _ in range(5):  # Inner steps
                outputs = task_model.forward(
                    support_data['features'],
                    adaptation_data={
                        'support_x': support_data['features'],
                        'support_y': support_data['targets']
                    }
                )
                loss = task_model.get_adaptation_loss(outputs, support_data['targets'])
                
                task_optimizer.zero_grad()
                loss.backward()
                task_optimizer.step()
            
            # Outer loop: evaluate on query set
            query_data = task['query']
            query_outputs = task_model.forward(query_data['features'])
            query_loss = task_model.get_adaptation_loss(query_outputs, query_data['targets'])
            
            meta_loss += query_loss
        
        # Meta-optimization step
        meta_loss = meta_loss / len(task_batch)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()

# Configuration for RTX 4090 optimization
def get_meta_learning_config():
    """Optimized configuration for meta-learning adapter on RTX 4090"""
    return {
        'base_feature_dim': 1024,
        'adapter_dim': 256,
        'num_genres': 50,
        'num_platforms': 20,
        'num_adaptation_steps': 5,
        'meta_lr': 0.001,
        'inner_lr': 0.01,
        'batch_size': 32,  # Smaller for meta-learning
        'meta_batch_size': 8,  # Number of tasks per meta-batch
        'support_size': 5,  # Few-shot support size
        'query_size': 15,   # Query size per task
        'num_inner_steps': 5,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'use_mixed_precision': True,
        'gradient_clip': 1.0,
        'epochs': 50,
        'patience': 20,
    }