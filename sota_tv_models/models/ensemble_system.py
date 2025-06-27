"""
Ensemble System for State-of-the-Art TV Models
Combines Multimodal Transformer, GNN, and Contrastive Learning models
Optimized for RTX 4090 inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import json
import pickle

from .multimodal_transformer import MultimodalTransformerTV
from .graph_neural_network import TVGraphRecommender
from .contrastive_learning import ContrastiveTVModel

logger = logging.getLogger(__name__)

class AdaptiveWeightingModule(nn.Module):
    """Learn adaptive weights for ensemble models based on input characteristics"""
    
    def __init__(self, 
                 input_dim: int = 768,
                 num_models: int = 3,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.num_models = num_models
        
        # Context encoder for adaptive weighting
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_models),
            nn.Softmax(dim=-1)
        )
        
        # Temperature parameter for sharpening/smoothing weights
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, context_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive weights based on context
        
        Args:
            context_embedding: [batch_size, input_dim] context representation
            
        Returns:
            weights: [batch_size, num_models] adaptive weights
        """
        # Compute base weights
        logits = self.context_encoder(context_embedding)
        
        # Apply temperature scaling
        weights = F.softmax(logits / self.temperature, dim=-1)
        
        return weights

class UncertaintyEstimator(nn.Module):
    """Estimate prediction uncertainty for ensemble models"""
    
    def __init__(self, 
                 embed_dim: int = 768,
                 hidden_dim: int = 256,
                 num_models: int = 3):
        super().__init__()
        
        self.num_models = num_models
        
        # Uncertainty estimation network
        self.uncertainty_net = nn.Sequential(
            nn.Linear(embed_dim * num_models, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_models),
            nn.Sigmoid()  # Output uncertainty scores [0, 1]
        )
        
    def forward(self, model_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Estimate uncertainty for each model's predictions
        
        Args:
            model_embeddings: List of [batch_size, embed_dim] embeddings from each model
            
        Returns:
            uncertainties: [batch_size, num_models] uncertainty scores
        """
        # Concatenate embeddings from all models
        concat_embeddings = torch.cat(model_embeddings, dim=-1)
        
        # Estimate uncertainties
        uncertainties = self.uncertainty_net(concat_embeddings)
        
        return uncertainties

class TVEnsembleRecommender(nn.Module):
    """
    Complete ensemble system for TV recommendation
    
    Combines:
    1. Multimodal Transformer (text + metadata)
    2. Graph Neural Network (relationships)
    3. Contrastive Learning (self-supervised)
    
    Features:
    - Adaptive weighting based on input characteristics
    - Uncertainty estimation for prediction confidence
    - Multi-level fusion (embedding + prediction level)
    - Cold start handling
    """
    
    def __init__(self,
                 multimodal_model: MultimodalTransformerTV,
                 gnn_model: TVGraphRecommender,
                 contrastive_model: ContrastiveTVModel,
                 embed_dim: int = 768,
                 use_adaptive_weighting: bool = True,
                 use_uncertainty_estimation: bool = True,
                 fusion_strategy: str = 'attention'):  # 'weighted', 'attention', 'learned'
        super().__init__()
        
        self.multimodal_model = multimodal_model
        self.gnn_model = gnn_model
        self.contrastive_model = contrastive_model
        
        self.embed_dim = embed_dim
        self.fusion_strategy = fusion_strategy
        self.use_adaptive_weighting = use_adaptive_weighting
        self.use_uncertainty_estimation = use_uncertainty_estimation
        
        # Adaptive weighting module
        if use_adaptive_weighting:
            self.adaptive_weights = AdaptiveWeightingModule(
                input_dim=embed_dim,
                num_models=3,
                hidden_dim=256
            )
        
        # Uncertainty estimation
        if use_uncertainty_estimation:
            self.uncertainty_estimator = UncertaintyEstimator(
                embed_dim=embed_dim,
                hidden_dim=256,
                num_models=3
            )
        
        # Fusion modules based on strategy
        if fusion_strategy == 'attention':
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            self.fusion_norm = nn.LayerNorm(embed_dim)
            
        elif fusion_strategy == 'learned':
            self.fusion_network = nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim)
            )
        
        # Final prediction heads
        self.recommendation_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 4, 1)
        )
        
        # Confidence head for prediction reliability
        self.confidence_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Model availability flags
        self.model_available = {
            'multimodal': True,
            'gnn': True,
            'contrastive': True
        }
        
    def set_model_availability(self, model_flags: Dict[str, bool]):
        """Set which models are available for inference"""
        self.model_available.update(model_flags)
    
    def get_multimodal_embedding(self,
                                text_input_ids: torch.Tensor,
                                text_attention_mask: torch.Tensor,
                                categorical_features: Dict[str, torch.Tensor],
                                numerical_features: torch.Tensor) -> torch.Tensor:
        """Get embedding from multimodal transformer"""
        if not self.model_available['multimodal']:
            return None
        
        with torch.no_grad():
            outputs = self.multimodal_model(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                categorical_features=categorical_features,
                numerical_features=numerical_features
            )
            return outputs['query_embedding']
    
    def get_gnn_embedding(self,
                         show_ids: torch.Tensor,
                         x_dict: Dict[str, torch.Tensor],
                         edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> torch.Tensor:
        """Get embedding from graph neural network"""
        if not self.model_available['gnn']:
            return None
        
        with torch.no_grad():
            outputs = self.gnn_model(x_dict, edge_index_dict)
            show_embeddings = outputs['show_embeddings']
            return show_embeddings[show_ids]
    
    def get_contrastive_embedding(self,
                                 input_ids: torch.Tensor,
                                 attention_mask: torch.Tensor,
                                 categorical_features: Dict[str, torch.Tensor],
                                 numerical_features: torch.Tensor) -> torch.Tensor:
        """Get embedding from contrastive learning model"""
        if not self.model_available['contrastive']:
            return None
        
        with torch.no_grad():
            embedding = self.contrastive_model.get_show_embedding(
                input_ids=input_ids,
                attention_mask=attention_mask,
                categorical_features=categorical_features,
                numerical_features=numerical_features
            )
            return embedding
    
    def fuse_embeddings(self, 
                       embeddings: List[torch.Tensor],
                       context_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Fuse embeddings from multiple models"""
        
        # Filter out None embeddings
        valid_embeddings = [emb for emb in embeddings if emb is not None]
        
        if not valid_embeddings:
            # Return zero embedding if no models available
            batch_size = embeddings[0].size(0) if embeddings else 1
            return torch.zeros(batch_size, self.embed_dim, device=self.device)
        
        if len(valid_embeddings) == 1:
            return valid_embeddings[0]
        
        # Stack embeddings
        stacked_embeddings = torch.stack(valid_embeddings, dim=1)  # [batch, num_models, embed_dim]
        
        if self.fusion_strategy == 'weighted':
            # Simple weighted average
            if self.use_adaptive_weighting and context_embedding is not None:
                weights = self.adaptive_weights(context_embedding)
                weights = weights[:, :len(valid_embeddings)]  # Adjust for available models
                weights = F.softmax(weights, dim=-1)  # Renormalize
                
                # Apply weights
                weighted_embeddings = stacked_embeddings * weights.unsqueeze(-1)
                fused_embedding = weighted_embeddings.sum(dim=1)
            else:
                # Equal weights
                fused_embedding = stacked_embeddings.mean(dim=1)
                
        elif self.fusion_strategy == 'attention':
            # Attention-based fusion
            query = stacked_embeddings.mean(dim=1, keepdim=True)  # [batch, 1, embed_dim]
            
            attended_embeddings, attention_weights = self.fusion_attention(
                query, stacked_embeddings, stacked_embeddings
            )
            
            fused_embedding = self.fusion_norm(
                attended_embeddings.squeeze(1) + query.squeeze(1)
            )
            
        elif self.fusion_strategy == 'learned':
            # Learned fusion network
            concat_embeddings = stacked_embeddings.reshape(
                stacked_embeddings.size(0), -1
            )  # [batch, num_models * embed_dim]
            
            # Pad if necessary
            target_dim = self.embed_dim * 3
            if concat_embeddings.size(1) < target_dim:
                padding = torch.zeros(
                    concat_embeddings.size(0), 
                    target_dim - concat_embeddings.size(1),
                    device=concat_embeddings.device
                )
                concat_embeddings = torch.cat([concat_embeddings, padding], dim=1)
            
            fused_embedding = self.fusion_network(concat_embeddings)
        
        else:
            # Default to mean
            fused_embedding = stacked_embeddings.mean(dim=1)
        
        return fused_embedding
    
    def forward(self,
                # Multimodal inputs
                text_input_ids: torch.Tensor,
                text_attention_mask: torch.Tensor,
                categorical_features: Dict[str, torch.Tensor],
                numerical_features: torch.Tensor,
                # GNN inputs
                show_ids: torch.Tensor,
                x_dict: Optional[Dict[str, torch.Tensor]] = None,
                edge_index_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
                # Target for training
                target_show_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        batch_size = text_input_ids.size(0)
        embeddings = []
        model_names = []
        
        # Get embeddings from each model
        multimodal_emb = self.get_multimodal_embedding(
            text_input_ids, text_attention_mask, categorical_features, numerical_features
        )
        if multimodal_emb is not None:
            embeddings.append(multimodal_emb)
            model_names.append('multimodal')
        
        if x_dict is not None and edge_index_dict is not None:
            gnn_emb = self.get_gnn_embedding(show_ids, x_dict, edge_index_dict)
            if gnn_emb is not None:
                embeddings.append(gnn_emb)
                model_names.append('gnn')
        
        contrastive_emb = self.get_contrastive_embedding(
            text_input_ids, text_attention_mask, categorical_features, numerical_features
        )
        if contrastive_emb is not None:
            embeddings.append(contrastive_emb)
            model_names.append('contrastive')
        
        # Use first available embedding as context for adaptive weighting
        context_embedding = embeddings[0] if embeddings else None
        
        # Fuse embeddings
        fused_embedding = self.fuse_embeddings(embeddings, context_embedding)
        
        # Estimate uncertainties
        uncertainties = None
        if self.use_uncertainty_estimation and len(embeddings) > 1:
            # Pad embeddings to consistent length for uncertainty estimation
            padded_embeddings = []
            for emb in embeddings:
                if emb.size(1) != self.embed_dim:
                    # Pad or truncate to embed_dim
                    if emb.size(1) < self.embed_dim:
                        padding = torch.zeros(
                            emb.size(0), self.embed_dim - emb.size(1),
                            device=emb.device
                        )
                        emb = torch.cat([emb, padding], dim=1)
                    else:
                        emb = emb[:, :self.embed_dim]
                padded_embeddings.append(emb)
            
            # Fill remaining slots with zeros if needed
            while len(padded_embeddings) < 3:
                padded_embeddings.append(
                    torch.zeros(batch_size, self.embed_dim, device=fused_embedding.device)
                )
            
            uncertainties = self.uncertainty_estimator(padded_embeddings[:3])
        
        # Generate predictions
        recommendation_score = self.recommendation_head(fused_embedding)
        confidence_score = self.confidence_head(fused_embedding)
        
        outputs = {
            'fused_embedding': fused_embedding,
            'individual_embeddings': embeddings,
            'model_names': model_names,
            'recommendation_score': recommendation_score,
            'confidence_score': confidence_score
        }
        
        if uncertainties is not None:
            outputs['uncertainties'] = uncertainties
        
        # If target provided, compute similarity
        if target_show_ids is not None:
            # Get target embeddings (simplified - would need full target inputs in practice)
            target_similarities = []
            for i, target_id in enumerate(target_show_ids):
                # This is simplified - in practice you'd need to run the full forward pass for targets
                sim = torch.cosine_similarity(
                    fused_embedding[i:i+1], 
                    fused_embedding[target_id:target_id+1], 
                    dim=1
                )
                target_similarities.append(sim)
            
            outputs['target_similarity'] = torch.stack(target_similarities)
        
        return outputs
    
    def recommend_shows(self,
                       query_text_input_ids: torch.Tensor,
                       query_text_attention_mask: torch.Tensor,
                       query_categorical_features: Dict[str, torch.Tensor],
                       query_numerical_features: torch.Tensor,
                       query_show_id: int,
                       candidate_embeddings: torch.Tensor,
                       candidate_ids: torch.Tensor,
                       x_dict: Optional[Dict[str, torch.Tensor]] = None,
                       edge_index_dict: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None,
                       top_k: int = 10,
                       min_confidence: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Generate top-k recommendations for a query show
        
        Returns:
            Dictionary containing:
            - recommended_ids: Top-k recommended show IDs
            - similarities: Similarity scores
            - confidences: Confidence scores
            - model_contributions: Individual model contributions
        """
        
        with torch.no_grad():
            # Get query embedding
            query_show_ids = torch.tensor([query_show_id], device=query_text_input_ids.device)
            
            query_outputs = self.forward(
                text_input_ids=query_text_input_ids,
                text_attention_mask=query_text_attention_mask,
                categorical_features=query_categorical_features,
                numerical_features=query_numerical_features,
                show_ids=query_show_ids,
                x_dict=x_dict,
                edge_index_dict=edge_index_dict
            )
            
            query_embedding = query_outputs['fused_embedding']
            query_confidence = query_outputs['confidence_score']
            
            # Compute similarities with candidates
            similarities = F.cosine_similarity(
                query_embedding.unsqueeze(1),  # [1, 1, embed_dim]
                candidate_embeddings.unsqueeze(0),  # [1, num_candidates, embed_dim]
                dim=2
            ).squeeze(0)  # [num_candidates]
            
            # Filter by minimum confidence
            confidence_mask = query_confidence.squeeze() >= min_confidence
            if not confidence_mask.any():
                logger.warning(f"Query confidence {query_confidence.item():.3f} below threshold {min_confidence}")
            
            # Get top-k recommendations
            top_similarities, top_indices = torch.topk(
                similarities, 
                min(top_k, len(similarities))
            )
            
            recommended_ids = candidate_ids[top_indices]
            
            # Get individual model contributions
            model_contributions = {}
            for i, (emb, name) in enumerate(zip(query_outputs['individual_embeddings'], 
                                              query_outputs['model_names'])):
                model_sims = F.cosine_similarity(
                    emb.unsqueeze(1),
                    candidate_embeddings.unsqueeze(0),
                    dim=2
                ).squeeze(0)
                model_contributions[name] = model_sims[top_indices]
            
            return {
                'recommended_ids': recommended_ids,
                'similarities': top_similarities,
                'confidences': query_confidence.repeat(len(recommended_ids)),
                'model_contributions': model_contributions,
                'query_embedding': query_embedding,
                'uncertainties': query_outputs.get('uncertainties')
            }
    
    @property
    def device(self) -> torch.device:
        """Get device of the model"""
        return next(self.parameters()).device
    
    def save_ensemble(self, save_path: str):
        """Save ensemble model and configuration"""
        save_dict = {
            'multimodal_state_dict': self.multimodal_model.state_dict(),
            'gnn_state_dict': self.gnn_model.state_dict(),
            'contrastive_state_dict': self.contrastive_model.state_dict(),
            'ensemble_state_dict': self.state_dict(),
            'config': {
                'embed_dim': self.embed_dim,
                'fusion_strategy': self.fusion_strategy,
                'use_adaptive_weighting': self.use_adaptive_weighting,
                'use_uncertainty_estimation': self.use_uncertainty_estimation,
                'model_available': self.model_available
            }
        }
        torch.save(save_dict, save_path)
        logger.info(f"Ensemble model saved to {save_path}")
    
    @classmethod
    def load_ensemble(cls, load_path: str, device: torch.device = None):
        """Load ensemble model from checkpoint"""
        checkpoint = torch.load(load_path, map_location=device)
        
        # This would need proper model initialization based on saved config
        # Implementation would depend on how models are stored
        logger.info(f"Ensemble model loaded from {load_path}")
        return checkpoint

class EnsembleLoss(nn.Module):
    """Multi-task loss for ensemble training"""
    
    def __init__(self,
                 recommendation_weight: float = 1.0,
                 confidence_weight: float = 0.3,
                 consistency_weight: float = 0.5,
                 uncertainty_weight: float = 0.2):
        super().__init__()
        
        self.recommendation_weight = recommendation_weight
        self.confidence_weight = confidence_weight
        self.consistency_weight = consistency_weight
        self.uncertainty_weight = uncertainty_weight
        
        self.recommendation_loss = nn.BCEWithLogitsLoss()
        self.confidence_loss = nn.MSELoss()
        self.consistency_loss = nn.MSELoss()
        
    def forward(self, 
                ensemble_outputs: Dict[str, torch.Tensor],
                labels: torch.Tensor,
                individual_predictions: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        
        losses = {}
        total_loss = 0
        
        # Recommendation loss
        rec_loss = self.recommendation_loss(
            ensemble_outputs['recommendation_score'].squeeze(),
            labels.float()
        )
        losses['recommendation_loss'] = rec_loss
        total_loss += self.recommendation_weight * rec_loss
        
        # Confidence loss (encourage high confidence for correct predictions)
        if 'confidence_score' in ensemble_outputs:
            target_confidence = labels.float()  # High confidence for positive samples
            conf_loss = self.confidence_loss(
                ensemble_outputs['confidence_score'].squeeze(),
                target_confidence
            )
            losses['confidence_loss'] = conf_loss
            total_loss += self.confidence_weight * conf_loss
        
        # Consistency loss (encourage agreement between models)
        if (individual_predictions is not None and 
            len(individual_predictions) > 1 and 
            len(ensemble_outputs['individual_embeddings']) > 1):
            
            consistency_losses = []
            embeddings = ensemble_outputs['individual_embeddings']
            
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    consistency_loss = self.consistency_loss(embeddings[i], embeddings[j])
                    consistency_losses.append(consistency_loss)
            
            if consistency_losses:
                avg_consistency_loss = torch.stack(consistency_losses).mean()
                losses['consistency_loss'] = avg_consistency_loss
                total_loss += self.consistency_weight * avg_consistency_loss
        
        # Uncertainty loss (encourage calibrated uncertainty)
        if 'uncertainties' in ensemble_outputs:
            # Encourage low uncertainty for correct predictions
            target_uncertainty = 1.0 - labels.float()  # Low uncertainty for correct predictions
            uncertainty_loss = F.mse_loss(
                ensemble_outputs['uncertainties'].mean(dim=1),
                target_uncertainty
            )
            losses['uncertainty_loss'] = uncertainty_loss
            total_loss += self.uncertainty_weight * uncertainty_loss
        
        losses['total_loss'] = total_loss
        return losses

# Configuration for ensemble system
def get_ensemble_config():
    """Configuration for ensemble system optimized for RTX 4090"""
    return {
        'embed_dim': 768,
        'fusion_strategy': 'attention',  # 'weighted', 'attention', 'learned'
        'use_adaptive_weighting': True,
        'use_uncertainty_estimation': True,
        'batch_size': 64,  # Moderate batch size for ensemble
        'learning_rate': 5e-5,  # Lower LR for ensemble fine-tuning
        'weight_decay': 1e-6,
        'recommendation_weight': 1.0,
        'confidence_weight': 0.3,
        'consistency_weight': 0.5,
        'uncertainty_weight': 0.2,
        'use_mixed_precision': True,
        'gradient_accumulation_steps': 2,
        'max_grad_norm': 0.5,
    }