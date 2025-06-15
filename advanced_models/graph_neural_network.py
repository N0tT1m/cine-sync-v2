import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import Tuple, Dict, Optional, List, Union
import numpy as np
import scipy.sparse as sp
from torch_sparse import SparseTensor


class LightGCNConv(MessagePassing):
    """Light Graph Convolutional Network layer"""
    
    def __init__(self, **kwargs):
        super(LightGCNConv, self).__init__(aggr='add', **kwargs)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of LightGCN convolution
        
        Args:
            x: Node features [num_nodes, embedding_dim]
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Updated node features
        """
        # Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Propagate messages
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        """Message function"""
        return norm.view(-1, 1) * x_j


class LightGCN(nn.Module):
    """
    Light Graph Convolutional Network for Collaborative Filtering
    
    Paper: He, X., et al. "LightGCN: Simplifying and Powering Graph Convolution Network 
    for Recommendation." SIGIR 2020.
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        reg_weight: float = 1e-4
    ):
        super(LightGCN, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.reg_weight = reg_weight
        
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Graph convolution layers
        self.convs = nn.ModuleList([
            LightGCNConv() for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings with Xavier normal"""
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
    
    def create_bipartite_graph(self, user_indices: torch.Tensor, 
                              item_indices: torch.Tensor) -> torch.Tensor:
        """
        Create bipartite graph edge index from user-item interactions
        
        Args:
            user_indices: User indices [num_interactions]
            item_indices: Item indices [num_interactions]
            
        Returns:
            Edge index for bipartite graph [2, num_edges]
        """
        # Create edges: user -> item and item -> user
        user_to_item = torch.stack([user_indices, item_indices + self.num_users])
        item_to_user = torch.stack([item_indices + self.num_users, user_indices])
        
        edge_index = torch.cat([user_to_item, item_to_user], dim=1)
        
        return edge_index
    
    def forward(self, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LightGCN
        
        Args:
            edge_index: Edge indices [2, num_edges]
            
        Returns:
            Tuple of (user_embeddings, item_embeddings)
        """
        # Initial embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        # Concatenate user and item embeddings for graph convolution
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        
        # Store embeddings from each layer
        emb_layers = [all_emb]
        
        # Graph convolution layers
        for conv in self.convs:
            all_emb = conv(all_emb, edge_index)
            all_emb = self.dropout(all_emb)
            emb_layers.append(all_emb)
        
        # Mean pooling of all layers (including initial)
        final_emb = torch.mean(torch.stack(emb_layers), dim=0)
        
        # Split back to user and item embeddings
        user_final = final_emb[:self.num_users]
        item_final = final_emb[self.num_users:]
        
        return user_final, item_final
    
    def predict(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Predict ratings for user-item pairs
        
        Args:
            user_ids: User IDs [batch_size]
            item_ids: Item IDs [batch_size]
            edge_index: Edge indices for graph structure
            
        Returns:
            Predicted ratings [batch_size]
        """
        user_embs, item_embs = self.forward(edge_index)
        
        user_emb = user_embs[user_ids]
        item_emb = item_embs[item_ids]
        
        # Dot product for rating prediction
        ratings = torch.sum(user_emb * item_emb, dim=1)
        
        return ratings
    
    def get_user_recommendations(self, user_id: int, edge_index: torch.Tensor,
                               k: int = 10, exclude_items: Optional[List[int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k recommendations for a user
        
        Args:
            user_id: User ID
            edge_index: Edge indices for graph structure
            k: Number of recommendations
            exclude_items: Items to exclude from recommendations
            
        Returns:
            Tuple of (top_k_items, top_k_scores)
        """
        self.eval()
        with torch.no_grad():
            user_embs, item_embs = self.forward(edge_index)
            
            user_emb = user_embs[user_id].unsqueeze(0)  # [1, embedding_dim]
            
            # Compute scores with all items
            scores = torch.matmul(user_emb, item_embs.T).squeeze()  # [num_items]
            
            # Exclude items if specified
            if exclude_items:
                scores[exclude_items] = -float('inf')
            
            # Get top-k
            top_k_scores, top_k_items = torch.topk(scores, k)
            
            return top_k_items, top_k_scores
    
    def compute_loss(self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor,
                    neg_item_ids: torch.Tensor, edge_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute BPR loss with regularization
        
        Args:
            user_ids: User IDs [batch_size]
            pos_item_ids: Positive item IDs [batch_size]  
            neg_item_ids: Negative item IDs [batch_size]
            edge_index: Edge indices for graph structure
            
        Returns:
            Dictionary with loss components
        """
        user_embs, item_embs = self.forward(edge_index)
        
        user_emb = user_embs[user_ids]
        pos_item_emb = item_embs[pos_item_ids]
        neg_item_emb = item_embs[neg_item_ids]
        
        # BPR loss
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)
        
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(user_emb) ** 2 + 
            torch.norm(pos_item_emb) ** 2 + 
            torch.norm(neg_item_emb) ** 2
        ) / user_emb.size(0)
        
        total_loss = bpr_loss + reg_loss
        
        return {
            'bpr_loss': bpr_loss,
            'reg_loss': reg_loss,
            'total_loss': total_loss
        }


class EnhancedLightGCN(LightGCN):
    """
    Enhanced LightGCN with additional features for better performance
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_genres: int = 20,
        embedding_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.2,
        reg_weight: float = 1e-4,
        use_node_features: bool = True,
        use_attention: bool = True,
        use_residual: bool = True
    ):
        super(EnhancedLightGCN, self).__init__(
            num_users, num_items, embedding_dim, num_layers, dropout, reg_weight
        )
        
        self.num_genres = num_genres
        self.use_node_features = use_node_features
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # Genre embeddings for content information
        if use_node_features:
            self.genre_embedding = nn.Embedding(num_genres, embedding_dim // 4)
            self.user_feature_fusion = nn.Sequential(
                nn.Linear(embedding_dim + embedding_dim // 4, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.item_feature_fusion = nn.Sequential(
                nn.Linear(embedding_dim + embedding_dim // 4, embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # Attention mechanism for layer combination
        if use_attention:
            self.layer_attention = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Linear(embedding_dim // 2, 1)
            )
        
        # Enhanced convolution layers with residual connections
        if use_residual:
            self.residual_convs = nn.ModuleList([
                nn.Sequential(
                    LightGCNConv(),
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ) for _ in range(num_layers)
            ])
        
        # Multi-task heads
        self.rating_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1)
        )
        
        self.popularity_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1)
        )
    
    def forward(self, edge_index: torch.Tensor, 
                user_genres: Optional[torch.Tensor] = None,
                item_genres: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhanced forward pass with features and attention
        
        Args:
            edge_index: Edge indices [2, num_edges]
            user_genres: User genre preferences [num_users, num_genres]
            item_genres: Item genre information [num_items, num_genres]
            
        Returns:
            Enhanced user and item embeddings
        """
        # Initial embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        # Add genre information if available
        if self.use_node_features and user_genres is not None and item_genres is not None:
            # User genre features
            user_genre_emb = torch.matmul(user_genres.float(), self.genre_embedding.weight)
            user_emb = self.user_feature_fusion(torch.cat([user_emb, user_genre_emb], dim=1))
            
            # Item genre features
            item_genre_emb = torch.matmul(item_genres.float(), self.genre_embedding.weight)
            item_emb = self.item_feature_fusion(torch.cat([item_emb, item_genre_emb], dim=1))
        
        # Concatenate for graph convolution
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        emb_layers = [all_emb]
        
        # Enhanced graph convolution with residual connections
        if self.use_residual:
            for i, residual_conv in enumerate(self.residual_convs):
                # Standard convolution
                conv_out = self.convs[i](all_emb, edge_index)
                
                # Residual connection
                residual_out = residual_conv[1:](conv_out)  # Skip LightGCNConv
                all_emb = all_emb + residual_out
                
                all_emb = self.dropout(all_emb)
                emb_layers.append(all_emb)
        else:
            # Standard LightGCN layers
            for conv in self.convs:
                all_emb = conv(all_emb, edge_index)
                all_emb = self.dropout(all_emb)
                emb_layers.append(all_emb)
        
        # Attention-based layer combination
        if self.use_attention:
            layer_weights = []
            for emb in emb_layers:
                weight = self.layer_attention(emb).mean(dim=0)  # [1]
                layer_weights.append(weight)
            
            layer_weights = F.softmax(torch.stack(layer_weights), dim=0)
            final_emb = sum(w * emb for w, emb in zip(layer_weights, emb_layers))
        else:
            # Mean pooling
            final_emb = torch.mean(torch.stack(emb_layers), dim=0)
        
        # Split embeddings
        user_final = final_emb[:self.num_users]
        item_final = final_emb[self.num_users:]
        
        return user_final, item_final
    
    def predict_rating(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                      edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """Predict explicit ratings using multi-task head"""
        user_embs, item_embs = self.forward(edge_index, **kwargs)
        
        user_emb = user_embs[user_ids]
        item_emb = item_embs[item_ids]
        
        # Use rating head for explicit rating prediction
        combined = torch.cat([user_emb, item_emb], dim=1)
        rating = self.rating_head(combined).squeeze()
        
        return torch.sigmoid(rating) * 5.0  # Scale to 0-5


class GraphSAINT(nn.Module):
    """
    GraphSAINT-based model for large-scale graph recommendation
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 128,
        num_layers: int = 3,
        sample_coverage: int = 20,
        dropout: float = 0.2
    ):
        super(GraphSAINT, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.sample_coverage = sample_coverage
        
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Graph convolution layers
        self.convs = nn.ModuleList([
            LightGCNConv() for _ in range(num_layers)
        ])
        
        # Batch normalization for each layer
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(embedding_dim) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
    
    def forward(self, edge_index: torch.Tensor, node_norm: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with GraphSAINT sampling
        
        Args:
            edge_index: Sampled edge indices
            node_norm: Node normalization from sampling
            
        Returns:
            User and item embeddings
        """
        # Initial embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        
        emb_layers = [all_emb]
        
        # Graph convolution with batch normalization
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            all_emb = conv(all_emb, edge_index)
            all_emb = bn(all_emb)
            all_emb = F.relu(all_emb)
            all_emb = self.dropout(all_emb)
            emb_layers.append(all_emb)
        
        # Combine layers
        final_emb = torch.mean(torch.stack(emb_layers), dim=0)
        
        # Apply node normalization if provided
        if node_norm is not None:
            final_emb = final_emb * node_norm.unsqueeze(1)
        
        user_final = final_emb[:self.num_users]
        item_final = final_emb[self.num_users:]
        
        return user_final, item_final


class GCNTrainer:
    """Advanced trainer for Graph Neural Network models"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        use_amp: bool = True,
        gradient_clip_val: float = 1.0
    ):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp
        self.gradient_clip_val = gradient_clip_val
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Mixed precision scaler
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step for GCN models"""
        self.model.train()
        
        # Move batch to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        
        self.optimizer.zero_grad()
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                if hasattr(self.model, 'compute_loss'):
                    losses = self.model.compute_loss(
                        user_ids=batch['user_ids'],
                        pos_item_ids=batch['pos_item_ids'],
                        neg_item_ids=batch['neg_item_ids'],
                        edge_index=batch['edge_index']
                    )
                    loss = losses['total_loss']
                else:
                    # Custom loss computation
                    user_embs, item_embs = self.model(batch['edge_index'])
                    
                    user_emb = user_embs[batch['user_ids']]
                    pos_item_emb = item_embs[batch['pos_item_ids']]
                    neg_item_emb = item_embs[batch['neg_item_ids']]
                    
                    pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
                    neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)
                    
                    loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
                    losses = {'total_loss': loss}
            
            self.scaler.scale(loss).backward()
            
            if self.gradient_clip_val > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if hasattr(self.model, 'compute_loss'):
                losses = self.model.compute_loss(
                    user_ids=batch['user_ids'],
                    pos_item_ids=batch['pos_item_ids'],
                    neg_item_ids=batch['neg_item_ids'],
                    edge_index=batch['edge_index']
                )
                loss = losses['total_loss']
            else:
                user_embs, item_embs = self.model(batch['edge_index'])
                
                user_emb = user_embs[batch['user_ids']]
                pos_item_emb = item_embs[batch['pos_item_ids']]
                neg_item_emb = item_embs[batch['neg_item_ids']]
                
                pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
                neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)
                
                loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
                losses = {'total_loss': loss}
            
            loss.backward()
            
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            
            self.optimizer.step()
        
        # Convert losses to metrics
        metrics = {}
        for key, value in losses.items():
            metrics[key] = value.item() if hasattr(value, 'item') else value
        
        metrics['lr'] = self.optimizer.param_groups[0]['lr']
        
        return metrics
    
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Validation step"""
        self.model.eval()
        
        with torch.no_grad():
            # Move batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)
            
            if hasattr(self.model, 'compute_loss'):
                losses = self.model.compute_loss(
                    user_ids=batch['user_ids'],
                    pos_item_ids=batch['pos_item_ids'],
                    neg_item_ids=batch['neg_item_ids'],
                    edge_index=batch['edge_index']
                )
            else:
                user_embs, item_embs = self.model(batch['edge_index'])
                
                user_emb = user_embs[batch['user_ids']]
                pos_item_emb = item_embs[batch['pos_item_ids']]
                neg_item_emb = item_embs[batch['neg_item_ids']]
                
                pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
                neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)
                
                loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
                losses = {'total_loss': loss}
        
        # Convert to metrics
        metrics = {}
        for key, value in losses.items():
            metrics[f'val_{key}'] = value.item() if hasattr(value, 'item') else value
        
        return metrics


def create_graph_from_ratings(ratings_df, num_users: int, num_items: int) -> torch.Tensor:
    """
    Create bipartite graph edge index from ratings dataframe
    
    Args:
        ratings_df: DataFrame with 'user_id' and 'item_id' columns
        num_users: Total number of users
        num_items: Total number of items
        
    Returns:
        Edge index tensor [2, num_edges]
    """
    user_ids = torch.LongTensor(ratings_df['user_id'].values)
    item_ids = torch.LongTensor(ratings_df['item_id'].values)
    
    # Create bidirectional edges
    edge_index = torch.stack([
        torch.cat([user_ids, item_ids + num_users]),
        torch.cat([item_ids + num_users, user_ids])
    ])
    
    return edge_index


def negative_sampling(positive_edges: torch.Tensor, num_users: int, num_items: int, 
                     num_neg_samples: int = 1) -> torch.Tensor:
    """
    Perform negative sampling for training
    
    Args:
        positive_edges: Positive user-item pairs [num_pos, 2]
        num_users: Total number of users
        num_items: Total number of items
        num_neg_samples: Number of negative samples per positive sample
        
    Returns:
        Negative user-item pairs [num_pos * num_neg_samples, 2]
    """
    num_pos = positive_edges.size(0)
    
    # Create set of positive pairs for fast lookup
    positive_set = set(map(tuple, positive_edges.tolist()))
    
    negative_pairs = []
    
    for i in range(num_pos * num_neg_samples):
        while True:
            user_id = torch.randint(0, num_users, (1,)).item()
            item_id = torch.randint(0, num_items, (1,)).item()
            
            if (user_id, item_id) not in positive_set:
                negative_pairs.append([user_id, item_id])
                break
    
    return torch.LongTensor(negative_pairs)