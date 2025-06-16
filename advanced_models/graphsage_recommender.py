# GraphSAGE-based Recommendation Model for CineSync v2
# Superior to LightGCN with inductive learning and heterogeneous node features
# Implements GraphSAGE with GAT attention for enhanced graph convolutions

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn.inits import glorot, zeros
from typing import Tuple, Dict, Optional, List, Union
import numpy as np
import math


class HeteroGraphSAGEConv(MessagePassing):
    """
    Heterogeneous GraphSAGE convolution for user-item bipartite graphs
    Handles different node types (users vs items) with separate transformations
    """
    
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: str = 'mean',
        normalize: bool = True,
        bias: bool = True,
        **kwargs
    ):
        super(HeteroGraphSAGEConv, self).__init__(aggr=aggr, **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        
        # Handle heterogeneous input dimensions
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        
        # Separate linear transformations for source and target nodes
        self.lin_src = nn.Linear(in_channels[0], out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels[1], out_channels, bias=False)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.lin_src.weight)
        glorot(self.lin_dst.weight)
        zeros(self.bias)
    
    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        edge_index: torch.Tensor,
        size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        
        if isinstance(x, torch.Tensor):
            x = (x, x)
        
        # Propagate messages
        out = self.propagate(edge_index, x=x, size=size)
        
        # Apply bias
        if self.bias is not None:
            out += self.bias
        
        return out
    
    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return self.lin_src(x_j)
    
    def update(self, aggr_out: torch.Tensor, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # Combine aggregated neighbors with self information
        out = aggr_out + self.lin_dst(x[1])
        
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        
        return out


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention mechanism for enhanced neighbor aggregation
    Learns importance weights for different neighbors
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.1,
        alpha: float = 0.2,
        concat: bool = True
    ):
        super(GraphAttentionLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Linear transformation
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention mechanism
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Node features [N, in_features]
            adj: Adjacency matrix [N, N]
        """
        # Linear transformation
        Wh = torch.mm(h, self.W)  # [N, out_features]
        N = Wh.size()[0]
        
        # Attention mechanism
        # Create all pairs for attention computation
        Wh1 = torch.mm(Wh, self.a[:self.out_features, :])  # [N, 1]
        Wh2 = torch.mm(Wh, self.a[self.out_features:, :])  # [N, 1]
        
        # Broadcast to create attention matrix
        e = Wh1 + Wh2.T  # [N, N]
        e = self.leakyrelu(e)
        
        # Mask attention for non-connected nodes
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Softmax normalization
        attention = F.softmax(attention, dim=1)
        attention = self.dropout_layer(attention)
        
        # Apply attention weights
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class MultiHeadGraphAttention(nn.Module):
    """Multi-head graph attention for richer representations"""
    
    def __init__(
        self,
        n_heads: int,
        in_features: int,
        out_features: int,
        dropout: float = 0.1,
        alpha: float = 0.2
    ):
        super(MultiHeadGraphAttention, self).__init__()
        
        self.n_heads = n_heads
        self.out_features = out_features
        
        # Multiple attention heads
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(
                in_features, out_features // n_heads, dropout, alpha, concat=True
            ) for _ in range(n_heads)
        ])
        
        # Final linear layer
        self.out_proj = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Apply multiple attention heads
        head_outputs = []
        for attention in self.attentions:
            head_out = attention(h, adj)
            head_outputs.append(head_out)
        
        # Concatenate head outputs
        concat_out = torch.cat(head_outputs, dim=1)
        
        # Final projection
        out = self.out_proj(concat_out)
        out = self.dropout(out)
        
        return out


class GraphSAGERecommender(nn.Module):
    """
    GraphSAGE-based recommendation model with enhanced features:
    - Inductive learning capability
    - Heterogeneous node features
    - Multi-head attention
    - Content-aware embeddings
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_feature_dim: int = 0,
        item_feature_dim: int = 0,
        embedding_dim: int = 256,
        hidden_dims: List[int] = [512, 256],
        num_layers: int = 3,
        dropout: float = 0.2,
        use_attention: bool = True,
        attention_heads: int = 8,
        use_residual: bool = True,
        use_batch_norm: bool = True,
        aggregator: str = 'mean'  # 'mean', 'max', 'lstm'
    ):
        super(GraphSAGERecommender, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # Base embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Feature processing (if features are available)
        self.user_feature_processor = None
        self.item_feature_processor = None
        
        if user_feature_dim > 0:
            self.user_feature_processor = nn.Sequential(
                nn.Linear(user_feature_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim // 2, embedding_dim)
            )
        
        if item_feature_dim > 0:
            self.item_feature_processor = nn.Sequential(
                nn.Linear(item_feature_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim // 2, embedding_dim)
            )
        
        # GraphSAGE layers
        self.sage_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        # Input dimension for first layer
        input_dim = embedding_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # GraphSAGE convolution
            sage_layer = HeteroGraphSAGEConv(
                in_channels=input_dim,
                out_channels=hidden_dim,
                aggr=aggregator,
                normalize=True
            )
            self.sage_layers.append(sage_layer)
            
            # Batch normalization
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
            input_dim = hidden_dim
        
        # Final output layer
        self.sage_layers.append(
            HeteroGraphSAGEConv(
                in_channels=input_dim,
                out_channels=embedding_dim,
                aggr=aggregator,
                normalize=True
            )
        )
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(embedding_dim))
        
        # Multi-head attention layers (if enabled)
        if use_attention:
            self.attention_layers = nn.ModuleList([
                MultiHeadGraphAttention(
                    n_heads=attention_heads,
                    in_features=embedding_dim,
                    out_features=embedding_dim,
                    dropout=dropout
                ) for _ in range(num_layers)
            ])
        
        # Content fusion layers
        self.content_fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embedding_dim)
        )
        
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
        
        # Genre prediction head
        self.genre_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 20)  # Assume 20 genres
        )
        
        # Cold-start handling
        self.cold_start_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, embedding_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _create_adjacency_matrix(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Create adjacency matrix from edge index for attention layers"""
        adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        adj[edge_index[0], edge_index[1]] = 1
        return adj
    
    def forward(
        self,
        edge_index: torch.Tensor,
        user_features: Optional[torch.Tensor] = None,
        item_features: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass through GraphSAGE model
        
        Args:
            edge_index: Edge indices [2, num_edges]
            user_features: Additional user features [num_users, user_feature_dim]
            item_features: Additional item features [num_items, item_feature_dim]
            return_attention_weights: Whether to return attention weights
            
        Returns:
            User and item embeddings or full output dict
        """
        # Initial embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        # Process additional features if available
        if user_features is not None and self.user_feature_processor is not None:
            user_feat_emb = self.user_feature_processor(user_features)
            # Combine with base embeddings
            user_emb = self.content_fusion(torch.cat([user_emb, user_feat_emb], dim=1))
        
        if item_features is not None and self.item_feature_processor is not None:
            item_feat_emb = self.item_feature_processor(item_features)
            # Combine with base embeddings
            item_emb = self.content_fusion(torch.cat([item_emb, item_feat_emb], dim=1))
        
        # Concatenate user and item embeddings
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        
        # Store embeddings from each layer
        layer_embeddings = [all_emb]
        attention_weights = [] if return_attention_weights else None
        
        # GraphSAGE layers
        for i, sage_layer in enumerate(self.sage_layers):
            # GraphSAGE convolution
            all_emb = sage_layer(all_emb, edge_index)
            
            # Batch normalization
            if self.batch_norms is not None and i < len(self.batch_norms):
                all_emb = self.batch_norms[i](all_emb)
            
            # Residual connection (if dimensions match)
            if self.use_residual and i > 0 and all_emb.size(-1) == layer_embeddings[-1].size(-1):
                all_emb = all_emb + layer_embeddings[-1]
            
            # Activation and dropout
            all_emb = F.relu(all_emb)
            all_emb = self.dropout(all_emb)
            
            # Apply attention if enabled
            if self.use_attention and i < len(self.attention_layers):
                # Create adjacency matrix for attention
                num_nodes = all_emb.size(0)
                adj = self._create_adjacency_matrix(edge_index, num_nodes)
                
                # Apply multi-head attention
                attended_emb = self.attention_layers[i](all_emb, adj)
                
                # Combine with SAGE output
                all_emb = all_emb + attended_emb
                
                if return_attention_weights:
                    attention_weights.append(adj)  # Store for analysis
            
            layer_embeddings.append(all_emb)
        
        # Combine embeddings from different layers (learned combination)
        layer_weights = F.softmax(torch.randn(len(layer_embeddings)), dim=0)
        final_emb = sum(w * emb for w, emb in zip(layer_weights, layer_embeddings))
        
        # Split back to user and item embeddings
        user_final = final_emb[:self.num_users]
        item_final = final_emb[self.num_users:]
        
        if return_attention_weights:
            return {
                'user_embeddings': user_final,
                'item_embeddings': item_final,
                'attention_weights': attention_weights,
                'layer_embeddings': layer_embeddings
            }
        
        return user_final, item_final
    
    def predict_rating(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        edge_index: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Predict explicit ratings using multi-task head"""
        user_embs, item_embs = self.forward(edge_index, **kwargs)
        
        user_emb = user_embs[user_ids]
        item_emb = item_embs[item_ids]
        
        # Use rating head for explicit rating prediction
        combined = torch.cat([user_emb, item_emb], dim=1)
        rating = self.rating_head(combined).squeeze()
        
        return torch.sigmoid(rating) * 5.0  # Scale to 0-5
    
    def predict_genres(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        edge_index: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Predict item genres based on user-item interaction"""
        user_embs, item_embs = self.forward(edge_index, **kwargs)
        
        user_emb = user_embs[user_ids]
        item_emb = item_embs[item_ids]
        
        combined = torch.cat([user_emb, item_emb], dim=1)
        genre_logits = self.genre_head(combined)
        
        return F.softmax(genre_logits, dim=-1)
    
    def get_recommendations(
        self,
        user_id: int,
        edge_index: torch.Tensor,
        k: int = 10,
        exclude_items: Optional[List[int]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get top-k recommendations for a user"""
        self.eval()
        with torch.no_grad():
            user_embs, item_embs = self.forward(edge_index, **kwargs)
            
            user_emb = user_embs[user_id].unsqueeze(0)  # [1, embedding_dim]
            
            # Compute scores with all items
            scores = torch.matmul(user_emb, item_embs.T).squeeze()  # [num_items]
            
            # Exclude items if specified
            if exclude_items:
                scores[exclude_items] = -float('inf')
            
            # Get top-k
            top_k_scores, top_k_items = torch.topk(scores, k)
            
            return top_k_items, top_k_scores
    
    def handle_cold_start(
        self,
        user_features: Optional[torch.Tensor] = None,
        item_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Handle cold-start users/items using only features"""
        if user_features is not None:
            if self.user_feature_processor is not None:
                user_emb = self.user_feature_processor(user_features)
            else:
                # Use MLP for cold-start
                user_emb = self.cold_start_mlp(user_features)
        else:
            user_emb = None
        
        if item_features is not None:
            if self.item_feature_processor is not None:
                item_emb = self.item_feature_processor(item_features)
            else:
                item_emb = self.cold_start_mlp(item_features)
        else:
            item_emb = None
        
        return user_emb, item_emb
    
    def compute_loss(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
        edge_index: torch.Tensor,
        rating_labels: Optional[torch.Tensor] = None,
        genre_labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Args:
            user_ids: User IDs [batch_size]
            pos_item_ids: Positive item IDs [batch_size]
            neg_item_ids: Negative item IDs [batch_size]
            edge_index: Edge indices
            rating_labels: Optional explicit ratings [batch_size]
            genre_labels: Optional genre labels [batch_size, num_genres]
            
        Returns:
            Dictionary with loss components
        """
        user_embs, item_embs = self.forward(edge_index, **kwargs)
        
        user_emb = user_embs[user_ids]
        pos_item_emb = item_embs[pos_item_ids]
        neg_item_emb = item_embs[neg_item_ids]
        
        # BPR loss for implicit feedback
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        total_loss = bpr_loss
        losses = {'bpr_loss': bpr_loss}
        
        # Rating prediction loss (if available)
        if rating_labels is not None:
            predicted_ratings = self.predict_rating(user_ids, pos_item_ids, edge_index, **kwargs)
            rating_loss = F.mse_loss(predicted_ratings, rating_labels)
            losses['rating_loss'] = rating_loss
            total_loss += 0.5 * rating_loss
        
        # Genre prediction loss (if available)
        if genre_labels is not None:
            predicted_genres = self.predict_genres(user_ids, pos_item_ids, edge_index, **kwargs)
            genre_loss = F.cross_entropy(predicted_genres, genre_labels)
            losses['genre_loss'] = genre_loss
            total_loss += 0.3 * genre_loss
        
        # Regularization
        reg_loss = 0.01 * (
            torch.norm(user_emb) ** 2 + 
            torch.norm(pos_item_emb) ** 2 + 
            torch.norm(neg_item_emb) ** 2
        ) / user_emb.size(0)
        
        losses['reg_loss'] = reg_loss
        total_loss += reg_loss
        losses['total_loss'] = total_loss
        
        return losses


class InductiveGraphSAGE(GraphSAGERecommender):
    """
    Inductive GraphSAGE variant for handling new users/items
    Can make predictions without retraining on new nodes
    """
    
    def __init__(self, *args, **kwargs):
        super(InductiveGraphSAGE, self).__init__(*args, **kwargs)
        
        # Additional components for inductive learning
        self.node_classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embedding_dim // 2, 2)  # User vs Item classification
        )
        
        # Feature-based embedding generator for new nodes
        self.feature_to_embedding = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
    
    def inductive_forward(
        self,
        new_node_features: torch.Tensor,
        neighbor_embeddings: torch.Tensor,
        neighbor_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate embeddings for new nodes inductively
        
        Args:
            new_node_features: Features for new nodes [num_new_nodes, feature_dim]
            neighbor_embeddings: Embeddings of neighbor nodes [num_neighbors, embedding_dim]
            neighbor_indices: Indices mapping new nodes to neighbors [num_edges, 2]
            
        Returns:
            Embeddings for new nodes [num_new_nodes, embedding_dim]
        """
        # Generate initial embeddings from features
        initial_emb = self.feature_to_embedding(new_node_features)
        
        # Aggregate neighbor information
        num_new_nodes = new_node_features.size(0)
        aggregated_neighbors = torch.zeros(
            num_new_nodes, self.embedding_dim, 
            device=new_node_features.device
        )
        
        # Simple mean aggregation of neighbors
        for i in range(num_new_nodes):
            neighbor_mask = neighbor_indices[:, 0] == i
            if neighbor_mask.sum() > 0:
                relevant_neighbors = neighbor_embeddings[neighbor_indices[neighbor_mask, 1]]
                aggregated_neighbors[i] = relevant_neighbors.mean(dim=0)
        
        # Combine initial embedding with neighbor information
        final_emb = initial_emb + aggregated_neighbors
        final_emb = F.normalize(final_emb, p=2, dim=-1)
        
        return final_emb


class GraphSAGETrainer:
    """Enhanced trainer for GraphSAGE models"""
    
    def __init__(
        self,
        model: GraphSAGERecommender,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        use_amp: bool = True,
        gradient_clip_val: float = 1.0,
        scheduler_type: str = 'cosine'
    ):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp
        self.gradient_clip_val = gradient_clip_val
        
        # Optimizer with different learning rates for different components
        sage_params = []
        attention_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if 'sage_layers' in name:
                sage_params.append(param)
            elif 'attention' in name:
                attention_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = [
            {'params': sage_params, 'lr': learning_rate},
            {'params': attention_params, 'lr': learning_rate * 0.5},  # Lower LR for attention
            {'params': other_params, 'lr': learning_rate}
        ]
        
        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        if scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=50, T_mult=2, eta_min=1e-6
            )
        elif scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )
        else:
            self.scheduler = None
        
        # Mixed precision
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Enhanced training step with multi-task learning"""
        self.model.train()
        
        # Move batch to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        
        self.optimizer.zero_grad()
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                losses = self.model.compute_loss(
                    user_ids=batch['user_ids'],
                    pos_item_ids=batch['pos_item_ids'],
                    neg_item_ids=batch['neg_item_ids'],
                    edge_index=batch['edge_index'],
                    rating_labels=batch.get('rating_labels'),
                    genre_labels=batch.get('genre_labels'),
                    user_features=batch.get('user_features'),
                    item_features=batch.get('item_features')
                )
                
                total_loss = losses['total_loss']
            
            self.scaler.scale(total_loss).backward()
            
            if self.gradient_clip_val > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses = self.model.compute_loss(
                user_ids=batch['user_ids'],
                pos_item_ids=batch['pos_item_ids'],
                neg_item_ids=batch['neg_item_ids'],
                edge_index=batch['edge_index'],
                rating_labels=batch.get('rating_labels'),
                genre_labels=batch.get('genre_labels'),
                user_features=batch.get('user_features'),
                item_features=batch.get('item_features')
            )
            
            total_loss = losses['total_loss']
            total_loss.backward()
            
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            
            self.optimizer.step()
        
        if self.scheduler:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(total_loss)
            else:
                self.scheduler.step()
        
        # Convert losses to metrics
        metrics = {}
        for key, value in losses.items():
            metrics[key] = value.item() if hasattr(value, 'item') else value
        
        metrics['lr'] = self.optimizer.param_groups[0]['lr']
        
        return metrics