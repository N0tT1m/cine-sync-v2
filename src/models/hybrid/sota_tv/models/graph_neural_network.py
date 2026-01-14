"""
Graph Neural Network TV Recommender (GNN-TV)
State-of-the-art GNN for TV recommendation using heterogeneous graphs
Optimized for RTX 4090 (24GB VRAM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, HeteroConv, Linear
from torch_geometric.data import HeteroData
from torch_geometric.utils import negative_sampling
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

class HeteroGraphSAGE(nn.Module):
    """
    Heterogeneous GraphSAGE for TV recommendation
    
    Node types: ['show', 'actor', 'genre', 'network', 'creator']
    Edge types: [
        ('show', 'has_actor', 'actor'),
        ('show', 'has_genre', 'genre'),
        ('show', 'on_network', 'network'),
        ('show', 'created_by', 'creator'),
        ('actor', 'acted_with', 'actor'),
        ('show', 'similar_to', 'show')
    ]
    """
    
    def __init__(self,
                 node_types: List[str],
                 edge_types: List[Tuple[str, str, str]],
                 hidden_dim: int = 512,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 use_attention: bool = True):
        super().__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        
        # Node embeddings initialization
        self.node_embeddings = nn.ModuleDict({
            node_type: nn.Embedding(10000, hidden_dim)  # Will be dynamically sized
            for node_type in node_types
        })
        
        # Heterogeneous convolution layers
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                src_type, rel_type, dst_type = edge_type
                if use_attention:
                    conv_dict[edge_type] = GATConv(
                        hidden_dim, hidden_dim // 8, heads=8, dropout=dropout
                    )
                else:
                    conv_dict[edge_type] = SAGEConv(
                        hidden_dim, hidden_dim, aggr='mean'
                    )
            
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
        
        # Layer normalization for each node type
        self.layer_norms = nn.ModuleDict({
            node_type: nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_layers)
            ]) for node_type in node_types
        })
        
        # Dropout layers
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(num_layers)
        ])
        
        # Output projections
        self.output_projections = nn.ModuleDict({
            node_type: nn.Linear(hidden_dim, hidden_dim)
            for node_type in node_types
        })
        
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        # Initialize node features with embeddings
        for node_type in self.node_types:
            if node_type not in x_dict:
                # Use node indices as features if not provided
                num_nodes = max(edge_index_dict[edge_key].max().item() + 1 
                              for edge_key in edge_index_dict.keys() 
                              if edge_key[0] == node_type or edge_key[2] == node_type)
                x_dict[node_type] = torch.arange(num_nodes, device=next(self.parameters()).device)
            
            if x_dict[node_type].dtype == torch.long:
                x_dict[node_type] = self.node_embeddings[node_type](x_dict[node_type])
        
        # Apply heterogeneous convolutions
        for i, conv in enumerate(self.convs):
            x_dict_new = conv(x_dict, edge_index_dict)
            
            # Apply layer norm and dropout
            for node_type in self.node_types:
                if node_type in x_dict_new:
                    # Residual connection
                    x_dict_new[node_type] = x_dict_new[node_type] + x_dict.get(node_type, 0)
                    # Layer norm
                    x_dict_new[node_type] = self.layer_norms[node_type][i](x_dict_new[node_type])
                    # Dropout
                    x_dict_new[node_type] = self.dropouts[i](x_dict_new[node_type])
                    # Activation
                    x_dict_new[node_type] = F.gelu(x_dict_new[node_type])
            
            x_dict = x_dict_new
        
        # Output projections
        for node_type in self.node_types:
            if node_type in x_dict:
                x_dict[node_type] = self.output_projections[node_type](x_dict[node_type])
        
        return x_dict

class MetaPathAggregator(nn.Module):
    """Aggregates information along different meta-paths in the heterogeneous graph"""
    
    def __init__(self, hidden_dim: int, num_metapaths: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_metapaths = num_metapaths
        
        # Attention weights for different meta-paths
        self.metapath_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # Meta-path specific transformations
        self.metapath_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_metapaths)
        ])
        
        # Final aggregation
        self.final_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, 
                show_embeddings: torch.Tensor,
                metapath_embeddings: List[torch.Tensor]) -> torch.Tensor:
        
        # Transform each meta-path embedding
        transformed_paths = []
        for i, path_emb in enumerate(metapath_embeddings[:self.num_metapaths]):
            if i < len(self.metapath_transforms):
                transformed = self.metapath_transforms[i](path_emb)
                transformed_paths.append(transformed)
        
        if not transformed_paths:
            return self.final_transform(show_embeddings)
        
        # Stack meta-path embeddings
        metapath_stack = torch.stack(transformed_paths, dim=1)  # [batch, num_paths, hidden_dim]
        
        # Add show embedding as query
        show_query = show_embeddings.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Apply attention
        attended_paths, attention_weights = self.metapath_attention(
            show_query, metapath_stack, metapath_stack
        )
        
        # Combine with original show embedding
        combined = show_embeddings + attended_paths.squeeze(1)
        
        return self.final_transform(combined)

class TVGraphRecommender(nn.Module):
    """
    Complete TV Graph Neural Network Recommender
    
    Combines:
    - Heterogeneous GraphSAGE for node embeddings
    - Meta-path aggregation for complex relationships
    - Multi-task learning for various recommendation tasks
    """
    
    def __init__(self,
                 num_shows: int,
                 num_actors: int,
                 num_genres: int,
                 num_networks: int,
                 num_creators: int,
                 hidden_dim: int = 512,
                 num_gnn_layers: int = 3,
                 dropout: float = 0.1,
                 use_metapath: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_metapath = use_metapath
        
        # Node type definitions
        self.node_types = ['show', 'actor', 'genre', 'network', 'creator']
        self.edge_types = [
            ('show', 'has_actor', 'actor'),
            ('show', 'has_genre', 'genre'),
            ('show', 'on_network', 'network'),
            ('show', 'created_by', 'creator'),
            ('actor', 'acted_with', 'actor'),
            ('show', 'similar_to', 'show')
        ]
        
        # Initialize node embedding sizes
        self.num_nodes = {
            'show': num_shows,
            'actor': num_actors,
            'genre': num_genres,
            'network': num_networks,
            'creator': num_creators
        }
        
        # Heterogeneous GraphSAGE
        self.gnn = HeteroGraphSAGE(
            node_types=self.node_types,
            edge_types=self.edge_types,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            dropout=dropout,
            use_attention=True
        )
        
        # Update embedding sizes based on actual data
        for node_type, num_nodes in self.num_nodes.items():
            self.gnn.node_embeddings[node_type] = nn.Embedding(num_nodes, hidden_dim)
        
        # Meta-path aggregator
        if use_metapath:
            self.metapath_aggregator = MetaPathAggregator(hidden_dim)
        
        # Recommendation heads
        self.similarity_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Genre prediction head (auxiliary task)
        self.genre_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_genres)
        )
        
        # Network prediction head (auxiliary task)
        self.network_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_networks)
        )
        
        # Temperature for contrastive learning
        self.temperature = nn.Parameter(torch.tensor(0.05))
        
    def get_metapath_embeddings(self, 
                               show_ids: torch.Tensor,
                               node_embeddings: Dict[str, torch.Tensor],
                               edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> List[torch.Tensor]:
        """Extract embeddings along different meta-paths"""
        
        metapath_embeddings = []
        
        # Meta-path 1: Show -> Actor -> Show
        if ('show', 'has_actor', 'actor') in edge_index_dict:
            show_actor_edges = edge_index_dict[('show', 'has_actor', 'actor')]
            # Implementation would involve graph traversal
            # For now, return actor embeddings for shows
            actor_embeds = node_embeddings.get('actor', torch.zeros(1, self.hidden_dim, device=show_ids.device))
            metapath_embeddings.append(actor_embeds.mean(dim=0, keepdim=True).expand(len(show_ids), -1))
        
        # Meta-path 2: Show -> Genre -> Show
        if ('show', 'has_genre', 'genre') in edge_index_dict:
            genre_embeds = node_embeddings.get('genre', torch.zeros(1, self.hidden_dim, device=show_ids.device))
            metapath_embeddings.append(genre_embeds.mean(dim=0, keepdim=True).expand(len(show_ids), -1))
        
        # Meta-path 3: Show -> Network -> Show
        if ('show', 'on_network', 'network') in edge_index_dict:
            network_embeds = node_embeddings.get('network', torch.zeros(1, self.hidden_dim, device=show_ids.device))
            metapath_embeddings.append(network_embeds.mean(dim=0, keepdim=True).expand(len(show_ids), -1))
        
        # Meta-path 4: Show -> Creator -> Show
        if ('show', 'created_by', 'creator') in edge_index_dict:
            creator_embeds = node_embeddings.get('creator', torch.zeros(1, self.hidden_dim, device=show_ids.device))
            metapath_embeddings.append(creator_embeds.mean(dim=0, keepdim=True).expand(len(show_ids), -1))
        
        # Meta-path 5: Show -> Similar -> Show
        if ('show', 'similar_to', 'show') in edge_index_dict:
            show_embeds = node_embeddings.get('show', torch.zeros(1, self.hidden_dim, device=show_ids.device))
            metapath_embeddings.append(show_embeds[show_ids])
        
        return metapath_embeddings
    
    def forward(self,
                x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                show_pairs: Optional[torch.Tensor] = None,
                target_genres: Optional[torch.Tensor] = None,
                target_networks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        # Get node embeddings from GNN
        node_embeddings = self.gnn(x_dict, edge_index_dict)
        
        show_embeddings = node_embeddings['show']
        
        # Apply meta-path aggregation if enabled
        if self.use_metapath and len(show_embeddings) > 0:
            metapath_embeds = self.get_metapath_embeddings(
                torch.arange(len(show_embeddings), device=show_embeddings.device),
                node_embeddings,
                edge_index_dict
            )
            if metapath_embeds:
                show_embeddings = self.metapath_aggregator(show_embeddings, metapath_embeds)
        
        outputs = {
            'show_embeddings': show_embeddings,
            'node_embeddings': node_embeddings
        }
        
        # Similarity prediction for show pairs
        if show_pairs is not None:
            show1_embeds = show_embeddings[show_pairs[:, 0]]
            show2_embeds = show_embeddings[show_pairs[:, 1]]
            
            # Concatenate embeddings
            pair_embeds = torch.cat([show1_embeds, show2_embeds], dim=1)
            similarity_scores = self.similarity_head(pair_embeds)
            
            # Cosine similarity for contrastive learning
            cosine_sim = F.cosine_similarity(show1_embeds, show2_embeds, dim=1)
            contrastive_logits = cosine_sim / self.temperature
            
            outputs.update({
                'similarity_scores': similarity_scores,
                'contrastive_logits': contrastive_logits,
                'cosine_similarity': cosine_sim
            })
        
        # Genre prediction (auxiliary task)
        if target_genres is not None:
            genre_logits = self.genre_head(show_embeddings)
            outputs['genre_logits'] = genre_logits
        
        # Network prediction (auxiliary task)
        if target_networks is not None:
            network_logits = self.network_head(show_embeddings)
            outputs['network_logits'] = network_logits
        
        return outputs
    
    def get_show_embedding(self, 
                          show_id: int,
                          x_dict: Dict[str, torch.Tensor],
                          edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> torch.Tensor:
        """Get embedding for a single show"""
        with torch.no_grad():
            outputs = self.forward(x_dict, edge_index_dict)
            return outputs['show_embeddings'][show_id]
    
    def recommend_similar_shows(self,
                               query_show_id: int,
                               x_dict: Dict[str, torch.Tensor],
                               edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                               top_k: int = 10,
                               exclude_ids: Optional[List[int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recommend similar shows using graph embeddings"""
        with torch.no_grad():
            outputs = self.forward(x_dict, edge_index_dict)
            show_embeddings = outputs['show_embeddings']
            
            query_embedding = show_embeddings[query_show_id].unsqueeze(0)
            
            # Compute similarities
            similarities = F.cosine_similarity(query_embedding, show_embeddings, dim=1)
            
            # Exclude query show and any specified shows
            exclude_mask = torch.ones_like(similarities, dtype=torch.bool)
            exclude_mask[query_show_id] = False
            if exclude_ids:
                for exclude_id in exclude_ids:
                    if exclude_id < len(exclude_mask):
                        exclude_mask[exclude_id] = False
            
            similarities[~exclude_mask] = -float('inf')
            
            # Get top-k recommendations
            top_similarities, top_indices = torch.topk(similarities, min(top_k, exclude_mask.sum().item()))
            
            return top_indices, top_similarities

class GraphLoss(nn.Module):
    """Multi-task loss for graph neural network"""
    
    def __init__(self,
                 similarity_weight: float = 1.0,
                 contrastive_weight: float = 0.5,
                 genre_weight: float = 0.3,
                 network_weight: float = 0.2):
        super().__init__()
        self.similarity_weight = similarity_weight
        self.contrastive_weight = contrastive_weight
        self.genre_weight = genre_weight
        self.network_weight = network_weight
        
        self.similarity_loss = nn.BCEWithLogitsLoss()
        self.contrastive_loss = nn.CrossEntropyLoss()
        self.genre_loss = nn.BCEWithLogitsLoss()
        self.network_loss = nn.CrossEntropyLoss()
    
    def forward(self, 
                outputs: Dict[str, torch.Tensor],
                similarity_labels: Optional[torch.Tensor] = None,
                genre_labels: Optional[torch.Tensor] = None,
                network_labels: Optional[torch.Tensor] = None,
                negative_samples: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        losses = {}
        total_loss = 0
        
        # Similarity prediction loss
        if similarity_labels is not None and 'similarity_scores' in outputs:
            sim_loss = self.similarity_loss(
                outputs['similarity_scores'].squeeze(),
                similarity_labels.float()
            )
            losses['similarity_loss'] = sim_loss
            total_loss += self.similarity_weight * sim_loss
        
        # Contrastive loss
        if negative_samples is not None and 'contrastive_logits' in outputs:
            # Create labels for contrastive learning
            batch_size = outputs['contrastive_logits'].size(0)
            contrastive_labels = torch.zeros(batch_size, dtype=torch.long, 
                                           device=outputs['contrastive_logits'].device)
            
            # Combine positive and negative logits
            pos_logits = outputs['contrastive_logits'].unsqueeze(1)
            neg_logits = negative_samples  # Should be [batch, num_negatives]
            all_logits = torch.cat([pos_logits, neg_logits], dim=1)
            
            cont_loss = self.contrastive_loss(all_logits, contrastive_labels)
            losses['contrastive_loss'] = cont_loss
            total_loss += self.contrastive_weight * cont_loss
        
        # Genre prediction loss (auxiliary task)
        if genre_labels is not None and 'genre_logits' in outputs:
            genre_loss = self.genre_loss(
                outputs['genre_logits'],
                genre_labels.float()
            )
            losses['genre_loss'] = genre_loss
            total_loss += self.genre_weight * genre_loss
        
        # Network prediction loss (auxiliary task)
        if network_labels is not None and 'network_logits' in outputs:
            net_loss = self.network_loss(
                outputs['network_logits'],
                network_labels
            )
            losses['network_loss'] = net_loss
            total_loss += self.network_weight * net_loss
        
        losses['total_loss'] = total_loss
        return losses

# Model configuration for RTX 4090
def get_gnn_config():
    """Optimized GNN configuration for RTX 4090"""
    return {
        'hidden_dim': 512,
        'num_gnn_layers': 4,
        'dropout': 0.1,
        'use_metapath': True,
        'batch_size': 512,  # Large batch for graph operations
        'num_negative_samples': 5,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'gradient_clip': 1.0,
        'use_mixed_precision': True,  # Important for 4090
    }


# Alias for backward compatibility
TVGraphNeuralNetwork = TVGraphRecommender