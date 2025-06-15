import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering model combining Generalized Matrix Factorization (GMF)
    and Multi-Layer Perceptron (MLP) for enhanced recommendation performance.
    
    Paper: He, X., et al. "Neural collaborative filtering." WWW 2017.
    """
    
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_layers=[128, 64], 
                 dropout=0.2, alpha=0.5):
        super(NeuralCollaborativeFiltering, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.alpha = alpha  # Weight for combining GMF and MLP
        
        # GMF (Generalized Matrix Factorization) embeddings
        self.gmf_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP embeddings (different from GMF to allow different representations)
        self.mlp_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        mlp_layers = []
        input_size = embedding_dim * 2  # User + Item embeddings
        
        for hidden_size in hidden_layers:
            mlp_layers.append(nn.Linear(input_size, hidden_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_size = hidden_size
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Final prediction layer
        # GMF output: embedding_dim, MLP output: last hidden layer size
        final_input_size = embedding_dim + hidden_layers[-1]
        self.prediction = nn.Linear(final_input_size, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, user_ids, item_ids):
        # GMF path
        gmf_user_emb = self.gmf_user_embedding(user_ids)
        gmf_item_emb = self.gmf_item_embedding(item_ids)
        gmf_output = gmf_user_emb * gmf_item_emb  # Element-wise product
        
        # MLP path
        mlp_user_emb = self.mlp_user_embedding(user_ids)
        mlp_item_emb = self.mlp_item_embedding(item_ids)
        mlp_input = torch.cat([mlp_user_emb, mlp_item_emb], dim=-1)
        mlp_output = self.mlp(mlp_input)
        
        # Combine GMF and MLP
        combined = torch.cat([gmf_output, mlp_output], dim=-1)
        prediction = self.prediction(combined)
        
        return torch.sigmoid(prediction).squeeze() * 5.0  # Scale to 0-5 rating range


class SimpleNCF(nn.Module):
    """
    Simplified Neural Collaborative Filtering model for faster training and inference.
    """
    
    def __init__(self, num_users, num_items, embedding_dim=32, hidden_dim=64, dropout=0.2):
        super(SimpleNCF, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        x = torch.cat([user_emb, item_emb], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return torch.sigmoid(x).squeeze() * 5.0


class DeepNCF(nn.Module):
    """
    Enhanced Neural Collaborative Filtering with additional features and deeper architecture.
    """
    
    def __init__(self, num_users, num_items, num_genres=20, embedding_dim=128, 
                 hidden_layers=[256, 128, 64], dropout=0.3, use_batch_norm=True):
        super(DeepNCF, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.genre_embedding = nn.Embedding(num_genres, embedding_dim // 4)
        
        # Deep network
        layers = []
        input_size = embedding_dim * 2 + (embedding_dim // 4)  # User + Item + Genre
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_size = hidden_size
        
        self.network = nn.Sequential(*layers)
        self.prediction = nn.Linear(hidden_layers[-1], 1)
        
        # Attention mechanism for genre importance
        self.genre_attention = nn.Linear(embedding_dim // 4, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.xavier_normal_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, user_ids, item_ids, genre_ids=None):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        if genre_ids is not None:
            # Handle multiple genres per item (assume genre_ids is padded)
            genre_emb = self.genre_embedding(genre_ids)  # [batch, max_genres, emb_dim]
            
            # Apply attention to genres
            genre_attn = F.softmax(self.genre_attention(genre_emb), dim=1)
            genre_emb = (genre_emb * genre_attn).sum(dim=1)  # Weighted sum
        else:
            genre_emb = torch.zeros(user_emb.size(0), self.genre_embedding.embedding_dim, 
                                  device=user_emb.device)
        
        x = torch.cat([user_emb, item_emb, genre_emb], dim=-1)
        x = self.network(x)
        x = self.prediction(x)
        
        return torch.sigmoid(x).squeeze() * 5.0