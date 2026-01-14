"""
Multimodal Feature Extraction for CineSync v2
Extracts rich features from text, images, audio, and metadata
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoModel, AutoTokenizer
import numpy as np


class TextFeatureExtractor(nn.Module):
    """
    Extract features from text (plot summaries, reviews, descriptions).

    Uses pre-trained language models (BERT, RoBERTa, etc.) and fine-tunes
    for recommendation domain.
    """

    def __init__(
        self,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        output_dim: int = 512,
        freeze_base: bool = False,
        use_pooling: str = 'mean'  # 'mean', 'cls', 'max'
    ):
        super().__init__()

        self.output_dim = output_dim
        self.use_pooling = use_pooling

        # Load pre-trained model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        # Freeze base model if specified
        if freeze_base:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Get embedding dimension from model
        self.base_dim = self.encoder.config.hidden_size

        # Projection layer to output dimension
        self.projection = nn.Sequential(
            nn.Linear(self.base_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # Optional attention pooling
        if use_pooling == 'attention':
            self.attention_pooling = nn.Sequential(
                nn.Linear(self.base_dim, 1),
                nn.Softmax(dim=1)
            )

    def forward(
        self,
        text: Union[str, List[str]],
        max_length: int = 256
    ) -> torch.Tensor:
        """
        Extract features from text.

        Args:
            text: Single text string or list of texts
            max_length: Maximum sequence length

        Returns:
            Text embeddings [batch_size, output_dim]
        """
        # Tokenize
        if isinstance(text, str):
            text = [text]

        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        # Move to same device as model
        device = next(self.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        # Get embeddings
        outputs = self.encoder(**encoded)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]

        # Pool embeddings
        if self.use_pooling == 'mean':
            # Mean pooling (with attention to padding)
            attention_mask = encoded['attention_mask'].unsqueeze(-1)
            embeddings = (hidden_states * attention_mask).sum(1) / attention_mask.sum(1)
        elif self.use_pooling == 'cls':
            # Use [CLS] token
            embeddings = hidden_states[:, 0, :]
        elif self.use_pooling == 'max':
            # Max pooling
            embeddings = hidden_states.max(1)[0]
        elif self.use_pooling == 'attention':
            # Attention pooling
            attention_weights = self.attention_pooling(hidden_states)
            embeddings = (hidden_states * attention_weights).sum(1)
        else:
            raise ValueError(f"Unknown pooling: {self.use_pooling}")

        # Project to output dimension
        embeddings = self.projection(embeddings)

        return embeddings

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> torch.Tensor:
        """Encode texts in batches for efficiency"""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_emb = self.forward(batch_texts)
            all_embeddings.append(batch_emb)

        return torch.cat(all_embeddings, dim=0)


class VisualFeatureExtractor(nn.Module):
    """
    Extract features from images (posters, thumbnails, frames).

    Uses pre-trained vision models (ResNet, ViT, CLIP) and fine-tunes
    for recommendation domain.
    """

    def __init__(
        self,
        model_type: str = 'resnet50',  # 'resnet50', 'vit', 'clip'
        output_dim: int = 512,
        freeze_base: bool = False,
        pretrained: bool = True
    ):
        super().__init__()

        self.output_dim = output_dim
        self.model_type = model_type

        if model_type == 'resnet50':
            # Use ResNet50 - requires torchvision
            try:
                import torchvision.models as models
                self.encoder = models.resnet50(pretrained=pretrained)
                # Remove final FC layer
                self.base_dim = self.encoder.fc.in_features
                self.encoder.fc = nn.Identity()
            except ImportError:
                # Fallback to simple CNN if torchvision not available
                self.base_dim = output_dim
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(3, stride=2, padding=1),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(128, output_dim)
                )

        elif model_type == 'vit':
            # Use Vision Transformer
            try:
                from transformers import ViTModel
                self.encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
                self.base_dim = self.encoder.config.hidden_size
            except Exception:
                self.base_dim = output_dim
                self.encoder = nn.Identity()

        elif model_type == 'clip':
            # Use CLIP vision encoder
            try:
                from transformers import CLIPVisionModel
                self.encoder = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32')
                self.base_dim = self.encoder.config.hidden_size
            except Exception:
                self.base_dim = output_dim
                self.encoder = nn.Identity()

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Freeze base model if specified
        if freeze_base:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.base_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.

        Args:
            images: Image tensor [batch_size, 3, H, W]

        Returns:
            Visual embeddings [batch_size, output_dim]
        """
        if self.model_type in ['vit', 'clip']:
            # Transformer-based models
            outputs = self.encoder(pixel_values=images)
            embeddings = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0]
        else:
            # CNN-based models (ResNet)
            embeddings = self.encoder(images)

        # Project
        embeddings = self.projection(embeddings)

        return embeddings


class AudioFeatureExtractor(nn.Module):
    """
    Extract features from audio (soundtracks, trailers).

    Uses pre-trained audio models or mel-spectrograms.
    """

    def __init__(
        self,
        output_dim: int = 512,
        use_pretrained: bool = True
    ):
        super().__init__()

        self.output_dim = output_dim

        if use_pretrained:
            # Use a pre-trained audio model (e.g., Wav2Vec2)
            from transformers import Wav2Vec2Model
            self.encoder = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
            self.base_dim = self.encoder.config.hidden_size
        else:
            # Use mel-spectrogram + CNN
            self.encoder = self._build_spectrogram_encoder()
            self.base_dim = 512

        # Projection
        self.projection = nn.Sequential(
            nn.Linear(self.base_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def _build_spectrogram_encoder(self):
        """Build CNN encoder for mel-spectrograms"""
        return nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),
            nn.Linear(256, 512)
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract features from audio.

        Args:
            audio: Audio waveform [batch_size, num_samples] or
                   Mel-spectrogram [batch_size, 1, n_mels, time]

        Returns:
            Audio embeddings [batch_size, output_dim]
        """
        if audio.dim() == 2:
            # Waveform input - use Wav2Vec2
            outputs = self.encoder(audio)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        else:
            # Spectrogram input - use CNN
            embeddings = self.encoder(audio)

        # Project
        embeddings = self.projection(embeddings)

        return embeddings


class MetadataEncoder(nn.Module):
    """
    Encode metadata features (year, rating, popularity, etc.).

    Handles both categorical and numerical metadata.
    """

    def __init__(
        self,
        categorical_features: Dict[str, int],  # {feature_name: vocab_size}
        numerical_features: List[str],
        embedding_dim: int = 64,
        output_dim: int = 512
    ):
        super().__init__()

        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.output_dim = output_dim

        # Categorical embeddings
        self.categorical_embeddings = nn.ModuleDict()
        total_cat_dim = 0

        for feat_name, vocab_size in categorical_features.items():
            emb_dim = min(embedding_dim, (vocab_size + 1) // 2)
            self.categorical_embeddings[feat_name] = nn.Embedding(vocab_size, emb_dim)
            total_cat_dim += emb_dim

        # Numerical feature encoder
        num_numerical = len(numerical_features)
        self.numerical_encoder = nn.Sequential(
            nn.Linear(num_numerical, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        ) if num_numerical > 0 else None

        # Combine categorical and numerical
        total_input_dim = total_cat_dim + (embedding_dim if num_numerical > 0 else 0)

        self.fusion = nn.Sequential(
            nn.Linear(total_input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(
        self,
        categorical_data: Dict[str, torch.Tensor],
        numerical_data: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode metadata.

        Args:
            categorical_data: Dict of {feature_name: tensor of IDs}
            numerical_data: Numerical features [batch_size, num_features]

        Returns:
            Metadata embeddings [batch_size, output_dim]
        """
        features = []

        # Encode categorical features
        for feat_name, feat_values in categorical_data.items():
            if feat_name in self.categorical_embeddings:
                emb = self.categorical_embeddings[feat_name](feat_values)
                features.append(emb)

        # Encode numerical features
        if numerical_data is not None and self.numerical_encoder is not None:
            num_emb = self.numerical_encoder(numerical_data)
            features.append(num_emb)

        # Concatenate and fuse
        combined = torch.cat(features, dim=-1)
        output = self.fusion(combined)

        return output


class MultimodalFusion(nn.Module):
    """
    Fuse features from multiple modalities.

    Supports:
    - Early fusion: concatenate then process
    - Late fusion: process separately then combine
    - Attention fusion: learn importance of each modality
    - Gated fusion: learn gating mechanism
    """

    def __init__(
        self,
        input_dims: Dict[str, int],  # {modality_name: dim}
        output_dim: int = 512,
        fusion_type: str = 'attention',  # 'early', 'late', 'attention', 'gated'
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dims = input_dims
        self.output_dim = output_dim
        self.fusion_type = fusion_type

        if fusion_type == 'early':
            # Concatenate then process
            total_dim = sum(input_dims.values())
            self.fusion = nn.Sequential(
                nn.Linear(total_dim, output_dim * 2),
                nn.LayerNorm(output_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim * 2, output_dim),
                nn.LayerNorm(output_dim)
            )

        elif fusion_type == 'late':
            # Process separately then combine
            self.modality_encoders = nn.ModuleDict()
            for modality, dim in input_dims.items():
                self.modality_encoders[modality] = nn.Sequential(
                    nn.Linear(dim, output_dim),
                    nn.LayerNorm(output_dim),
                    nn.GELU()
                )

            self.final_fusion = nn.Sequential(
                nn.Linear(output_dim * len(input_dims), output_dim),
                nn.LayerNorm(output_dim)
            )

        elif fusion_type == 'attention':
            # Attention-based fusion
            self.modality_projections = nn.ModuleDict()
            for modality, dim in input_dims.items():
                self.modality_projections[modality] = nn.Linear(dim, output_dim)

            self.attention = nn.MultiheadAttention(
                output_dim, num_heads=8, dropout=dropout, batch_first=True
            )

            self.norm = nn.LayerNorm(output_dim)

        elif fusion_type == 'gated':
            # Gated fusion
            self.modality_encoders = nn.ModuleDict()
            self.modality_gates = nn.ModuleDict()

            for modality, dim in input_dims.items():
                self.modality_encoders[modality] = nn.Sequential(
                    nn.Linear(dim, output_dim),
                    nn.GELU()
                )

                self.modality_gates[modality] = nn.Sequential(
                    nn.Linear(dim, output_dim),
                    nn.Sigmoid()
                )

            self.final_fusion = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

    def forward(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse multimodal features.

        Args:
            modality_features: Dict of {modality_name: features}

        Returns:
            Fused features [batch_size, output_dim]
        """
        if self.fusion_type == 'early':
            # Concatenate all features
            features = [modality_features[mod] for mod in self.input_dims.keys()
                       if mod in modality_features]
            combined = torch.cat(features, dim=-1)
            return self.fusion(combined)

        elif self.fusion_type == 'late':
            # Process each modality separately
            encoded = []
            for modality in self.input_dims.keys():
                if modality in modality_features:
                    enc = self.modality_encoders[modality](modality_features[modality])
                    encoded.append(enc)

            # Combine
            combined = torch.cat(encoded, dim=-1)
            return self.final_fusion(combined)

        elif self.fusion_type == 'attention':
            # Project all modalities to same dimension
            projected = []
            for modality in self.input_dims.keys():
                if modality in modality_features:
                    proj = self.modality_projections[modality](modality_features[modality])
                    projected.append(proj)

            # Stack for attention
            stacked = torch.stack(projected, dim=1)  # [batch, num_modalities, dim]

            # Self-attention
            attended, _ = self.attention(stacked, stacked, stacked)

            # Mean pooling
            fused = attended.mean(dim=1)

            return self.norm(fused)

        elif self.fusion_type == 'gated':
            # Gated fusion
            gated_features = []
            for modality in self.input_dims.keys():
                if modality in modality_features:
                    feat = modality_features[modality]
                    encoded = self.modality_encoders[modality](feat)
                    gate = self.modality_gates[modality](feat)
                    gated = encoded * gate
                    gated_features.append(gated)

            # Sum gated features
            combined = torch.stack(gated_features, dim=0).sum(dim=0)

            return self.final_fusion(combined)


class CompleteMultimodalEncoder(nn.Module):
    """
    Complete multimodal encoder that handles all feature types.

    Combines text, visual, audio, and metadata into a unified representation.
    """

    def __init__(
        self,
        text_output_dim: int = 512,
        visual_output_dim: int = 512,
        audio_output_dim: int = 512,
        metadata_output_dim: int = 512,
        final_output_dim: int = 768,
        fusion_type: str = 'attention'
    ):
        super().__init__()

        # Individual encoders
        self.text_encoder = TextFeatureExtractor(output_dim=text_output_dim)
        self.visual_encoder = VisualFeatureExtractor(output_dim=visual_output_dim)
        self.audio_encoder = AudioFeatureExtractor(output_dim=audio_output_dim)

        # Multimodal fusion
        self.fusion = MultimodalFusion(
            input_dims={
                'text': text_output_dim,
                'visual': visual_output_dim,
                'audio': audio_output_dim,
                'metadata': metadata_output_dim
            },
            output_dim=final_output_dim,
            fusion_type=fusion_type
        )

    def forward(
        self,
        text: Optional[List[str]] = None,
        images: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        metadata_encoder: Optional[nn.Module] = None,
        metadata_inputs: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Extract and fuse multimodal features.

        Args:
            text: Text descriptions
            images: Image tensors
            audio: Audio tensors
            metadata_encoder: Metadata encoder module
            metadata_inputs: Metadata inputs

        Returns:
            Fused multimodal features
        """
        features = {}

        if text is not None:
            features['text'] = self.text_encoder(text)

        if images is not None:
            features['visual'] = self.visual_encoder(images)

        if audio is not None:
            features['audio'] = self.audio_encoder(audio)

        if metadata_encoder is not None and metadata_inputs is not None:
            features['metadata'] = metadata_encoder(**metadata_inputs)

        # Fuse all available features
        fused = self.fusion(features)

        return fused


# Example usage
if __name__ == "__main__":
    # Text features
    text_encoder = TextFeatureExtractor(output_dim=512)
    texts = ["A thrilling action movie", "A romantic comedy"]
    text_features = text_encoder(texts)
    print(f"Text features shape: {text_features.shape}")

    # Metadata
    metadata_encoder = MetadataEncoder(
        categorical_features={'genre': 20, 'rating': 5},
        numerical_features=['year', 'popularity', 'runtime'],
        output_dim=512
    )

    cat_data = {
        'genre': torch.tensor([5, 10]),
        'rating': torch.tensor([3, 4])
    }
    num_data = torch.tensor([[2020, 0.8, 120], [2021, 0.9, 90]], dtype=torch.float)

    metadata_features = metadata_encoder(cat_data, num_data)
    print(f"Metadata features shape: {metadata_features.shape}")

    # Multimodal fusion
    fusion = MultimodalFusion(
        input_dims={'text': 512, 'metadata': 512},
        output_dim=768,
        fusion_type='attention'
    )

    fused = fusion({'text': text_features, 'metadata': metadata_features})
    print(f"Fused features shape: {fused.shape}")

# Alias for backward compatibility
MultimodalFeatures = CompleteMultimodalEncoder
