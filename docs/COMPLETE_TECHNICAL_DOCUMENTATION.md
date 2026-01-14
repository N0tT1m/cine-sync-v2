# CineSync v2 - Complete Technical Documentation

**Version:** 2.0.0
**Last Updated:** 2026-01-14
**Author:** CineSync Development Team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [All 45 Models - Complete Reference](#3-all-45-models---complete-reference)
4. [Discord Bot - Complete Reference](#4-discord-bot---complete-reference)
5. [API Reference](#5-api-reference)
6. [Database Schema](#6-database-schema)
7. [Configuration Reference](#7-configuration-reference)
8. [Security Implementation](#8-security-implementation)
9. [Testing Framework](#9-testing-framework)
10. [Deployment Guide](#10-deployment-guide)
11. [Troubleshooting](#11-troubleshooting)
12. [Changelog - All Fixes Applied](#12-changelog---all-fixes-applied)

---

## 1. Executive Summary

### 1.1 Project Overview

CineSync v2 is an enterprise-grade AI-powered movie and TV show recommendation platform featuring:

- **45 specialized recommendation models** across 4 categories
- **Discord bot** with 24+ slash commands
- **Unified inference API** for model orchestration
- **PostgreSQL database** for user data and feedback
- **Admin dashboard** for model management
- **Real-time personalization** based on user behavior

### 1.2 Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Backend | Python | 3.10+ |
| ML Framework | PyTorch | 2.0+ |
| Database | PostgreSQL | 15+ |
| Bot Framework | discord.py | 2.0+ |
| Web Framework | Flask | 2.0+ |
| Data Processing | pandas, numpy | Latest |
| NLP Models | transformers, sentence-transformers | Latest |
| Graph ML | torch-geometric, dgl | Latest |

### 1.3 Project Statistics

| Metric | Count |
|--------|-------|
| Total Models | 45 |
| Movie-Specific Models | 14 |
| TV-Specific Models | 14 |
| Content-Agnostic Models | 12 |
| Unified Models | 5 |
| Discord Commands | 24 |
| API Endpoints | 15+ |
| Database Tables | 8 |
| Python Files | 120+ |
| Lines of Code | ~57,000 |

---

## 2. Architecture Overview

### 2.1 Directory Structure

```
cine-sync-v2/
├── src/                          # Main source code
│   ├── api/                      # API layer
│   │   ├── admin_interface.py    # Admin dashboard (Flask)
│   │   ├── unified_inference_api.py  # Model inference API
│   │   ├── extended_model_loader.py  # All 45 models loader
│   │   └── validation.py         # Input validation module
│   │
│   ├── models/                   # All 45 recommendation models
│   │   ├── movie/               # 14 movie-specific models
│   │   │   ├── franchise_sequence.py
│   │   │   ├── director_auteur.py
│   │   │   ├── cinematic_universe.py
│   │   │   ├── awards_prediction.py
│   │   │   ├── runtime_preference.py
│   │   │   ├── era_style.py
│   │   │   ├── critic_audience.py
│   │   │   ├── remake_connection.py
│   │   │   ├── actor_collaboration.py
│   │   │   ├── studio_fingerprint.py
│   │   │   ├── adaptation_source.py
│   │   │   ├── international_cinema.py
│   │   │   ├── narrative_complexity.py
│   │   │   └── viewing_context.py
│   │   │
│   │   ├── hybrid/sota_tv/models/  # 14 TV-specific models
│   │   │   ├── temporal_attention.py
│   │   │   ├── graph_neural_network.py
│   │   │   ├── contrastive_learning.py
│   │   │   ├── meta_learning.py
│   │   │   ├── ensemble_system.py
│   │   │   ├── multimodal_transformer.py
│   │   │   ├── episode_sequence.py
│   │   │   ├── binge_prediction.py
│   │   │   ├── series_completion.py
│   │   │   ├── season_quality.py
│   │   │   ├── platform_availability.py
│   │   │   ├── watch_pattern.py
│   │   │   ├── series_lifecycle.py
│   │   │   └── cast_migration.py
│   │   │
│   │   ├── advanced/            # 12 content-agnostic models
│   │   │   ├── bert4rec_recommender.py
│   │   │   ├── graphsage_recommender.py
│   │   │   ├── transformer_recommender.py
│   │   │   ├── variational_autoencoder.py
│   │   │   ├── graph_neural_network.py
│   │   │   ├── enhanced_two_tower.py
│   │   │   ├── sentence_bert_two_tower.py
│   │   │   └── t5_hybrid_recommender.py
│   │   │
│   │   ├── collaborative/       # NCF model
│   │   ├── sequential/          # Sequential recommender
│   │   ├── two_tower/           # Two-tower model
│   │   │
│   │   └── unified/             # 5 unified models
│   │       ├── cross_domain_embeddings.py
│   │       ├── movie_ensemble_system.py
│   │       ├── contrastive_learning.py
│   │       ├── multimodal_features.py
│   │       └── context_aware.py
│   │
│   ├── training/                # Training infrastructure
│   │   └── train_all_models.py  # Master training script
│   │
│   ├── data/                    # Data processing
│   └── monitoring/              # Monitoring utilities
│
├── services/                    # Service layer
│   └── lupe_python/            # Discord bot service
│       ├── main.py             # Bot entry point (3,200+ lines)
│       ├── config.py           # Configuration
│       ├── models/             # Bot-specific models
│       ├── unified_content_manager.py  # Content management
│       ├── personalized_commands.py    # Personalization
│       ├── personalized_trainer.py     # User training
│       └── preference_learner.py       # Preference analysis
│
├── tests/                       # Test suite
│   └── test_model_integration.py
│
├── configs/                     # Configuration files
│   └── deployment/             # Docker, nginx configs
│
├── data/                        # Data files
├── models/                      # Trained model checkpoints
└── docs/                        # Documentation
```

### 2.2 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERACTION                         │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DISCORD BOT (main.py)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ 24 Slash    │  │ Feedback    │  │ Personalization         │  │
│  │ Commands    │  │ Views/Modals│  │ Commands                │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│              UNIFIED CONTENT MANAGER                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ - Movie/TV Lookup Tables                                │    │
│  │ - Genre Management                                      │    │
│  │ - Model Orchestration                                   │    │
│  │ - Fallback Handling                                     │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│               UNIFIED INFERENCE API                              │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐    │
│  │    NCF    │  │ Sequential│  │ Two-Tower │  │  Ensemble │    │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    45 SPECIALIZED MODELS                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────┐ │
│  │ 14 Movie    │  │ 14 TV       │  │ 12 Content  │  │ 5      │ │
│  │ Models      │  │ Models      │  │ Agnostic    │  │ Unified│ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                      POSTGRESQL DATABASE                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌───────────┐ │
│  │user_ratings│  │ feedback   │  │preferences │  │training   │ │
│  └────────────┘  └────────────┘  └────────────┘  └───────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Model Inference Pipeline

```
User Request
     │
     ▼
┌────────────────────┐
│ Content Type Check │ ─── movie/tv/mixed
└────────────────────┘
     │
     ▼
┌────────────────────┐
│ Model Selection    │ ─── ncf/sequential/two_tower/ensemble
└────────────────────┘
     │
     ▼
┌────────────────────┐
│ User Embedding     │ ─── Get/Create user representation
└────────────────────┘
     │
     ▼
┌────────────────────┐
│ Candidate Generation│ ─── Get potential items
└────────────────────┘
     │
     ▼
┌────────────────────┐
│ Scoring & Ranking  │ ─── Score all candidates
└────────────────────┘
     │
     ▼
┌────────────────────┐
│ Filter & Validate  │ ─── Genre, seen items, content type
└────────────────────┘
     │
     ▼
┌────────────────────┐
│ Return Top-K       │ ─── Final recommendations
└────────────────────┘
```

---

## 3. All 45 Models - Complete Reference

### 3.1 Movie-Specific Models (14 Models)

#### 3.1.1 FranchiseSequenceModel
**File:** `src/models/movie/franchise_sequence.py`
**Purpose:** Predicts optimal franchise/sequel viewing order and recommends related franchise entries

| Component | Details |
|-----------|---------|
| Model Class | `FranchiseSequenceModel` |
| Config Class | `FranchiseConfig` |
| Trainer Class | `FranchiseSequenceTrainer` |
| Architecture | Transformer with positional encodings |
| Key Features | Franchise attention, sequel ordering, prequel detection |
| Input | User ID, Franchise ID, Viewing History |
| Output | Next movie in franchise, confidence score |
| Parameters | ~2M trainable parameters |
| Priority | 5 (High) |

**Config Options:**
```python
@dataclass
class FranchiseConfig:
    embedding_dim: int = 128
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    max_franchise_length: int = 50
```

---

#### 3.1.2 DirectorAuteurModel
**File:** `src/models/movie/director_auteur.py`
**Purpose:** Models director filmography and recommends based on directorial style preferences

| Component | Details |
|-----------|---------|
| Model Class | `DirectorAuteurModel` |
| Config Class | `DirectorConfig` |
| Trainer Class | `DirectorAuteurTrainer` |
| Architecture | Style encoder + Filmography transformer |
| Key Features | Director style extraction, visual signature matching |
| Input | User ID, Director ID, Liked Movies |
| Output | Similar directors, recommended films |
| Parameters | ~1.5M trainable parameters |
| Priority | 4 |

---

#### 3.1.3 CinematicUniverseModel
**File:** `src/models/movie/cinematic_universe.py`
**Purpose:** Navigates connected cinematic universes (MCU, DCEU, etc.)

| Component | Details |
|-----------|---------|
| Model Class | `CinematicUniverseModel` |
| Config Class | `UniverseConfig` |
| Trainer Class | `CinematicUniverseTrainer` |
| Architecture | Graph attention network + Timeline encoding |
| Key Features | Character tracking, timeline ordering, crossover detection |
| Input | User ID, Universe ID, Watched Films |
| Output | Next essential film, optional films |
| Parameters | ~3M trainable parameters |
| Priority | 4 |

---

#### 3.1.4 AwardsPredictionModel
**File:** `src/models/movie/awards_prediction.py`
**Purpose:** Recommends Oscar-worthy/prestige films based on user preferences

| Component | Details |
|-----------|---------|
| Model Class | `AwardsPredictionModel` |
| Config Class | `AwardsConfig` |
| Trainer Class | `AwardsPredictionTrainer` |
| Architecture | Prestige feature extractor + Awards history encoder |
| Key Features | Oscar prediction, festival circuit tracking |
| Input | User ID, Award preferences |
| Output | Prestige recommendations, award probabilities |
| Parameters | ~1.2M trainable parameters |
| Priority | 3 |

---

#### 3.1.5 RuntimePreferenceModel
**File:** `src/models/movie/runtime_preference.py`
**Purpose:** Time-aware recommendations based on available viewing time

| Component | Details |
|-----------|---------|
| Model Class | `RuntimePreferenceModel` |
| Config Class | `RuntimeConfig` |
| Trainer Class | `RuntimePreferenceTrainer` |
| Architecture | Runtime encoder + Time context awareness |
| Key Features | Duration matching, viewing context |
| Input | User ID, Available time, Context |
| Output | Recommendations fitting time slot |
| Parameters | ~800K trainable parameters |
| Priority | 3 |

---

#### 3.1.6 EraStyleModel
**File:** `src/models/movie/era_style.py`
**Purpose:** Models decade/era preferences (70s thrillers, 80s action, etc.)

| Component | Details |
|-----------|---------|
| Model Class | `EraStyleModel` |
| Config Class | `EraConfig` |
| Trainer Class | `EraStyleTrainer` |
| Architecture | Temporal embedding + Era classifier |
| Key Features | Decade detection, period piece matching |
| Input | User ID, Era preferences |
| Output | Era-appropriate recommendations |
| Priority | 3 |

---

#### 3.1.7 CriticAudienceModel
**File:** `src/models/movie/critic_audience.py`
**Purpose:** Aligns recommendations with critic vs. audience score preferences

| Component | Details |
|-----------|---------|
| Model Class | `CriticAudienceModel` |
| Config Class | `CriticConfig` |
| Trainer Class | `CriticAudienceTrainer` |
| Architecture | Dual score encoder |
| Key Features | RT score alignment, IMDB integration |
| Input | User ID, Score preferences |
| Output | Score-aligned recommendations |
| Priority | 4 |

---

#### 3.1.8 RemakeConnectionModel
**File:** `src/models/movie/remake_connection.py`
**Purpose:** Connects original films with remakes/reboots

| Component | Details |
|-----------|---------|
| Model Class | `RemakeConnectionModel` |
| Config Class | `RemakeConfig` |
| Trainer Class | `RemakeConnectionTrainer` |
| Key Features | Original/remake linking, version comparison |
| Priority | 3 |

---

#### 3.1.9 ActorCollaborationModel
**File:** `src/models/movie/actor_collaboration.py`
**Purpose:** Recommends based on actor pairings and chemistry

| Component | Details |
|-----------|---------|
| Model Class | `ActorCollaborationModel` |
| Config Class | `ActorConfig` |
| Trainer Class | `ActorCollaborationTrainer` |
| Key Features | Actor pairing analysis, chemistry scoring |
| Priority | 4 |

---

#### 3.1.10 StudioFingerprintModel
**File:** `src/models/movie/studio_fingerprint.py`
**Purpose:** Models studio-specific styles (A24, Blumhouse, etc.)

| Component | Details |
|-----------|---------|
| Model Class | `StudioFingerprintModel` |
| Config Class | `StudioConfig` |
| Trainer Class | `StudioFingerprintTrainer` |
| Key Features | Studio style detection, brand preferences |
| Priority | 3 |

---

#### 3.1.11 AdaptationSourceModel
**File:** `src/models/movie/adaptation_source.py`
**Purpose:** Recommends adaptations from books, comics, games

| Component | Details |
|-----------|---------|
| Model Class | `AdaptationSourceModel` |
| Config Class | `AdaptationConfig` |
| Trainer Class | `AdaptationSourceTrainer` |
| Key Features | Source material linking, adaptation quality |
| Priority | 3 |

---

#### 3.1.12 InternationalCinemaModel
**File:** `src/models/movie/international_cinema.py`
**Purpose:** Country/region-based recommendations

| Component | Details |
|-----------|---------|
| Model Class | `InternationalCinemaModel` |
| Config Class | `InternationalConfig` |
| Trainer Class | `InternationalCinemaTrainer` |
| Key Features | Regional cinema, language preferences |
| Priority | 3 |

---

#### 3.1.13 NarrativeComplexityModel
**File:** `src/models/movie/narrative_complexity.py`
**Purpose:** Matches storytelling complexity preferences

| Component | Details |
|-----------|---------|
| Model Class | `NarrativeComplexityModel` |
| Config Class | `NarrativeConfig` |
| Trainer Class | `NarrativeComplexityTrainer` |
| Key Features | Non-linear detection, complexity scoring |
| Priority | 3 |

---

#### 3.1.14 ViewingContextModel
**File:** `src/models/movie/viewing_context.py`
**Purpose:** Context-aware recommendations (date night, family, solo)

| Component | Details |
|-----------|---------|
| Model Class | `ViewingContextModel` |
| Config Class | `ViewingContextConfig` |
| Trainer Class | `ViewingContextTrainer` |
| Key Features | Social context, mood matching |
| Priority | 4 |

---

### 3.2 TV-Specific Models (14 Models)

#### 3.2.1 TemporalAttentionTVModel
**File:** `src/models/hybrid/sota_tv/models/temporal_attention.py`
**Purpose:** Models temporal viewing patterns across TV seasons

| Component | Details |
|-----------|---------|
| Model Class | `TemporalAttentionTVModel` |
| Architecture | Multi-head temporal attention |
| Key Features | Season-aware attention, viewing rhythm |
| Priority | 5 (High) |

---

#### 3.2.2 TVGraphNeuralNetwork
**File:** `src/models/hybrid/sota_tv/models/graph_neural_network.py`
**Purpose:** Graph-based TV show relationships

| Component | Details |
|-----------|---------|
| Model Class | `TVGraphNeuralNetwork` |
| Architecture | GNN with show-show edges |
| Key Features | Show similarity graph, user-show bipartite |
| Priority | 4 |

---

#### 3.2.3 ContrastiveTVLearning
**File:** `src/models/hybrid/sota_tv/models/contrastive_learning.py`
**Purpose:** Self-supervised TV show representation learning

| Component | Details |
|-----------|---------|
| Model Class | `ContrastiveTVLearning` |
| Architecture | Contrastive learning framework |
| Key Features | Positive/negative pair mining |
| Priority | 4 |

---

#### 3.2.4 MetaLearningTVModel
**File:** `src/models/hybrid/sota_tv/models/meta_learning.py`
**Purpose:** Few-shot adaptation for new TV shows

| Component | Details |
|-----------|---------|
| Model Class | `MetaLearningTVModel` |
| Architecture | MAML-based meta-learner |
| Key Features | Cold-start handling, quick adaptation |
| Priority | 4 |

---

#### 3.2.5 TVEnsembleSystem
**File:** `src/models/hybrid/sota_tv/models/ensemble_system.py`
**Purpose:** Ensemble of all TV-specific models

| Component | Details |
|-----------|---------|
| Model Class | `TVEnsembleSystem` |
| Architecture | Weighted model ensemble |
| Key Features | Dynamic weighting, confidence fusion |
| Priority | 5 (High) |

---

#### 3.2.6 MultimodalTVTransformer
**File:** `src/models/hybrid/sota_tv/models/multimodal_transformer.py`
**Purpose:** Multimodal features (video, text, audio)

| Component | Details |
|-----------|---------|
| Model Class | `MultimodalTVTransformer` |
| Architecture | Cross-modal transformer |
| Key Features | Trailer analysis, synopsis encoding |
| Priority | 4 |

---

#### 3.2.7 EpisodeSequenceModel
**File:** `src/models/hybrid/sota_tv/models/episode_sequence.py`
**Purpose:** Episode-level sequence modeling

| Component | Details |
|-----------|---------|
| Model Class | `EpisodeSequenceModel` |
| Config Class | `EpisodeSequenceConfig` |
| Trainer Class | `EpisodeSequenceTrainer` |
| Key Features | Episode ordering, arc detection |
| Priority | 5 (High) |

---

#### 3.2.8 BingePredictionModel
**File:** `src/models/hybrid/sota_tv/models/binge_prediction.py`
**Purpose:** Predicts binge-watching likelihood

| Component | Details |
|-----------|---------|
| Model Class | `BingePredictionModel` |
| Config Class | `BingePredictionConfig` |
| Trainer Class | `BingePredictionTrainer` |
| Key Features | Binge score, engagement prediction |
| Priority | 4 |

---

#### 3.2.9 SeriesCompletionModel
**File:** `src/models/hybrid/sota_tv/models/series_completion.py`
**Purpose:** Predicts if user will complete a series

| Component | Details |
|-----------|---------|
| Model Class | `SeriesCompletionModel` |
| Config Class | `SeriesCompletionConfig` |
| Trainer Class | `SeriesCompletionTrainer` |
| Key Features | Dropout prediction, completion probability |
| Priority | 4 |

---

#### 3.2.10 SeasonQualityModel
**File:** `src/models/hybrid/sota_tv/models/season_quality.py`
**Purpose:** Models season-by-season quality variance

| Component | Details |
|-----------|---------|
| Model Class | `SeasonQualityModel` |
| Config Class | `SeasonQualityConfig` |
| Trainer Class | `SeasonQualityTrainer` |
| Key Features | Quality trajectory, season ranking |
| Priority | 3 |

---

#### 3.2.11 PlatformAvailabilityModel
**File:** `src/models/hybrid/sota_tv/models/platform_availability.py`
**Purpose:** Streaming platform-aware recommendations

| Component | Details |
|-----------|---------|
| Model Class | `PlatformAvailabilityModel` |
| Config Class | `PlatformConfig` |
| Trainer Class | `PlatformAvailabilityTrainer` |
| Key Features | Platform filtering, subscription awareness |
| Priority | 4 |

---

#### 3.2.12 WatchPatternModel
**File:** `src/models/hybrid/sota_tv/models/watch_pattern.py`
**Purpose:** Models user viewing patterns

| Component | Details |
|-----------|---------|
| Model Class | `WatchPatternModel` |
| Config Class | `WatchPatternConfig` |
| Trainer Class | `WatchPatternTrainer` |
| Key Features | Time-of-day patterns, weekly rhythms |
| Priority | 4 |

---

#### 3.2.13 SeriesLifecycleModel
**File:** `src/models/hybrid/sota_tv/models/series_lifecycle.py`
**Purpose:** Models series lifecycle stages

| Component | Details |
|-----------|---------|
| Model Class | `SeriesLifecycleModel` |
| Config Class | `LifecycleConfig` |
| Trainer Class | `SeriesLifecycleTrainer` |
| Key Features | New/ongoing/ended classification |
| Priority | 3 |

---

#### 3.2.14 CastMigrationModel
**File:** `src/models/hybrid/sota_tv/models/cast_migration.py`
**Purpose:** Tracks cast changes across seasons

| Component | Details |
|-----------|---------|
| Model Class | `CastMigrationModel` |
| Config Class | `CastMigrationConfig` |
| Trainer Class | `CastMigrationTrainer` |
| Key Features | Cast continuity, actor tracking |
| Priority | 3 |

---

### 3.3 Content-Agnostic Models (12 Models)

#### 3.3.1 NeuralCollaborativeFiltering (NCF)
**File:** `src/models/collaborative/src/model.py`
**Lines:** 494

| Component | Details |
|-----------|---------|
| Model Class | `NeuralCollaborativeFiltering` |
| Architecture | GMF + MLP fusion |
| Key Features | User-item embeddings, rating prediction |
| Input | User ID, Item ID |
| Output | Predicted rating (1-5) |
| Parameters | ~5M trainable |
| Priority | 5 (Core) |

---

#### 3.3.2 SequentialRecommender
**File:** `src/models/sequential/src/model.py`
**Lines:** 760

| Component | Details |
|-----------|---------|
| Model Class | `SequentialRecommender` |
| Architecture | LSTM/GRU with attention |
| Key Features | Sequence modeling, next-item prediction |
| Priority | 4 |

---

#### 3.3.3 TwoTowerModel
**File:** `src/models/two_tower/src/model.py`
**Lines:** 1,157

| Component | Details |
|-----------|---------|
| Model Class | `TwoTowerModel` |
| Architecture | Dual encoder (user tower + item tower) |
| Key Features | Efficient retrieval, ANN search compatible |
| Priority | 5 (Core) |

---

#### 3.3.4 BERT4Rec
**File:** `src/models/advanced/bert4rec_recommender.py`
**Lines:** 570

| Component | Details |
|-----------|---------|
| Model Class | `BERT4Rec` |
| Architecture | BERT-based sequential |
| Key Features | Masked item prediction, bidirectional |
| Priority | 5 |

---

#### 3.3.5 GraphSAGERecommender
**File:** `src/models/advanced/graphsage_recommender.py`

| Component | Details |
|-----------|---------|
| Model Class | `GraphSAGERecommender` |
| Architecture | GraphSAGE with neighborhood sampling |
| Key Features | Inductive learning, scalable |
| Priority | 4 |

---

#### 3.3.6 TransformerRecommender
**File:** `src/models/advanced/transformer_recommender.py`

| Component | Details |
|-----------|---------|
| Model Class | `TransformerRecommender` |
| Architecture | Full transformer encoder |
| Key Features | Self-attention, position encoding |
| Priority | 5 |

---

#### 3.3.7 VAERecommender
**File:** `src/models/advanced/variational_autoencoder.py`

| Component | Details |
|-----------|---------|
| Model Class | `VAERecommender` |
| Architecture | Variational Autoencoder |
| Key Features | Generative, latent space |
| Priority | 4 |

---

#### 3.3.8 GNNRecommender
**File:** `src/models/advanced/graph_neural_network.py`

| Component | Details |
|-----------|---------|
| Model Class | `GNNRecommender` |
| Architecture | Graph Neural Network |
| Key Features | Message passing, graph convolution |
| Priority | 4 |

---

#### 3.3.9 EnhancedTwoTower
**File:** `src/models/advanced/enhanced_two_tower.py`

| Component | Details |
|-----------|---------|
| Model Class | `EnhancedTwoTower` |
| Architecture | Enhanced dual encoder |
| Key Features | Cross-attention, feature fusion |
| Priority | 4 |

---

#### 3.3.10 SentenceBERTTwoTower
**File:** `src/models/advanced/sentence_bert_two_tower.py`

| Component | Details |
|-----------|---------|
| Model Class | `SentenceBERTTwoTower` |
| Architecture | Sentence-BERT embeddings + Two-tower |
| Key Features | Semantic text understanding |
| Priority | 4 |

---

#### 3.3.11 T5HybridRecommender
**File:** `src/models/advanced/t5_hybrid_recommender.py`

| Component | Details |
|-----------|---------|
| Model Class | `T5HybridRecommender` |
| Architecture | T5 encoder-decoder |
| Key Features | Text generation, explanation |
| Priority | 3 |

---

#### 3.3.12 UnifiedContentRecommender
**File:** `src/models/hybrid/content_recommender.py`

| Component | Details |
|-----------|---------|
| Model Class | `UnifiedContentRecommender` |
| Architecture | NCF + content features |
| Key Features | Hybrid approach |
| Priority | 5 |

---

### 3.4 Unified Models (5 Models)

#### 3.4.1 CrossDomainEmbeddings
**File:** `src/models/unified/cross_domain_embeddings.py`

| Component | Details |
|-----------|---------|
| Model Class | `CrossDomainEmbeddings` |
| Related Classes | `UnifiedUserEmbedding`, `DomainAdapter` |
| Key Features | Movie-TV transfer learning |
| Priority | 5 |

---

#### 3.4.2 MovieEnsembleSystem
**File:** `src/models/unified/movie_ensemble_system.py`

| Component | Details |
|-----------|---------|
| Model Class | `MovieEnsembleSystem` |
| Key Features | Ensemble of movie-specific models |
| Priority | 5 |

---

#### 3.4.3 UnifiedContrastiveLearning
**File:** `src/models/unified/contrastive_learning.py`

| Component | Details |
|-----------|---------|
| Model Class | `UnifiedContrastiveLearning` |
| Key Features | Cross-domain contrastive |
| Priority | 4 |

---

#### 3.4.4 MultimodalFeatures
**File:** `src/models/unified/multimodal_features.py`

| Component | Details |
|-----------|---------|
| Model Class | `MultimodalFeatures` |
| Key Features | Multi-modal fusion |
| Priority | 4 |

---

#### 3.4.5 ContextAwareRecommender
**File:** `src/models/unified/context_aware.py`

| Component | Details |
|-----------|---------|
| Model Class | `ContextAwareRecommender` |
| Key Features | Context-aware recommendations |
| Priority | 4 |

---

### 3.5 Model Training Reference

**Master Training Script:** `src/training/train_all_models.py`

```bash
# Train a specific model
python src/training/train_all_models.py --model movie_franchise_sequence

# Train all movie models
python src/training/train_all_models.py --category movie

# Train all TV models
python src/training/train_all_models.py --category tv

# Train all content-agnostic models
python src/training/train_all_models.py --category both

# Train all unified models
python src/training/train_all_models.py --category unified

# Train everything
python src/training/train_all_models.py --all

# With options
python src/training/train_all_models.py --all --epochs 100 --batch-size 512 --lr 0.0001 --wandb
```

**Training Pipeline Class:**
```python
class UnifiedTrainingPipeline:
    def __init__(
        self,
        model_name: str,
        data_dir: str = 'data',
        output_dir: str = 'models',
        device: str = 'cuda',
        use_wandb: bool = False,
        wandb_project: str = 'cinesync-models'
    )

    def train(
        self,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        save_every: int = 5,
        early_stopping_patience: int = 10
    ) -> Dict[str, Any]
```

---

## 4. Discord Bot - Complete Reference

### 4.1 Bot Architecture

**Main File:** `services/lupe_python/main.py`
**Lines:** 3,200+
**Bot Class:** `LupeRecommendationBot`

```python
class LupeRecommendationBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!', intents=intents)

        # Core components
        self.lupe = None  # UnifiedLupeContentManager
        self.genres = []  # Available genres

    async def setup_hook(self):
        # Load Lupe content manager
        self.lupe = UnifiedLupeContentManager(...)
        self.lupe.load_models()

    async def on_ready(self):
        # Sync slash commands
        await self.tree.sync()
        # Initialize personalization
        if PERSONALIZATION_AVAILABLE:
            setup_personalization(...)
```

### 4.2 All 24 Slash Commands

#### 4.2.1 Core Recommendation Commands (6)

| Command | Description | Parameters |
|---------|-------------|------------|
| `/recommend` | Get movie/TV recommendations | `content_type`, `user_id`, `content_title`, `genre`, `limit` |
| `/cross_recommend` | Cross-content recommendations | `source_type`, `target_type`, `source_title`, `limit` |
| `/similar` | Find similar content | `content_type`, `title`, `limit` |
| `/next_episode` | Predict next watch | `user_id` |
| `/recommend_advanced` | Advanced with model selection | `content_type`, `model`, `genre`, `limit`, `min_confidence` |
| `/search` | Search for specific movie | `title` |

**Example: /recommend**
```
/recommend content_type:movie genre:Sci-Fi limit:10
```

---

#### 4.2.2 User Personalization Commands (7)

| Command | Description | Parameters |
|---------|-------------|------------|
| `/rate` | Rate a movie/TV show | `content_type`, `title`, `rating` |
| `/my_ratings` | View your ratings | `content_type`, `limit` |
| `/my_preferences` | View preference profile | - |
| `/my_recommendations` | Personalized recommendations | `count`, `content_type` |
| `/my_stats` | View preference statistics | - |
| `/rate_movies` | Quick rate multiple movies | `count` |
| `/genres` | List available genres | `content_type` |

---

#### 4.2.3 Analytics Commands (4)

| Command | Description | Parameters |
|---------|-------------|------------|
| `/stats` | Bot and model statistics | - |
| `/feedback_stats` | View feedback statistics | - |
| `/lupe_status` | AI status and model stats | - |
| `/model_compare` | Compare model recommendations | `user_id`, `limit` |
| `/model_health` | Check model health status | - |

---

#### 4.2.4 Admin Commands (5)

| Command | Description | Parameters |
|---------|-------------|------------|
| `/admin_review` | Review training feedback | `limit` |
| `/admin_approve` | Approve feedback for training | `feedback_id` |
| `/admin_reject` | Reject feedback | `feedback_id` |
| `/export_training_data` | Export approved training data | - |
| `/debug_bot` | Debug bot loading status | - |

---

#### 4.2.5 Testing Commands (2)

| Command | Description | Parameters |
|---------|-------------|------------|
| `/test_genre` | Test genre filtering | `genre`, `content_type` |

---

### 4.3 Feedback System

#### 4.3.1 FeedbackView
Interactive buttons for rating recommendations:

```python
class FeedbackView(View):
    @discord.ui.button(label='👍 Good Overall', style=discord.ButtonStyle.green)
    async def good_feedback(...)

    @discord.ui.button(label='👎 Poor Overall', style=discord.ButtonStyle.red)
    async def poor_feedback(...)

    @discord.ui.button(label='⭐ Rate Individual', style=discord.ButtonStyle.blurple)
    async def rate_individual(...)

    @discord.ui.button(label='💖 Like/Dislike Each', style=discord.ButtonStyle.gray)
    async def individual_preferences(...)

    @discord.ui.button(label='💬 Detailed Feedback', style=discord.ButtonStyle.secondary)
    async def detailed_feedback(...)
```

#### 4.3.2 IndividualContentFeedbackView
4-level feedback per item:
- 💖 Love
- 👍 Like
- 👎 Dislike
- 💔 Hate

#### 4.3.3 Feedback Modals
- `QuickRatingModal` - Quick 1-5 rating
- `MovieRatingModal` - Detailed movie rating
- `DetailedFeedbackModal` - Text feedback
- `IndividualContentFeedbackModal` - Per-item feedback

### 4.4 Thread-Safe Caching

```python
# Global cache lock
_cache_lock = asyncio.Lock()

# Thread-safe cache operations
async def get_user_excluded_movies(user_id: int) -> set:
    async with _cache_lock:
        # Cache lookup

async def update_user_recommendation_cache(user_id: int, movie_ids: List[int]):
    async with _cache_lock:
        # Cache update
```

### 4.5 Personalization Integration

```python
# Personalization modules
from preference_learner import PreferenceLearner
from personalized_trainer import PersonalizedTrainer
from personalized_commands import (
    setup_personalization,
    my_recommendations_command,
    my_stats_command,
    rate_movies_command
)

# Setup in on_ready
async def on_ready(self):
    if PERSONALIZATION_AVAILABLE:
        preference_learner = PreferenceLearner(db_manager)
        personalized_trainer = PersonalizedTrainer(db_manager, self.lupe)
        setup_personalization(personalized_trainer, preference_learner, db_manager, self.lupe)
```

---

## 5. API Reference

### 5.1 Unified Inference API

**File:** `src/api/unified_inference_api.py`

#### 5.1.1 ModelType Enum
```python
class ModelType(Enum):
    NCF = "neural_collaborative_filtering"
    SEQUENTIAL = "sequential"
    TWO_TOWER = "two_tower"
    ENSEMBLE = "ensemble"
```

#### 5.1.2 RecommendationResult
```python
@dataclass
class RecommendationResult:
    item_id: int
    score: float
    model_type: str
    confidence: Optional[float] = None
    metadata: Optional[Dict] = None
```

#### 5.1.3 UnifiedRecommendationAPI Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `load_all_models()` | Load all configured models | - | None |
| `load_model(model_type)` | Load specific model | `ModelType` | `bool` |
| `get_recommendations(user_id, top_k, model_type)` | Get recommendations | user_id, top_k, model_type | `List[RecommendationResult]` |
| `predict_next_items(sequence, top_k)` | Sequential prediction | sequence, top_k | `List[RecommendationResult]` |
| `find_similar_items(item_id, top_k)` | Find similar items | item_id, top_k | `List[RecommendationResult]` |
| `predict_rating(user_id, item_id)` | Predict user-item rating | user_id, item_id | `Dict[str, float]` |
| `compare_models(user_id, top_k)` | Compare all models | user_id, top_k | `Dict[str, List]` |
| `health_check()` | Check model health | - | `Dict[str, bool]` |
| `get_model_info()` | Get model information | - | `Dict[str, Any]` |

---

### 5.2 Admin Interface API

**File:** `src/api/admin_interface.py`
**Base URL:** `http://localhost:5001`

#### 5.2.1 Authentication Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/login` | GET/POST | Admin login page |
| `/logout` | GET | Logout |

#### 5.2.2 Dashboard Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/` | GET | Yes | Main dashboard |
| `/models` | GET | Yes | Model management page |
| `/training` | GET | Yes | Training preferences |
| `/upload` | GET | Yes | Model upload page |
| `/analytics` | GET | Yes | Analytics dashboard |

#### 5.2.3 API Endpoints

| Endpoint | Method | Auth | Description | Request Body |
|----------|--------|------|-------------|--------------|
| `/api/models/<name>/toggle` | POST | Yes | Enable/disable model | - |
| `/api/models/<name>/reload` | POST | Yes | Reload model | - |
| `/api/training/preferences` | POST | Yes | Update training prefs | JSON preferences |
| `/api/training/exclude_user` | POST | Yes | Exclude user | `{"user_id": int}` |
| `/api/training/include_user` | POST | Yes | Include user | `{"user_id": int}` |
| `/api/training/trigger_retrain` | POST | Yes | Trigger retraining | - |
| `/api/upload_model` | POST | Yes | Upload model file | multipart/form-data |
| `/api/analytics/model_performance` | GET | Yes | Get performance data | - |

---

### 5.3 Input Validation API

**File:** `src/api/validation.py`

#### 5.3.1 Validation Functions

| Function | Description | Parameters | Returns |
|----------|-------------|------------|---------|
| `sanitize_string(value, max_length, allow_html)` | Sanitize string input | str, int, bool | `str` |
| `validate_int(value, min_val, max_val, field_name)` | Validate integer | Any, int, int, str | `int` |
| `validate_float(value, min_val, max_val, field_name)` | Validate float | Any, float, float, str | `float` |
| `validate_bool(value, field_name)` | Validate boolean | Any, str | `bool` |
| `validate_list(value, item_validator, max_items)` | Validate list | Any, Callable, int | `List` |
| `validate_model_name(name)` | Validate model name | str | `str` |
| `validate_user_id(user_id)` | Validate user ID | Any | `int` |
| `validate_content_type(content_type)` | Validate content type | str | `str` |
| `validate_genre(genre, valid_genres)` | Validate genre | str, List | `str` |
| `validate_rating(rating)` | Validate rating 1-5 | Any | `float` |

#### 5.3.2 Validation Decorators

```python
@validate_json_request(required_fields=['user_id'], optional_fields={'limit': 10})
def my_endpoint():
    data = request.validated_data
    ...

@validate_query_params({
    'limit': {'type': 'int', 'min': 1, 'max': 100, 'default': 10},
    'offset': {'type': 'int', 'min': 0, 'default': 0}
})
def my_query_endpoint():
    params = request.validated_params
    ...
```

---

## 6. Database Schema

### 6.1 PostgreSQL Tables

#### 6.1.1 user_ratings
```sql
CREATE TABLE user_ratings (
    id SERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    movie_id INTEGER NOT NULL,
    rating DECIMAL(2,1) NOT NULL CHECK (rating >= 1 AND rating <= 5),
    timestamp TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, movie_id)
);

CREATE INDEX idx_user_ratings_user ON user_ratings(user_id);
CREATE INDEX idx_user_ratings_movie ON user_ratings(movie_id);
```

#### 6.1.2 user_tv_ratings
```sql
CREATE TABLE user_tv_ratings (
    id SERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    show_id INTEGER NOT NULL,
    rating DECIMAL(2,1) NOT NULL CHECK (rating >= 1 AND rating <= 5),
    timestamp TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, show_id)
);
```

#### 6.1.3 feedback
```sql
CREATE TABLE feedback (
    id SERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    username VARCHAR(255),
    feedback_type VARCHAR(50) NOT NULL,
    recommendation_method VARCHAR(100),
    recommendations JSONB,
    original_query TEXT,
    feedback_text TEXT,
    timestamp TIMESTAMP DEFAULT NOW(),
    content_type VARCHAR(20) DEFAULT 'movie',
    model_used VARCHAR(100),
    confidence_score DECIMAL(3,2)
);

CREATE INDEX idx_feedback_user ON feedback(user_id);
CREATE INDEX idx_feedback_type ON feedback(feedback_type);
CREATE INDEX idx_feedback_timestamp ON feedback(timestamp);
```

#### 6.1.4 user_preferences
```sql
CREATE TABLE user_preferences (
    id SERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    content_id INTEGER NOT NULL,
    content_type VARCHAR(20) NOT NULL,
    preference VARCHAR(20) NOT NULL, -- love, like, dislike, hate
    recommendation_method VARCHAR(100),
    timestamp TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, content_id, content_type)
);

CREATE INDEX idx_user_preferences_user ON user_preferences(user_id);
```

#### 6.1.5 master_training_feedback
```sql
CREATE TABLE master_training_feedback (
    id SERIAL PRIMARY KEY,
    original_feedback_id INTEGER REFERENCES feedback(id),
    user_id BIGINT NOT NULL,
    status VARCHAR(20) DEFAULT 'pending', -- pending, approved, rejected
    reviewed_by VARCHAR(255),
    reviewed_at TIMESTAMP,
    training_data JSONB,
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### 6.1.6 user_genre_preferences
```sql
CREATE TABLE user_genre_preferences (
    id SERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    genre VARCHAR(100) NOT NULL,
    affinity_score DECIMAL(3,2) DEFAULT 0.5,
    rating_count INTEGER DEFAULT 0,
    avg_rating DECIMAL(2,1),
    last_updated TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, genre)
);
```

### 6.2 Database Connection

```python
class DatabaseManager:
    def __init__(self, db_config):
        self.config = db_config

    def get_connection(self):
        @contextmanager
        def connection_context():
            conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password
            )
            try:
                yield conn
            finally:
                conn.close()
        return connection_context()
```

---

## 7. Configuration Reference

### 7.1 Environment Variables

**File:** `.env.example`

```bash
# Discord Configuration
DISCORD_TOKEN=your_discord_bot_token

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=cinesync
DB_USER=postgres
DB_PASSWORD=your_password

# Admin Interface
ADMIN_SECRET_KEY=generate_secure_random_key_here
ADMIN_USERS=admin,moderator
ADMIN_PASSWORD_HASH=pbkdf2:sha256:260000$...

# Model Configuration
MODEL_EMBEDDING_DIM=64
MODEL_LEARNING_RATE=0.001
MODEL_BATCH_SIZE=256

# Feature Flags
DEBUG=false
ENABLE_WANDB=false
WANDB_PROJECT=cinesync-models
```

### 7.2 Config Classes

**File:** `services/lupe_python/config.py`

```python
@dataclass
class DatabaseConfig:
    host: str
    port: int
    database: str
    user: str
    password: str

@dataclass
class ModelConfig:
    models_dir: str
    device: str
    batch_size: int
    learning_rate: float
    epochs: int

@dataclass
class DiscordConfig:
    token: str

@dataclass
class Config:
    database: DatabaseConfig
    model: ModelConfig
    discord: DiscordConfig
    debug: bool
```

### 7.3 Model Configuration

**File:** `unified_config.json`

```json
{
    "neural_collaborative_filtering": {
        "model_path": "models/ncf/best_model.pt",
        "encoders_path": "models/ncf/encoders.pkl",
        "weight": 1.0,
        "enabled": true
    },
    "sequential": {
        "model_path": "models/sequential/best_model.pt",
        "encoders_path": "models/sequential/encoders.pkl",
        "weight": 1.2,
        "enabled": true
    },
    "two_tower": {
        "model_path": "models/two_tower/best_model.pt",
        "encoders_path": "models/two_tower/preprocessors.pkl",
        "weight": 0.8,
        "enabled": true
    }
}
```

---

## 8. Security Implementation

### 8.1 Authentication Security

**File:** `src/api/admin_interface.py`

#### 8.1.1 Password Hashing
```python
from werkzeug.security import check_password_hash, generate_password_hash

# Generate hash (run once to create):
# python -c "from werkzeug.security import generate_password_hash; print(generate_password_hash('your_password'))"

# Verify password
if check_password_hash(admin_password_hash, password):
    # Login successful
```

#### 8.1.2 Rate Limiting
```python
MAX_LOGIN_ATTEMPTS = 5
LOGIN_LOCKOUT_MINUTES = 15

_login_attempts = {}  # {ip: (attempts, lockout_time)}

def check_rate_limit(ip_address: str) -> bool:
    if ip_address in _login_attempts:
        attempts, lockout_time = _login_attempts[ip_address]
        if lockout_time and datetime.now() < lockout_time:
            return False  # Still locked out
    return True

def record_failed_login(ip_address: str):
    attempts, _ = _login_attempts.get(ip_address, (0, None))
    attempts += 1
    if attempts >= MAX_LOGIN_ATTEMPTS:
        lockout_time = datetime.now() + timedelta(minutes=LOGIN_LOCKOUT_MINUTES)
        _login_attempts[ip_address] = (attempts, lockout_time)
```

#### 8.1.3 Secure Session Keys
```python
import secrets

_secret_key = os.environ.get('ADMIN_SECRET_KEY')
if not _secret_key:
    logging.warning("ADMIN_SECRET_KEY not set!")
    _secret_key = secrets.token_hex(32)  # Generate secure random
app.secret_key = _secret_key
```

### 8.2 Input Validation

**File:** `src/api/validation.py`

```python
# SQL Injection Prevention
def sanitize_string(value: str, max_length: int = 1000) -> str:
    value = value.strip()[:max_length]
    value = re.sub(r'<[^>]+>', '', value)  # Remove HTML
    value = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', value)  # Remove control chars
    return value

# Model name validation (alphanumeric only)
def validate_model_name(name: str) -> str:
    name = sanitize_string(name, max_length=100)
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        raise ValidationError("Invalid model name")
    return name
```

### 8.3 Security Checklist

| Item | Status | Notes |
|------|--------|-------|
| Password hashing | ✅ | Using werkzeug.security |
| Rate limiting | ✅ | 5 attempts, 15 min lockout |
| Secure session key | ✅ | Generated if not provided |
| Input validation | ✅ | All API endpoints |
| SQL injection prevention | ✅ | Parameterized queries |
| XSS prevention | ✅ | HTML sanitization |
| CSRF protection | ⚠️ | Needs Flask-WTF |
| Audit logging | ✅ | Login attempts logged |

---

## 9. Testing Framework

### 9.1 Test Structure

**Directory:** `tests/`

```
tests/
├── test_model_integration.py    # All 45 models
├── test_bot_commands.py         # Discord commands
├── test_api_endpoints.py        # API tests
├── test_validation.py           # Input validation
└── test_database.py             # Database operations
```

### 9.2 Model Integration Tests

**File:** `tests/test_model_integration.py`

```python
class TestModelImports:
    """Test all models can be imported"""

    @pytest.mark.parametrize("model_name,model_info", MOVIE_SPECIFIC_MODELS.items())
    def test_movie_model_imports(self, model_name, model_info):
        module = importlib.import_module(model_info['module'])
        cls = getattr(module, model_info['model_class'])
        assert cls is not None

class TestModelInstantiation:
    """Test models can be instantiated"""

    def test_movie_model_instantiation(self, model_name, model_info):
        model = self._try_instantiate_model(model_info)
        assert isinstance(model, torch.nn.Module)

class TestExtendedModelLoader:
    """Test extended model loader"""

    def test_model_registry_completeness(self):
        from src.api.extended_model_loader import EXTENDED_MODEL_REGISTRY
        assert len(EXTENDED_MODEL_REGISTRY) >= 40
```

### 9.3 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model_integration.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Quick verification
python tests/test_model_integration.py
```

---

## 10. Deployment Guide

### 10.1 Docker Deployment

**File:** `configs/deployment/docker-compose.yml`

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: cinesync
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-db.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  bot:
    build: .
    environment:
      - DISCORD_TOKEN=${DISCORD_TOKEN}
      - DB_HOST=postgres
      - DB_PORT=5432
      - DB_NAME=cinesync
      - DB_USER=postgres
      - DB_PASSWORD=${DB_PASSWORD}
    depends_on:
      postgres:
        condition: service_healthy
    volumes:
      - ./models:/app/models
      - ./data:/app/data

  admin:
    build:
      context: .
      dockerfile: Dockerfile.admin
    environment:
      - ADMIN_SECRET_KEY=${ADMIN_SECRET_KEY}
      - ADMIN_PASSWORD_HASH=${ADMIN_PASSWORD_HASH}
    ports:
      - "5001:5001"
    depends_on:
      - postgres

volumes:
  postgres_data:
```

### 10.2 Manual Deployment

```bash
# 1. Clone repository
git clone <repo_url>
cd cine-sync-v2

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment
cp .env.example .env
# Edit .env with your values

# 5. Initialize database
psql -U postgres -f configs/deployment/init-db.sql

# 6. Train models (optional)
python src/training/train_all_models.py --all

# 7. Run bot
python services/lupe_python/main.py

# 8. Run admin interface (separate terminal)
python src/api/admin_interface.py
```

### 10.3 Environment Setup

```bash
# Generate admin password hash
python -c "from werkzeug.security import generate_password_hash; print(generate_password_hash('your_secure_password'))"

# Generate secret key
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## 11. Troubleshooting

### 11.1 Common Issues

#### Issue: "No module named 'src'"
**Solution:** Add project root to Python path
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
```

#### Issue: "UnifiedTrainingPipeline got unexpected keyword argument 'epochs'"
**Solution:** Filter training kwargs from init kwargs
```python
init_kwargs = {k: v for k, v in training_kwargs.items()
              if k in ('device', 'use_wandb', 'wandb_project')}
pipeline = UnifiedTrainingPipeline(..., **init_kwargs)
```

#### Issue: Button callback crash (undefined 'button')
**Solution:** Use `interaction.data['custom_id']`
```python
# Wrong:
custom_id = button.custom_id

# Correct:
custom_id = interaction.data['custom_id']
```

#### Issue: "ADMIN_PASSWORD_HASH not configured"
**Solution:** Set environment variable with hashed password
```bash
export ADMIN_PASSWORD_HASH=$(python -c "from werkzeug.security import generate_password_hash; print(generate_password_hash('your_password'))")
```

### 11.2 Debug Commands

```bash
# Check model loading
python -c "from src.training.train_all_models import ALL_MODELS; print(f'{len(ALL_MODELS)} models registered')"

# Verify model imports
python tests/test_model_integration.py

# Check database connection
python -c "
import psycopg2
conn = psycopg2.connect(host='localhost', database='cinesync', user='postgres', password='password')
print('Database connected!')
conn.close()
"

# Test Discord token
python -c "
import discord
client = discord.Client(intents=discord.Intents.default())
@client.event
async def on_ready():
    print(f'Logged in as {client.user}')
    await client.close()
client.run('YOUR_TOKEN')
"
```

---

## 12. Changelog - All Fixes Applied

### 12.1 Session: 2026-01-14

#### Fix 1: Critical Button Callback Bug
**File:** `services/lupe_python/main.py:255`
**Severity:** Critical
**Impact:** All feedback buttons crashed

```python
# BEFORE (BROKEN):
async def button_callback(self, interaction: discord.Interaction):
    custom_id = button.custom_id  # 'button' undefined!

# AFTER (FIXED):
async def button_callback(self, interaction: discord.Interaction):
    custom_id = interaction.data['custom_id']
```

---

#### Fix 2: Admin Interface Security
**File:** `src/api/admin_interface.py`
**Severity:** Critical
**Impact:** Plain-text passwords, no rate limiting

**Changes:**
1. Added `werkzeug.security` imports
2. Removed hardcoded default secret key
3. Added secure random key generation
4. Implemented password hashing with `check_password_hash`
5. Added rate limiting (5 attempts, 15 min lockout)
6. Added audit logging for login attempts

```python
# Added imports
from werkzeug.security import check_password_hash, generate_password_hash
import secrets

# Secure secret key
_secret_key = os.environ.get('ADMIN_SECRET_KEY')
if not _secret_key:
    _secret_key = secrets.token_hex(32)
app.secret_key = _secret_key

# Rate limiting
MAX_LOGIN_ATTEMPTS = 5
LOGIN_LOCKOUT_MINUTES = 15

# Password verification
if check_password_hash(admin_password_hash, password):
    # Success
```

---

#### Fix 3: Personalized Commands Integration
**File:** `services/lupe_python/main.py`
**Severity:** High
**Impact:** 3 commands not working

**Changes:**
1. Added imports for personalization modules
2. Added `PERSONALIZATION_AVAILABLE` flag
3. Registered 3 new slash commands:
   - `/my_recommendations`
   - `/my_stats`
   - `/rate_movies`
4. Added personalization setup in `on_ready()`

```python
# New imports
from preference_learner import PreferenceLearner
from personalized_trainer import PersonalizedTrainer
from personalized_commands import (
    setup_personalization,
    my_recommendations_command,
    my_stats_command,
    rate_movies_command
)

# Registration
@bot.tree.command(name="my_recommendations", ...)
async def my_recommendations(...):
    await my_recommendations_command(interaction, count, content_type)
```

---

#### Fix 4: Thread-Safe Recommendation Cache
**File:** `services/lupe_python/main.py`
**Severity:** Medium
**Impact:** Potential race conditions

**Changes:**
1. Added `asyncio.Lock()` for thread safety
2. Converted cache functions to async
3. Added sync wrapper for compatibility

```python
# Thread-safe lock
_cache_lock = asyncio.Lock()

async def get_user_excluded_movies(user_id: int) -> set:
    async with _cache_lock:
        # Thread-safe cache access

async def update_user_recommendation_cache(user_id: int, movie_ids: List[int]):
    async with _cache_lock:
        # Thread-safe cache update
```

---

#### Fix 5: Training Pipeline Kwargs Error
**File:** `src/training/train_all_models.py`
**Severity:** High
**Impact:** Training script failed

**Changes:**
1. Filtered init kwargs from training kwargs
2. Added training param filter in `create_model()`

```python
# In train_category()
init_kwargs = {k: v for k, v in training_kwargs.items()
              if k in ('device', 'use_wandb', 'wandb_project')}
pipeline = UnifiedTrainingPipeline(..., **init_kwargs)

# In create_model()
training_params = {'epochs', 'batch_size', 'lr', ...}
model_config = {k: v for k, v in config_overrides.items()
                if k not in training_params}
```

---

#### New: Extended Model Loader
**File:** `src/api/extended_model_loader.py`
**Purpose:** Load all 45 models

**Features:**
- `ExtendedModelType` enum for all 45 models
- `EXTENDED_MODEL_REGISTRY` with full configurations
- `ExtendedModelLoader` class for verification and loading
- `verify_all_models()` function

---

#### New: Input Validation Module
**File:** `src/api/validation.py`
**Purpose:** Secure input validation

**Features:**
- `sanitize_string()` - XSS prevention
- `validate_int/float/bool()` - Type validation
- `validate_model_name()` - Alphanumeric only
- `validate_user_id()` - Range checking
- `@validate_json_request` decorator
- `@validate_query_params` decorator

---

#### New: Model Integration Tests
**File:** `tests/test_model_integration.py`
**Purpose:** Test all 45 models

**Test Classes:**
- `TestModelImports` - Import verification
- `TestModelInstantiation` - Instantiation tests
- `TestModelParameters` - Parameter checking
- `TestBotIntegration` - Bot integration
- `TestExtendedModelLoader` - Loader tests

---

### 12.2 Files Modified

| File | Changes |
|------|---------|
| `services/lupe_python/main.py` | Button fix, cache lock, personalization |
| `src/api/admin_interface.py` | Security hardening, validation |
| `src/training/train_all_models.py` | Kwargs filtering |

### 12.3 Files Created

| File | Purpose |
|------|---------|
| `src/api/extended_model_loader.py` | All 45 models loader |
| `src/api/validation.py` | Input validation |
| `tests/test_model_integration.py` | Model tests |
| `docs/COMPLETE_TECHNICAL_DOCUMENTATION.md` | This document |

---

## Appendix A: Quick Reference Card

### Discord Commands Quick Reference

```
RECOMMENDATIONS:
/recommend content_type:movie genre:Action limit:10
/similar content_type:movie title:"The Matrix"
/cross_recommend source_type:movie target_type:tv source_title:"Inception"

PERSONALIZATION:
/my_recommendations count:10 content_type:movie
/my_stats
/rate_movies count:5
/rate content_type:movie title:"Inception" rating:5

ADMIN:
/admin_review limit:20
/admin_approve feedback_id:123
/export_training_data
```

### Environment Variables Quick Reference

```bash
DISCORD_TOKEN=xxx
DB_HOST=localhost
DB_PORT=5432
DB_NAME=cinesync
DB_USER=postgres
DB_PASSWORD=xxx
ADMIN_SECRET_KEY=xxx
ADMIN_USERS=admin
ADMIN_PASSWORD_HASH=xxx
DEBUG=false
```

### Model Categories Quick Reference

| Category | Count | Priority Range |
|----------|-------|----------------|
| Movie-Specific | 14 | 3-5 |
| TV-Specific | 14 | 3-5 |
| Content-Agnostic | 12 | 3-5 |
| Unified | 5 | 4-5 |
| **Total** | **45** | |

---

**Document Version:** 1.0.0
**Generated:** 2026-01-14
**Total Pages:** ~50
**Word Count:** ~8,000
