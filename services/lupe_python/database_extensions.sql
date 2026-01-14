-- Database Extensions for Personalization System
-- CineSync v2 - Discord AI Agent
-- Created: 2025-10-25

-- User embeddings table
-- Stores learned vector representations of each user's preferences
CREATE TABLE IF NOT EXISTS user_embeddings (
    user_id BIGINT PRIMARY KEY,
    embedding BYTEA NOT NULL,  -- Numpy array serialized with pickle
    embedding_dim INTEGER NOT NULL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    rating_count INTEGER DEFAULT 0,
    feedback_count INTEGER DEFAULT 0
);

-- User preferences table (analyzed patterns)
-- Stores extracted preference patterns from user interactions
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id BIGINT PRIMARY KEY,
    favorite_genres JSONB,
    favorite_directors JSONB,
    favorite_actors JSONB,
    preferred_decades JSONB,
    avg_rating REAL,
    rating_distribution JSONB,
    diversity_score REAL,
    last_analyzed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User model weights (which models work best for each user)
-- Allows per-user ensemble optimization
CREATE TABLE IF NOT EXISTS user_model_weights (
    user_id BIGINT PRIMARY KEY,
    ncf_weight REAL DEFAULT 1.0,
    sequential_weight REAL DEFAULT 1.0,
    two_tower_weight REAL DEFAULT 1.0,
    bert4rec_weight REAL DEFAULT 1.0,
    graphsage_weight REAL DEFAULT 1.0,
    ensemble_performance JSONB,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_user_embeddings_updated ON user_embeddings(last_updated);
CREATE INDEX IF NOT EXISTS idx_user_preferences_updated ON user_preferences(last_analyzed);
CREATE INDEX IF NOT EXISTS idx_user_model_weights_updated ON user_model_weights(last_updated);

-- Comments for documentation
COMMENT ON TABLE user_embeddings IS 'Stores learned vector representations of user preferences for real-time personalization';
COMMENT ON TABLE user_preferences IS 'Stores analyzed preference patterns (genres, directors, actors) extracted from user ratings';
COMMENT ON TABLE user_model_weights IS 'Stores per-user model ensemble weights for optimized recommendations';

COMMENT ON COLUMN user_embeddings.embedding IS 'Serialized numpy array representing user in embedding space';
COMMENT ON COLUMN user_embeddings.embedding_dim IS 'Dimensionality of the embedding vector';
COMMENT ON COLUMN user_embeddings.rating_count IS 'Number of ratings used to build this embedding';
COMMENT ON COLUMN user_embeddings.feedback_count IS 'Number of feedback interactions (incremental updates)';

COMMENT ON COLUMN user_preferences.diversity_score IS 'Entropy-based score (0-1) measuring variety in user preferences';
COMMENT ON COLUMN user_preferences.rating_distribution IS 'Histogram of user rating behavior (generous vs harsh)';
