-- Initialize CineSync Database Schema
-- This script runs automatically when the PostgreSQL container starts

\echo 'Creating CineSync database schema...'

-- Create feedback table
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    username TEXT,
    feedback_type TEXT NOT NULL,
    movie_id INTEGER,
    movie_title TEXT,
    rating INTEGER,
    recommendation_method TEXT,
    original_query TEXT,
    feedback_text TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create user_ratings table  
CREATE TABLE IF NOT EXISTS user_ratings (
    id SERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    movie_id INTEGER NOT NULL,
    rating REAL NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, movie_id)
);

-- Create core movies table
CREATE TABLE IF NOT EXISTS movies (
    media_id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    genres TEXT,
    year INTEGER,
    tmdb_id INTEGER,
    imdb_id TEXT,
    overview TEXT,
    poster_path TEXT,
    backdrop_path TEXT,
    vote_average REAL,
    vote_count INTEGER,
    popularity REAL,
    release_date DATE,
    original_language TEXT,
    original_title TEXT,
    runtime INTEGER,
    budget BIGINT,
    revenue BIGINT,
    status TEXT,
    tagline TEXT
);

-- Create core ratings table
CREATE TABLE IF NOT EXISTS ratings (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    media_id INTEGER NOT NULL,
    rating REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, media_id)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_user_ratings_user_id ON user_ratings(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_user_id ON feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type);
CREATE INDEX IF NOT EXISTS idx_movies_title ON movies(title);
CREATE INDEX IF NOT EXISTS idx_movies_year ON movies(year);
CREATE INDEX IF NOT EXISTS idx_movies_genres ON movies(genres);
CREATE INDEX IF NOT EXISTS idx_ratings_user_id ON ratings(user_id);
CREATE INDEX IF NOT EXISTS idx_ratings_media_id ON ratings(media_id);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

\echo 'CineSync database schema created successfully!'