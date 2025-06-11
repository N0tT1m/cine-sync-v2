use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;
use tch::{Device, Tensor};
use tracing::{info, debug, warn};

use crate::data::{MovieData, GenreFeatures};
use crate::model::{RecommendationModel, ModelMetadata, BatchInferenceInput};
use crate::error::AppError;

pub struct InferenceEngine {
    model: RecommendationModel,
    metadata: ModelMetadata,
    rating_scaler: Option<RatingScaler>,
    similarity_matrix: Option<HashMap<i64, HashMap<i64, f32>>>,
}

impl InferenceEngine {
    pub fn new<P: AsRef<Path>>(
        model: RecommendationModel,
        metadata: ModelMetadata,
        rating_scaler_path: P,
    ) -> Result<Self> {
        info!("Initializing inference engine");
        
        // Try to load rating scaler (for hybrid models)
        let rating_scaler = if metadata.model_type == "hybrid" {
            match RatingScaler::load_from_pickle(&rating_scaler_path) {
                Ok(scaler) => Some(scaler),
                Err(e) => {
                    warn!("Could not load rating scaler: {}. Using default scaling.", e);
                    Some(RatingScaler::default())
                }
            }
        } else {
            None
        };

        // Try to load similarity matrix (for content-based models)
        let similarity_matrix = if metadata.model_type == "content-based" {
            match Self::load_similarity_matrix() {
                Ok(matrix) => Some(matrix),
                Err(e) => {
                    warn!("Could not load similarity matrix: {}. Will use genre-based recommendations.", e);
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            model,
            metadata,
            rating_scaler,
            similarity_matrix,
        })
    }

    pub fn device(&self) -> Device {
        self.model.device()
    }

    pub async fn get_user_recommendations(
        &self,
        user_id: i64,
        top_k: usize,
        movie_data: &MovieData,
    ) -> Result<Vec<(i64, f32)>, AppError> {
        if self.metadata.model_type != "hybrid" {
            return Err(AppError::BadRequest(
                "User recommendations require hybrid model".to_string()
            ));
        }

        debug!("Getting recommendations for user {}", user_id);

        // Get a sample of movies to score (in production, you might want to be smarter about this)
        let candidate_movies: Vec<i64> = movie_data.movies.keys().cloned().collect();
        
        // Limit candidates to avoid memory issues
        let candidates: Vec<i64> = if candidate_movies.len() > 1000 {
            // Take every nth movie for efficiency
            let step = candidate_movies.len() / 1000;
            candidate_movies.into_iter().step_by(step).take(1000).collect()
        } else {
            candidate_movies
        };

        let mut batch_input = BatchInferenceInput::new();

        // Prepare batch input
        for &movie_id in &candidates {
            if let Some(movie) = movie_data.get_movie(movie_id) {
                let genre_features = GenreFeatures::from_movie(movie, &self.metadata.genres);
                batch_input.add_sample(user_id, movie_id, genre_features.to_vec());
            }
        }

        if batch_input.is_empty() {
            return Err(AppError::InternalError("No valid movies found".to_string()));
        }

        // Run inference
        let predictions = self.model.predict_batch(&batch_input)
            .map_err(|e| AppError::InternalError(format!("Inference failed: {}", e)))?;

        // Convert predictions to CPU and extract scores
        let predictions_cpu = predictions.to(Device::Cpu);
        let scores: Vec<f32> = predictions_cpu.try_into().unwrap_or_else(|_| vec![0.0; candidates.len()]);

        // Unscale ratings if we have a scaler
        let scores = if let Some(scaler) = &self.rating_scaler {
            scaler.inverse_transform(&scores)
        } else {
            scores
        };

        // Pair scores with movie IDs and sort
        let mut scored_movies: Vec<(i64, f32)> = candidates
            .into_iter()
            .zip(scores.into_iter())
            .collect();

        scored_movies.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top K
        Ok(scored_movies.into_iter().take(top_k).collect())
    }

    pub async fn get_content_based_recommendations(
        &self,
        seed_movie_ids: &[i64],
        top_k: usize,
        movie_data: &MovieData,
    ) -> Result<Vec<(i64, f32)>, AppError> {
        debug!("Getting content-based recommendations for {} seed movies", seed_movie_ids.len());

        // If we have a similarity matrix, use it
        if let Some(similarity_matrix) = &self.similarity_matrix {
            return self.get_similarity_based_recommendations(seed_movie_ids, top_k, movie_data);
        }

        // Otherwise, use genre-based recommendations
        self.get_genre_similarity_recommendations(seed_movie_ids, top_k, movie_data).await
    }

    pub async fn get_genre_based_recommendations(
        &self,
        genres: &[String],
        top_k: usize,
        movie_data: &MovieData,
    ) -> Result<Vec<(i64, f32)>, AppError> {
        debug!("Getting genre-based recommendations for genres: {:?}", genres);

        let mut scored_movies: Vec<(i64, f32)> = Vec::new();

        // Get movies for each genre and score them
        for genre in genres {
            let movie_ids = movie_data.get_movies_by_genre(genre);
            for movie_id in movie_ids {
                // Simple scoring: count how many requested genres the movie has
                if let Some(movie) = movie_data.get_movie(movie_id) {
                    let movie_genres: Vec<&str> = movie.genres.split('|').collect();
                    let score = genres.iter()
                        .filter(|g| movie_genres.contains(&g.as_str()))
                        .count() as f32 / genres.len() as f32;
                    
                    scored_movies.push((movie_id, score));
                }
            }
        }

        // Remove duplicates and sort
        scored_movies.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored_movies.dedup_by_key(|item| item.0);

        Ok(scored_movies.into_iter().take(top_k).collect())
    }

    fn get_similarity_based_recommendations(
        &self,
        seed_movie_ids: &[i64],
        top_k: usize,
        movie_data: &MovieData,
    ) -> Result<Vec<(i64, f32)>, AppError> {
        let similarity_matrix = self.similarity_matrix.as_ref().unwrap();
        let mut candidate_scores: HashMap<i64, f32> = HashMap::new();

        // For each seed movie, get similar movies
        for &seed_id in seed_movie_ids {
            if let Some(similarities) = similarity_matrix.get(&seed_id) {
                for (&movie_id, &similarity) in similarities {
                    // Don't recommend the seed movies themselves
                    if !seed_movie_ids.contains(&movie_id) {
                        *candidate_scores.entry(movie_id).or_insert(0.0) += similarity;
                    }
                }
            }
        }

        // Convert to sorted vector
        let mut scored_movies: Vec<(i64, f32)> = candidate_scores.into_iter().collect();
        scored_movies.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored_movies.into_iter().take(top_k).collect())
    }

    async fn get_genre_similarity_recommendations(
        &self,
        seed_movie_ids: &[i64],
        top_k: usize,
        movie_data: &MovieData,
    ) -> Result<Vec<(i64, f32)>, AppError> {
        // Collect genres from seed movies
        let mut seed_genres = std::collections::HashSet::new();
        for &movie_id in seed_movie_ids {
            if let Some(movie) = movie_data.get_movie(movie_id) {
                for genre in movie.genres.split('|') {
                    seed_genres.insert(genre.trim().to_string());
                }
            }
        }

        if seed_genres.is_empty() {
            return Err(AppError::InternalError("No genres found in seed movies".to_string()));
        }

        let seed_genres: Vec<String> = seed_genres.into_iter().collect();
        self.get_genre_based_recommendations(&seed_genres, top_k, movie_data).await
    }

    fn load_similarity_matrix() -> Result<HashMap<i64, HashMap<i64, f32>>> {
        // In a real implementation, you would load this from the pickle file
        // For now, return an empty matrix
        warn!("Similarity matrix loading not implemented. Using genre-based fallback.");
        Ok(HashMap::new())
    }
}

#[derive(Debug, Clone)]
pub struct RatingScaler {
    min_val: f32,
    max_val: f32,
    scale: f32,
}

impl RatingScaler {
    pub fn new(min_val: f32, max_val: f32) -> Self {
        let scale = max_val - min_val;
        Self {
            min_val,
            max_val,
            scale,
        }
    }

    pub fn transform(&self, ratings: &[f32]) -> Vec<f32> {
        ratings.iter()
            .map(|&rating| (rating - self.min_val) / self.scale)
            .collect()
    }

    pub fn inverse_transform(&self, scaled_ratings: &[f32]) -> Vec<f32> {
        scaled_ratings.iter()
            .map(|&scaled| scaled * self.scale + self.min_val)
            .collect()
    }

    pub fn load_from_pickle<P: AsRef<Path>>(_path: P) -> Result<Self> {
        // In a real implementation, you would load this from the pickle file
        // For now, return a default scaler for movie ratings (typically 0.5 to 5.0)
        warn!("Rating scaler loading from pickle not implemented. Using default scaler.");
        Ok(Self::default())
    }
}

impl Default for RatingScaler {
    fn default() -> Self {
        // Default scaler for movie ratings (0.5 to 5.0 -> 0.0 to 1.0)
        Self::new(0.5, 5.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rating_scaler() {
        let scaler = RatingScaler::new(0.5, 5.0);
        
        let ratings = vec![0.5, 2.75, 5.0];
        let scaled = scaler.transform(&ratings);
        let unscaled = scaler.inverse_transform(&scaled);
        
        assert!((scaled[0] - 0.0).abs() < 1e-6);
        assert!((scaled[1] - 0.5).abs() < 1e-6);
        assert!((scaled[2] - 1.0).abs() < 1e-6);
        
        for (original, recovered) in ratings.iter().zip(unscaled.iter()) {
            assert!((original - recovered).abs() < 1e-6);
        }
    }

    #[test]
    fn test_default_rating_scaler() {
        let scaler = RatingScaler::default();
        assert_eq!(scaler.min_val, 0.5);
        assert_eq!(scaler.max_val, 5.0);
        assert_eq!(scaler.scale, 4.5);
    }
}