use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::debug;

#[derive(Clone)]
pub struct LupeApiClient {
    client: Client,
    base_url: String,
}

impl LupeApiClient {
    pub fn new(base_url: String, timeout: Duration) -> Result<Self> {
        let client = Client::builder()
            .timeout(timeout)
            .user_agent("Lupe-Discord-Bot/1.0")
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to create HTTP client: {}", e))?;

        Ok(Self { client, base_url })
    }

    pub async fn health_check(&self) -> Result<HealthResponse> {
        let url = format!("{}/health", self.base_url);
        debug!("Health check: {}", url);

        let response = self
            .client
            .get(&url)
            .query(&[("detailed", "true")])
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send health check request: {}", e))?;

        let health: HealthResponse = response
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse health response: {}", e))?;

        Ok(health)
    }

    pub async fn get_user_recommendations(
        &self,
        user_id: u64,
        top_k: Option<usize>,
    ) -> Result<RecommendationResponse> {
        let url = format!("{}/recommend", self.base_url);
        debug!("Getting user recommendations for user {}", user_id);

        let request = UserRecommendationRequest {
            user_id: Some(user_id as i64),
            top_k: top_k.or(Some(10)),
            movie_ids: None,
            genres: None,
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send recommendation request: {}", e))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow::anyhow!("API error: {}", error_text));
        }

        let recommendations: RecommendationResponse = response
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse recommendation response: {}", e))?;

        Ok(recommendations)
    }

    pub async fn get_genre_recommendations(
        &self,
        genres: Vec<String>,
        top_k: Option<usize>,
    ) -> Result<RecommendationResponse> {
        let url = format!("{}/recommend", self.base_url);
        debug!("Getting genre recommendations for: {:?}", genres);

        let request = GenreRecommendationRequest {
            genres: Some(genres),
            top_k: top_k.or(Some(10)),
            user_id: None,
            movie_ids: None,
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send genre recommendation request: {}", e))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow::anyhow!("API error: {}", error_text));
        }

        let recommendations: RecommendationResponse = response
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse genre recommendation response: {}", e))?;

        Ok(recommendations)
    }

    pub async fn get_similar_movie_recommendations(
        &self,
        movie_ids: Vec<i64>,
        top_k: Option<usize>,
    ) -> Result<RecommendationResponse> {
        let url = format!("{}/recommend", self.base_url);
        debug!("Getting similar movie recommendations for: {:?}", movie_ids);

        let request = SimilarMovieRequest {
            movie_ids: Some(movie_ids),
            top_k: top_k.or(Some(10)),
            user_id: None,
            genres: None,
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send similar movie request: {}", e))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow::anyhow!("API error: {}", error_text));
        }

        let recommendations: RecommendationResponse = response
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse similar movie response: {}", e))?;

        Ok(recommendations)
    }

    pub async fn get_all_genres(&self) -> Result<Vec<String>> {
        let url = format!("{}/genres", self.base_url);
        debug!("Getting all genres");

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to get genres: {}", e))?;

        let genres: Vec<String> = response
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse genres response: {}", e))?;

        Ok(genres)
    }

    pub async fn search_movies(&self, query: &str) -> Result<Vec<Movie>> {
        // Note: This endpoint might not exist in your Rust server yet
        // You might need to add it or implement client-side search
        let url = format!("{}/movies", self.base_url);
        debug!("Searching movies with query: {}", query);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to search movies: {}", e))?;

        let all_movies: Vec<Movie> = response
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse movies response: {}", e))?;

        // Simple client-side filtering
        let query_lower = query.to_lowercase();
        let filtered_movies: Vec<Movie> = all_movies
            .into_iter()
            .filter(|movie| {
                movie.title.to_lowercase().contains(&query_lower) ||
                movie.genres.to_lowercase().contains(&query_lower)
            })
            .take(20) // Limit results
            .collect();

        Ok(filtered_movies)
    }
}

// API Request/Response types
#[derive(Serialize)]
struct UserRecommendationRequest {
    user_id: Option<i64>,
    top_k: Option<usize>,
    movie_ids: Option<Vec<i64>>,
    genres: Option<Vec<String>>,
}

#[derive(Serialize)]
struct GenreRecommendationRequest {
    genres: Option<Vec<String>>,
    top_k: Option<usize>,
    user_id: Option<i64>,
    movie_ids: Option<Vec<i64>>,
}

#[derive(Serialize)]
struct SimilarMovieRequest {
    movie_ids: Option<Vec<i64>>,
    top_k: Option<usize>,
    user_id: Option<i64>,
    genres: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub model_loaded: bool,
    pub model_type: Option<String>,
    pub device: String,
    pub movies_count: usize,
    pub genres_count: usize,
    pub uptime_seconds: u64,
    pub version: String,
}

#[derive(Debug, Deserialize)]
pub struct RecommendationResponse {
    pub recommendations: Vec<MovieRecommendation>,
    pub request_id: String,
    pub model_type: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct MovieRecommendation {
    pub movie_id: i64,
    pub title: String,
    pub genres: String,
    pub score: f32,
    pub rank: usize,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Movie {
    pub media_id: i64,
    pub title: String,
    pub genres: String,
}

// Utility functions
impl MovieRecommendation {
    pub fn format_score(&self) -> String {
        format!("{:.1}%", self.score * 100.0)
    }

    pub fn format_genres(&self) -> String {
        self.genres.replace('|', ", ")
    }

    pub fn get_primary_genre(&self) -> String {
        self.genres
            .split('|')
            .next()
            .unwrap_or("Unknown")
            .to_string()
    }
}

impl Movie {
    pub fn format_genres(&self) -> String {
        self.genres.replace('|', ", ")
    }
}