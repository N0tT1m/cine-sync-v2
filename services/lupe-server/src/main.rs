use anyhow::{Context, Result};
use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, net::SocketAddr, path::PathBuf, sync::Arc};
use tch::{Device, Tensor, TchError};
use tower_http::cors::CorsLayer;
use tracing::{info, warn, error};

mod model;
mod data;
mod inference;
mod error;

use model::RecommendationModel;
use data::{MovieData, ModelMetadata};
use inference::InferenceEngine;
use error::AppError;

#[derive(Parser)]
#[command(name = "movie-recommendation-server")]
#[command(about = "A Rust server for hosting PyTorch movie recommendation models")]
struct Args {
    /// Path to the models directory
    #[arg(short, long, default_value = "models")]
    models_path: PathBuf,
    
    /// Port to bind the server to
    #[arg(short, long, default_value = "3000")]
    port: u16,
    
    /// Host to bind the server to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    
    /// Use CPU only (disable CUDA)
    #[arg(long)]
    cpu_only: bool,
}

#[derive(Clone)]
struct AppState {
    inference_engine: Arc<InferenceEngine>,
    movie_data: Arc<MovieData>,
    metadata: Arc<ModelMetadata>,
}

#[derive(Deserialize)]
struct RecommendationRequest {
    user_id: Option<i64>,
    movie_ids: Option<Vec<i64>>,
    genres: Option<Vec<String>>,
    top_k: Option<usize>,
}

#[derive(Serialize)]
struct RecommendationResponse {
    recommendations: Vec<MovieRecommendation>,
    request_id: String,
    model_type: String,
}

#[derive(Serialize)]
struct MovieRecommendation {
    movie_id: i64,
    title: String,
    genres: String,
    score: f32,
    rank: usize,
}

#[derive(Deserialize)]
struct HealthQuery {
    detailed: Option<bool>,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    model_loaded: bool,
    model_type: Option<String>,
    device: String,
    movies_count: usize,
    genres_count: usize,
    uptime_seconds: u64,
    version: String,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
    request_id: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    info!("Starting movie recommendation server");
    info!("Models path: {:?}", args.models_path);
    info!("Server will bind to {}:{}", args.host, args.port);

    // Determine device
    let device = if args.cpu_only {
        info!("Using CPU device (forced)");
        Device::Cpu
    } else {
        match Device::cuda_if_available() {
            Device::Cuda(_) => {
                info!("CUDA available, using GPU");
                Device::cuda_if_available()
            }
            Device::Cpu => {
                info!("CUDA not available, using CPU");
                Device::Cpu
            }
        }
    };

    // Load model and data
    info!("Loading model and data...");
    let app_state = load_app_state(&args.models_path, device).await?;
    
    info!("Model loaded successfully");
    info!("Model type: {}", app_state.metadata.model_type);
    info!("Movies count: {}", app_state.movie_data.movies.len());
    info!("Genres count: {}", app_state.metadata.genres.len());

    // Build the application router
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/recommend", post(recommend_handler))
        .route("/movies", get(list_movies_handler))
        .route("/genres", get(list_genres_handler))
        .layer(CorsLayer::permissive())
        .with_state(app_state);

    // Start the server
    let addr = format!("{}:{}", args.host, args.port)
        .parse::<SocketAddr>()
        .context("Invalid address")?;

    info!("Server listening on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn load_app_state(models_path: &PathBuf, device: Device) -> Result<AppState> {
    // Load metadata
    let metadata_path = models_path.join("model_metadata.pkl");
    let metadata = ModelMetadata::load_from_pickle(&metadata_path)
        .context("Failed to load model metadata")?;

    // Load movie data
    let movie_lookup_path = models_path.join("movie_lookup.pkl");
    let movies_csv_path = models_path.join("movies_data.csv");
    let movie_data = MovieData::load(&movie_lookup_path, &movies_csv_path)
        .context("Failed to load movie data")?;

    // Load model
    let model_path = models_path.join("recommendation_model.pt");
    let rating_scaler_path = models_path.join("rating_scaler.pkl");
    
    let model = RecommendationModel::load(&model_path, device)
        .context("Failed to load PyTorch model")?;

    // Create inference engine
    let inference_engine = InferenceEngine::new(
        model,
        metadata.clone(),
        rating_scaler_path,
    )?;

    Ok(AppState {
        inference_engine: Arc::new(inference_engine),
        movie_data: Arc::new(movie_data),
        metadata: Arc::new(metadata),
    })
}

async fn health_handler(
    Query(params): Query<HealthQuery>,
    State(state): State<AppState>,
) -> Result<Json<HealthResponse>, AppError> {
    let detailed = params.detailed.unwrap_or(false);
    
    let response = HealthResponse {
        status: "healthy".to_string(),
        model_loaded: true,
        model_type: Some(state.metadata.model_type.clone()),
        device: format!("{:?}", state.inference_engine.device()),
        movies_count: state.movie_data.movies.len(),
        genres_count: state.metadata.genres.len(),
        uptime_seconds: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    };

    Ok(Json(response))
}

async fn recommend_handler(
    State(state): State<AppState>,
    Json(request): Json<RecommendationRequest>,
) -> Result<Json<RecommendationResponse>, AppError> {
    let request_id = uuid::Uuid::new_v4().to_string();
    
    info!("Processing recommendation request: {}", request_id);

    let top_k = request.top_k.unwrap_or(10).min(100); // Cap at 100

    let recommendations = match state.metadata.model_type.as_str() {
        "hybrid" => {
            let user_id = request.user_id.ok_or_else(|| {
                AppError::BadRequest("user_id required for hybrid model".to_string())
            })?;
            
            state.inference_engine
                .get_user_recommendations(user_id, top_k, &state.movie_data)
                .await?
        }
        "content-based" => {
            if let Some(movie_ids) = request.movie_ids {
                state.inference_engine
                    .get_content_based_recommendations(&movie_ids, top_k, &state.movie_data)
                    .await?
            } else if let Some(genres) = request.genres {
                state.inference_engine
                    .get_genre_based_recommendations(&genres, top_k, &state.movie_data)
                    .await?
            } else {
                return Err(AppError::BadRequest(
                    "movie_ids or genres required for content-based model".to_string()
                ));
            }
        }
        _ => {
            return Err(AppError::InternalError(
                "Unknown model type".to_string()
            ));
        }
    };

    let movie_recommendations: Vec<MovieRecommendation> = recommendations
        .into_iter()
        .enumerate()
        .map(|(rank, (movie_id, score))| {
            let movie = state.movie_data.movies.get(&movie_id);
            MovieRecommendation {
                movie_id,
                title: movie.map(|m| m.title.clone()).unwrap_or_else(|| "Unknown".to_string()),
                genres: movie.map(|m| m.genres.clone()).unwrap_or_else(|| "Unknown".to_string()),
                score,
                rank: rank + 1,
            }
        })
        .collect();

    let response = RecommendationResponse {
        recommendations: movie_recommendations,
        request_id,
        model_type: state.metadata.model_type.clone(),
    };

    Ok(Json(response))
}

async fn list_movies_handler(
    State(state): State<AppState>,
) -> Result<Json<Vec<&data::Movie>>, AppError> {
    let movies: Vec<&data::Movie> = state.movie_data.movies.values().collect();
    Ok(Json(movies))
}

async fn list_genres_handler(
    State(state): State<AppState>,
) -> Result<Json<&Vec<String>>, AppError> {
    Ok(Json(&state.metadata.genres))
}