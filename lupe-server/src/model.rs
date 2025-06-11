use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tch::{CModule, Device, Tensor, TchError};
use tracing::{info, warn};

#[derive(Debug)]
pub struct RecommendationModel {
    module: CModule,
    device: Device,
}

impl RecommendationModel {
    pub fn load<P: AsRef<Path>>(model_path: P, device: Device) -> Result<Self> {
        info!("Loading PyTorch model from {:?}", model_path.as_ref());
        
        let module = CModule::load_on_device(model_path, device)
            .context("Failed to load TorchScript model")?;
        
        info!("Model loaded successfully on device: {:?}", device);
        
        Ok(Self { module, device })
    }

    pub fn predict(&self, user_ids: &Tensor, movie_ids: &Tensor, genre_features: &Tensor) -> Result<Tensor> {
        // Ensure all tensors are on the correct device
        let user_ids = user_ids.to_device(self.device);
        let movie_ids = movie_ids.to_device(self.device);
        let genre_features = genre_features.to_device(self.device);

        // Run inference
        let outputs = self.module
            .forward_ts(&[user_ids, movie_ids, genre_features])
            .context("Model inference failed")?;

        Ok(outputs)
    }

    pub fn predict_batch(&self, batch_data: &BatchInferenceInput) -> Result<Tensor> {
        let user_ids = Tensor::from_slice(&batch_data.user_ids).to_device(self.device);
        let movie_ids = Tensor::from_slice(&batch_data.movie_ids).to_device(self.device);
        let genre_features = Tensor::from_slice2(&batch_data.genre_features).to_device(self.device);

        self.predict(&user_ids, &movie_ids, &genre_features)
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

#[derive(Debug)]
pub struct BatchInferenceInput {
    pub user_ids: Vec<i64>,
    pub movie_ids: Vec<i64>,
    pub genre_features: Vec<Vec<f32>>,
}

impl BatchInferenceInput {
    pub fn new() -> Self {
        Self {
            user_ids: Vec::new(),
            movie_ids: Vec::new(),
            genre_features: Vec::new(),
        }
    }

    pub fn add_sample(&mut self, user_id: i64, movie_id: i64, genre_features: Vec<f32>) {
        self.user_ids.push(user_id);
        self.movie_ids.push(movie_id);
        self.genre_features.push(genre_features);
    }

    pub fn len(&self) -> usize {
        self.user_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.user_ids.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub num_users: Option<i64>,
    pub num_movies: Option<i64>,
    pub genres: Vec<String>,
    pub embedding_size: Option<i64>,
    pub model_type: String,
    pub training_duration: Option<String>,
    pub trained_with_gpu: Option<bool>,
    pub epochs_completed: Option<i64>,
    pub early_stopping: Option<bool>,
    pub best_val_loss: Option<f64>,
    pub timestamp: Option<String>,
    pub pytorch_version: Option<String>,
    pub cuda_version: Option<String>,
}

impl ModelMetadata {
    pub fn load_from_pickle<P: AsRef<Path>>(path: P) -> Result<Self> {
        info!("Loading model metadata from {:?}", path.as_ref());
        
        // For now, we'll create a simple fallback method since pickle parsing in Rust is complex
        // In a production system, you might want to save metadata as JSON instead
        
        // Try to read as JSON first (if you modify the Python script to also save as JSON)
        if let Ok(json_content) = std::fs::read_to_string(path.as_ref().with_extension("json")) {
            let metadata: ModelMetadata = serde_json::from_str(&json_content)
                .context("Failed to parse metadata JSON")?;
            return Ok(metadata);
        }

        // Fallback: Create default metadata and warn user
        warn!("Could not load metadata from pickle file. Using default values.");
        warn!("Consider modifying the Python script to also save metadata as JSON for better Rust compatibility.");
        
        Ok(ModelMetadata {
            num_users: Some(1000000),  // Default reasonable values
            num_movies: Some(100000),
            genres: vec![
                "Action".to_string(),
                "Adventure".to_string(),
                "Animation".to_string(),
                "Children".to_string(),
                "Comedy".to_string(),
                "Crime".to_string(),
                "Documentary".to_string(),
                "Drama".to_string(),
                "Fantasy".to_string(),
                "Film-Noir".to_string(),
                "Horror".to_string(),
                "Musical".to_string(),
                "Mystery".to_string(),
                "Romance".to_string(),
                "Sci-Fi".to_string(),
                "Thriller".to_string(),
                "War".to_string(),
                "Western".to_string(),
            ],
            embedding_size: Some(64),
            model_type: "hybrid".to_string(),
            training_duration: None,
            trained_with_gpu: Some(true),
            epochs_completed: None,
            early_stopping: None,
            best_val_loss: None,
            timestamp: None,
            pytorch_version: None,
            cuda_version: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn test_batch_inference_input() {
        let mut batch = BatchInferenceInput::new();
        assert!(batch.is_empty());

        batch.add_sample(1, 100, vec![1.0, 0.0, 1.0]);
        batch.add_sample(2, 101, vec![0.0, 1.0, 0.0]);

        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
        assert_eq!(batch.user_ids, vec![1, 2]);
        assert_eq!(batch.movie_ids, vec![100, 101]);
    }

    #[test]
    fn test_metadata_default() {
        let metadata = ModelMetadata {
            num_users: Some(1000),
            num_movies: Some(500),
            genres: vec!["Action".to_string(), "Comedy".to_string()],
            embedding_size: Some(64),
            model_type: "hybrid".to_string(),
            training_duration: None,
            trained_with_gpu: Some(true),
            epochs_completed: Some(10),
            early_stopping: Some(false),
            best_val_loss: Some(0.15),
            timestamp: Some("2024-01-01 12:00:00".to_string()),
            pytorch_version: Some("2.0.0".to_string()),
            cuda_version: Some("11.8".to_string()),
        };

        assert_eq!(metadata.model_type, "hybrid");
        assert_eq!(metadata.genres.len(), 2);
    }
}