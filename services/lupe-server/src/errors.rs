use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Internal server error: {0}")]
    InternalError(String),

    #[error("Model error: {0}")]
    ModelError(String),

    #[error("Data error: {0}")]
    DataError(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("PyTorch error: {0}")]
    TorchError(#[from] tch::TchError),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("CSV error: {0}")]
    CsvError(#[from] csv::Error),

    #[error("Generic error: {0}")]
    AnyhowError(#[from] anyhow::Error),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            AppError::InternalError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            AppError::ModelError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, format!("Model error: {}", msg)),
            AppError::DataError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, format!("Data error: {}", msg)),
            AppError::TorchError(err) => (StatusCode::INTERNAL_SERVER_ERROR, format!("PyTorch error: {}", err)),
            AppError::SerializationError(err) => (StatusCode::INTERNAL_SERVER_ERROR, format!("Serialization error: {}", err)),
            AppError::IoError(err) => (StatusCode::INTERNAL_SERVER_ERROR, format!("IO error: {}", err)),
            AppError::CsvError(err) => (StatusCode::INTERNAL_SERVER_ERROR, format!("CSV error: {}", err)),
            AppError::AnyhowError(err) => (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {}", err)),
        };

        let body = Json(json!({
            "error": error_message,
            "status": status.as_u16()
        }));

        (status, body).into_response()
    }
}

// Helper function to convert various error types to AppError
impl From<String> for AppError {
    fn from(msg: String) -> Self {
        AppError::InternalError(msg)
    }
}

impl From<&str> for AppError {
    fn from(msg: &str) -> Self {
        AppError::InternalError(msg.to_string())
    }
}