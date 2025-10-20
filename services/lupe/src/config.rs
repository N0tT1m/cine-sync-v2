use anyhow::{Context, Result};
use std::env;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct Config {
    pub discord_token: String,
    pub command_prefix: String,
    pub api_base_url: String,
    pub api_timeout: Duration,
    pub model_type: String,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        Ok(Config {
            discord_token: env::var("DISCORD_TOKEN")
                .context("DISCORD_TOKEN environment variable is required")?,
            
            command_prefix: env::var("COMMAND_PREFIX")
                .unwrap_or_else(|_| "!".to_string()),
            
            api_base_url: env::var("API_BASE_URL")
                .unwrap_or_else(|_| "http://localhost:3000".to_string()),
            
            api_timeout: Duration::from_secs(
                env::var("API_TIMEOUT_SECONDS")
                    .unwrap_or_else(|_| "30".to_string())
                    .parse()
                    .context("Invalid API_TIMEOUT_SECONDS")?
            ),
            
            model_type: env::var("MODEL_TYPE")
                .unwrap_or_else(|_| "hybrid".to_string()),
        })
    }
}