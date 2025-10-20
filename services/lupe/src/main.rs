use anyhow::Result;
use serenity::{
    async_trait,
    client::{Client, EventHandler},
    framework::standard::{
        macros::group,
        StandardFramework, Configuration,
    },
    model::{
        channel::Message,
        gateway::Ready,
    },
    prelude::*,
};
use tracing::{info, error};

mod api;
mod commands;
mod config;
mod embeds;
mod cache;

use api::LupeApiClient;
use commands::*;
use config::Config;
use cache::UserCache;

// Command groups
#[group]
#[commands(recommend, recommend_by_genre, recommend_similar, help_lupe, stats, profile)]
struct General;

// Bot event handler
struct LupeBot {
    api_client: LupeApiClient,
    user_cache: UserCache,
}

#[async_trait]
impl EventHandler for LupeBot {
    async fn ready(&self, _ctx: serenity::client::Context, ready: Ready) {
        info!("🎬 Lupe is connected as {}", ready.user.name);
        info!("🤖 Bot is ready to recommend movies!");
    }

    async fn message(&self, ctx: serenity::client::Context, msg: Message) {
        // Ignore bot messages
        if msg.author.bot {
            return;
        }

        // Handle mentions or direct messages
        if msg.mentions_me(&ctx).await.unwrap_or(false) {
            let response = "👋 Hey there! I'm **Lupe**, your movie recommendation bot!\n\n\
                          Use `!recommend` to get personalized movie recommendations, or `!help` to see all commands.";
            
            if let Err(e) = msg.channel_id.say(&ctx.http, response).await {
                error!("Error sending mention response: {}", e);
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Load configuration
    dotenvy::dotenv().ok();
    let config = Config::from_env()?;

    info!("🎬 Starting Lupe Discord Bot");
    info!("📡 API Endpoint: {}", config.api_base_url);

    // Initialize API client
    let api_client = LupeApiClient::new(config.api_base_url.clone(), config.api_timeout)?;

    // Test API connection
    match api_client.health_check().await {
        Ok(health) => {
            info!("✅ Connected to Lupe API successfully");
            info!("🎭 Model type: {}", health.model_type.unwrap_or_else(|| "unknown".to_string()));
            info!("🎬 Movies available: {}", health.movies_count);
        }
        Err(e) => {
            error!("❌ Failed to connect to Lupe API: {}", e);
            return Err(e.into());
        }
    }

    // Initialize user cache
    let user_cache = UserCache::new();

    // Create bot handler
    let handler = LupeBot {
        api_client: api_client.clone(),
        user_cache: user_cache.clone(),
    };

    // Setup framework
    let configuration = Configuration::new()
        .prefix(&config.command_prefix)
        .case_insensitivity(true)
        .ignore_bots(true);
    
    let framework = StandardFramework::new();
    framework.configure(configuration);
    let framework = framework.group(&GENERAL_GROUP);

    // Insert shared data
    let mut client = Client::builder(&config.discord_token, GatewayIntents::GUILD_MESSAGES | GatewayIntents::DIRECT_MESSAGES | GatewayIntents::MESSAGE_CONTENT)
        .event_handler(handler)
        .framework(framework)
        .type_map_insert::<ApiClientKey>(api_client.clone())
        .type_map_insert::<UserCacheKey>(user_cache.clone())
        .type_map_insert::<ConfigKey>(config)
        .await
        .map_err(|e| anyhow::anyhow!("Error creating Discord client: {}", e))?;

    // Start the bot
    info!("🚀 Starting Discord bot...");
    if let Err(e) = client.start().await {
        error!("Client error: {}", e);
    }

    Ok(())
}

// Type map keys for shared data
struct ApiClientKey;
struct UserCacheKey;
struct ConfigKey;

impl TypeMapKey for ApiClientKey {
    type Value = LupeApiClient;
}

impl TypeMapKey for UserCacheKey {
    type Value = UserCache;
}

impl TypeMapKey for ConfigKey {
    type Value = Config;
}