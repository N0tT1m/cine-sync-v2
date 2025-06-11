use anyhow::Result;
use serenity::{
    framework::standard::{macros::command, Args, CommandResult},
    model::prelude::*,
    prelude::*,
    builder::CreateMessage,
};
use tracing::{error, info};

use crate::{
    api::{LupeApiClient, MovieRecommendation},
    cache::{UserCache, UserPreferences},
    embeds,
    ApiClientKey, ConfigKey, UserCacheKey,
};

#[command]
#[description = "Get personalized movie recommendations from Lupe"]
#[usage = "!recommend [number] - Get movie recommendations (default: 5)"]
#[example = "!recommend 10"]
pub async fn recommend(ctx: &Context, msg: &Message, mut args: Args) -> CommandResult {
    let typing = msg.channel_id.start_typing(&ctx.http);

    // Parse number of recommendations
    let count: usize = args.single::<usize>().unwrap_or(5).min(15).max(1);

    // Get shared data
    let data = ctx.data.read().await;
    let api_client = data.get::<ApiClientKey>().unwrap();
    let user_cache = data.get::<UserCacheKey>().unwrap();
    let config = data.get::<ConfigKey>().unwrap();

    // Try to get user recommendations
    let user_id = msg.author.id.get();
    
    match get_user_recommendations(api_client, user_cache, user_id, count, &config.model_type).await {
        Ok(recommendations) => {
            let embed = embeds::create_recommendation_embed(&recommendations, &msg.author, "🎬 Your Movie Recommendations");
            
            if let Err(e) = msg.channel_id.send_message(&ctx.http, CreateMessage::new().embed(embed)).await {
                error!("Failed to send recommendation embed: {}", e);
            }
        }
        Err(e) => {
            error!("Failed to get recommendations: {}", e);
            let error_msg = format!("🚫 Sorry, I couldn't get recommendations right now. Try `!recommend_by_genre` instead!\n\n*Error: {}*", e);
            msg.channel_id.say(&ctx.http, error_msg).await?;
        }
    }

    typing.stop();
    Ok(())
}

#[command]
#[aliases("genre", "genres")]
#[description = "Get movie recommendations by genre"]
#[usage = "!recommend_by_genre <genres...> [number] - Get recommendations by genre"]
#[example = "!recommend_by_genre Action Comedy 8"]
pub async fn recommend_by_genre(ctx: &Context, msg: &Message, args: Args) -> CommandResult {
    let typing = msg.channel_id.start_typing(&ctx.http);

    if args.is_empty() {
        let help_msg = "🎭 **Genre Recommendations**\n\n\
                       Usage: `!recommend_by_genre <genres...> [number]`\n\n\
                       **Available genres:** Action, Adventure, Animation, Comedy, Crime, Drama, Fantasy, Horror, Romance, Sci-Fi, Thriller, and more!\n\n\
                       **Examples:**\n\
                       • `!recommend_by_genre Action` - Action movies\n\
                       • `!recommend_by_genre Horror Thriller 10` - 10 Horror/Thriller movies\n\
                       • `!recommend_by_genre Comedy Romance` - Romantic comedies";
        
        msg.channel_id.say(&ctx.http, help_msg).await?;
        typing.stop();
        return Ok(());
    }

    let args_vec: Vec<String> = args.raw_quoted().map(String::from).collect();
    
    // Parse genres and optional count
    let (genres, count) = parse_genres_and_count(args_vec);
    
    if genres.is_empty() {
        msg.channel_id.say(&ctx.http, "❌ Please specify at least one genre!").await?;
        typing.stop();
        return Ok(());
    }

    // Get shared data
    let data = ctx.data.read().await;
    let api_client = data.get::<ApiClientKey>().unwrap();

    match api_client.get_genre_recommendations(genres.clone(), Some(count)).await {
        Ok(response) => {
            if response.recommendations.is_empty() {
                let msg_text = format!("🔍 No movies found for genres: {}", genres.join(", "));
                msg.channel_id.say(&ctx.http, msg_text).await?;
            } else {
                let title = format!("🎭 {} Movies", genres.join(" + "));
                let embed = embeds::create_recommendation_embed(&response.recommendations, &msg.author, &title);
                
                msg.channel_id.send_message(&ctx.http, CreateMessage::new().embed(embed)).await?;
            }
        }
        Err(e) => {
            error!("Failed to get genre recommendations: {}", e);
            msg.channel_id.say(&ctx.http, "🚫 Sorry, I couldn't get genre recommendations right now.").await?;
        }
    }

    typing.stop();
    Ok(())
}

#[command]
#[aliases("similar")]
#[description = "Get movies similar to ones you specify"]
#[usage = "!recommend_similar <movie titles...> - Find similar movies"]
#[example = "!recommend_similar The Matrix Inception"]
pub async fn recommend_similar(ctx: &Context, msg: &Message, args: Args) -> CommandResult {
    let typing = msg.channel_id.start_typing(&ctx.http);

    if args.is_empty() {
        let help_msg = "🔍 **Similar Movie Recommendations**\n\n\
                       Usage: `!recommend_similar <movie titles...>`\n\n\
                       **Examples:**\n\
                       • `!recommend_similar The Matrix` - Movies like The Matrix\n\
                       • `!recommend_similar \"Star Wars\" \"Lord of the Rings\"` - Movies like both\n\n\
                       💡 Use quotes for multi-word titles!";
        
        msg.channel_id.say(&ctx.http, help_msg).await?;
        typing.stop();
        return Ok(());
    }

    let search_terms: Vec<String> = args.raw_quoted().map(String::from).collect();
    
    // Get shared data
    let data = ctx.data.read().await;
    let api_client = data.get::<ApiClientKey>().unwrap();

    // Search for movies
    let mut found_movies = Vec::new();
    for term in &search_terms {
        match api_client.search_movies(term).await {
            Ok(movies) => {
                if let Some(movie) = movies.first() {
                    found_movies.push(movie.media_id);
                    info!("Found movie: {} (ID: {})", movie.title, movie.media_id);
                }
            }
            Err(e) => {
                error!("Failed to search for movie '{}': {}", term, e);
            }
        }
    }

    if found_movies.is_empty() {
        let msg_text = format!("🔍 Couldn't find movies matching: {}", search_terms.join(", "));
        msg.channel_id.say(&ctx.http, msg_text).await?;
        typing.stop();
        return Ok(());
    }

    // Get similar recommendations
    match api_client.get_similar_movie_recommendations(found_movies, Some(10)).await {
        Ok(response) => {
            if response.recommendations.is_empty() {
                msg.channel_id.say(&ctx.http, "🔍 No similar movies found.").await?;
            } else {
                let title = format!("🎬 Movies Similar to {}", search_terms.join(" & "));
                let embed = embeds::create_recommendation_embed(&response.recommendations, &msg.author, &title);
                
                msg.channel_id.send_message(&ctx.http, CreateMessage::new().embed(embed)).await?;
            }
        }
        Err(e) => {
            error!("Failed to get similar recommendations: {}", e);
            msg.channel_id.say(&ctx.http, "🚫 Sorry, I couldn't get similar movie recommendations right now.").await?;
        }
    }

    typing.stop();
    Ok(())
}

#[command]
#[aliases("help")]
#[description = "Show help information for Lupe"]
pub async fn help_lupe(ctx: &Context, msg: &Message) -> CommandResult {
    let embed = embeds::create_help_embed();
    
    msg.channel_id.send_message(&ctx.http, CreateMessage::new().embed(embed)).await?;

    Ok(())
}

#[command]
#[description = "Show Lupe's statistics and status"]
pub async fn stats(ctx: &Context, msg: &Message) -> CommandResult {
    let typing = msg.channel_id.start_typing(&ctx.http);

    // Get shared data
    let data = ctx.data.read().await;
    let api_client = data.get::<ApiClientKey>().unwrap();

    match api_client.health_check().await {
        Ok(health) => {
            let embed = embeds::create_stats_embed(&health);
            
            msg.channel_id.send_message(&ctx.http, CreateMessage::new().embed(embed)).await?;
        }
        Err(e) => {
            error!("Failed to get health stats: {}", e);
            msg.channel_id.say(&ctx.http, "🚫 Sorry, I couldn't get my stats right now.").await?;
        }
    }

    typing.stop();
    Ok(())
}

#[command]
#[description = "View or update your movie preferences"]
#[usage = "!profile [set genres <genre1,genre2,...>]"]
pub async fn profile(ctx: &Context, msg: &Message, mut args: Args) -> CommandResult {
    let data = ctx.data.read().await;
    let user_cache = data.get::<UserCacheKey>().unwrap();
    
    let user_id = msg.author.id.get();

    // Check if setting preferences
    if let Ok(command) = args.single::<String>() {
        if command.to_lowercase() == "set" {
            if let Ok(pref_type) = args.single::<String>() {
                if pref_type.to_lowercase() == "genres" {
                    let genres_str = args.rest();
                    let genres: Vec<String> = genres_str
                        .split(',')
                        .map(|g| g.trim().to_string())
                        .filter(|g| !g.is_empty())
                        .collect();

                    if !genres.is_empty() {
                        let prefs = UserPreferences {
                            preferred_genres: Some(genres.clone()),
                            ..Default::default()
                        };
                        
                        user_cache.set_preferences(user_id, prefs);
                        
                        let msg_text = format!("✅ Saved your preferred genres: {}", genres.join(", "));
                        msg.channel_id.say(&ctx.http, msg_text).await?;
                        return Ok(());
                    }
                }
            }
        }
    }

    // Show current profile
    let embed = embeds::create_profile_embed(&msg.author, user_cache.get_preferences(user_id).as_ref());
    
    msg.channel_id.send_message(&ctx.http, CreateMessage::new().embed(embed)).await?;

    Ok(())
}

// Helper functions
async fn get_user_recommendations(
    api_client: &LupeApiClient,
    user_cache: &UserCache,
    user_id: u64,
    count: usize,
    model_type: &str,
) -> Result<Vec<MovieRecommendation>> {
    // For hybrid models, try user-based recommendations
    if model_type == "hybrid" {
        match api_client.get_user_recommendations(user_id, Some(count)).await {
            Ok(response) => return Ok(response.recommendations),
            Err(_) => {
                // Fallback to genre-based recommendations
            }
        }
    }

    // Fallback: use genre preferences or random genres
    let user_prefs = user_cache.get_preferences(user_id);
    let genres = if let Some(preferred_genres) = user_prefs.and_then(|p| p.preferred_genres) {
        preferred_genres
    } else {
        // Use popular genres as fallback
        vec!["Action".to_string(), "Comedy".to_string(), "Drama".to_string()]
    };

    let response = api_client.get_genre_recommendations(genres, Some(count)).await?;
    Ok(response.recommendations)
}

fn parse_genres_and_count(args: Vec<String>) -> (Vec<String>, usize) {
    let mut genres = Vec::new();
    let mut count = 8; // default count

    for arg in args {
        if let Ok(num) = arg.parse::<usize>() {
            count = num.min(15).max(1);
        } else {
            // Capitalize first letter for genre matching
            let genre = arg.chars().next().unwrap().to_uppercase().collect::<String>() 
                + &arg[1..].to_lowercase();
            genres.push(genre);
        }
    }

    (genres, count)
}