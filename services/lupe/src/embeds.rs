use serenity::builder::CreateEmbed;
use serenity::model::user::User;
use serenity::model::colour::Color;

use crate::api::{HealthResponse, MovieRecommendation};
use crate::cache::UserPreferences;

pub fn create_recommendation_embed(
    recommendations: &[MovieRecommendation],
    user: &User,
    title: &str,
) -> CreateEmbed {
    // Add description with recommendations
    let mut description = String::new();
    
    for (i, rec) in recommendations.iter().take(10).enumerate() {
        let emoji = match i {
            0 => "🥇",
            1 => "🥈", 
            2 => "🥉",
            _ => "🎬"
        };
        
        description.push_str(&format!(
            "{} **{}**\n📊 Score: {} | 🎭 {}\n\n",
            emoji,
            rec.title,
            rec.format_score(),
            rec.format_genres()
        ));
    }
    
    if description.len() > 4096 {
        description.truncate(4090);
        description.push_str("...");
    }
    
    let mut embed = CreateEmbed::default()
        .title(title)
        .color(Color::from_rgb(255, 193, 7)) // Golden color
        .timestamp(time::OffsetDateTime::now_utc())
        .footer(serenity::builder::CreateEmbedFooter::new("🎬 Powered by Lupe • Made with ❤️"))
        .description(&description)
        .author(serenity::builder::CreateEmbedAuthor::new(format!("Recommendations for {}", user.name)).icon_url(user.face()));
    
    // Add stats field
    if !recommendations.is_empty() {
        let avg_score = recommendations.iter().map(|r| r.score).sum::<f32>() / recommendations.len() as f32;
        embed = embed.field(
            "📈 Stats",
            format!("Average Score: {:.1}% | Total: {}", avg_score * 100.0, recommendations.len()),
            true
        );
    }
    
    embed
}

pub fn create_help_embed() -> CreateEmbed {
    CreateEmbed::default()
        .title("🎬 Lupe - Movie Recommendation Bot")
        .description("I'm **Lupe**, your personal movie recommendation AI! I can suggest movies based on your preferences, genres, or similar movies.")
        .color(Color::from_rgb(138, 43, 226)) // Purple color
        .thumbnail("https://cdn.discordapp.com/emojis/🎭.png")
        .footer(serenity::builder::CreateEmbedFooter::new("🎬 Powered by AI • Made with ❤️"))
        .field(
            "🎯 Personal Recommendations",
            "`!recommend [number]` - Get personalized movie recommendations\n\
             Example: `!recommend 10`",
            false
        )
        .field(
            "🎭 Genre-Based",
            "`!recommend_by_genre <genres...> [number]` - Get movies by genre\n\
             Example: `!recommend_by_genre Action Comedy 8`",
            false
        )
        .field(
            "🔍 Similar Movies",
            "`!recommend_similar <movie titles...>` - Find similar movies\n\
             Example: `!recommend_similar \"The Matrix\" \"Inception\"`",
            false
        )
        .field(
            "👤 User Profile",
            "`!profile` - View your preferences\n\
             `!profile set genres Comedy,Action` - Set preferred genres",
            false
        )
        .field(
            "📊 Bot Info",
            "`!stats` - Show bot statistics and status\n\
             `!help` - Show this help message",
            false
        )
        .field(
            "🎬 Available Genres",
            "Action, Adventure, Animation, Comedy, Crime, Drama, Fantasy, Horror, Mystery, Romance, Sci-Fi, Thriller, War, Western, and more!",
            false
        )
}

pub fn create_stats_embed(health: &HealthResponse) -> CreateEmbed {
    let mut embed = CreateEmbed::default()
        .title("📊 Lupe Statistics")
        .color(Color::from_rgb(0, 255, 127)) // Spring green
        .timestamp(time::OffsetDateTime::now_utc())
        .footer(serenity::builder::CreateEmbedFooter::new("🎬 Live Statistics"))
        .field("🤖 Status", &health.status, true)
        .field("🧠 Model Type", health.model_type.as_ref().unwrap_or(&"Unknown".to_string()), true)
        .field("🎬 Movies Available", &health.movies_count.to_string(), true)
        .field("🎭 Genres Available", &health.genres_count.to_string(), true)
        .field("⚡ Device", &health.device, true)
        .field("🕐 Uptime", &format!("{}s", health.uptime_seconds), true);
    
    if health.model_loaded {
        embed = embed.field("✅ Model Status", "Loaded and Ready", false);
    } else {
        embed = embed.field("❌ Model Status", "Not Loaded", false);
    }
    
    embed
}

pub fn create_profile_embed(user: &User, preferences: Option<&UserPreferences>) -> CreateEmbed {
    let mut embed = CreateEmbed::default()
        .title("👤 User Profile")
        .color(Color::from_rgb(255, 105, 180)) // Hot pink
        .author(serenity::builder::CreateEmbedAuthor::new(&user.name).icon_url(user.face()))
        .timestamp(time::OffsetDateTime::now_utc());
    
    if let Some(prefs) = preferences {
        if let Some(genres) = &prefs.preferred_genres {
            embed = embed.field(
                "🎭 Preferred Genres",
                genres.join(", "),
                false
            );
        } else {
            embed = embed.field(
                "🎭 Preferred Genres", 
                "Not set - use `!profile set genres Action,Comedy` to set preferences",
                false
            );
        }
        
        if let Some(last_request) = &prefs.last_request_time {
            embed = embed.field(
                "🕐 Last Request",
                last_request.format("%Y-%m-%d %H:%M:%S UTC").to_string(),
                true
            );
        }
        
        embed = embed.field(
            "📊 Total Requests",
            prefs.request_count.to_string(),
            true
        );
    } else {
        embed = embed.description("No preferences set yet. Use `!profile set genres Comedy,Action` to get started!");
    }
    
    embed.field(
        "⚙️ Available Commands",
        "`!profile set genres <genre1,genre2,...>` - Set preferred genres\n\
         `!recommend` - Get personalized recommendations",
        false
    )
}

pub fn create_error_embed(error_message: &str, title: Option<&str>) -> CreateEmbed {
    CreateEmbed::default()
        .title(title.unwrap_or("❌ Error"))
        .description(error_message)
        .color(Color::from_rgb(255, 69, 0)) // Red-orange
        .timestamp(time::OffsetDateTime::now_utc())
        .footer(serenity::builder::CreateEmbedFooter::new("Try !help for available commands"))
}