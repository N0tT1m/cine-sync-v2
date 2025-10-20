use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Movie {
    pub media_id: i64,
    pub title: String,
    pub genres: String,
}

#[derive(Debug)]
pub struct MovieData {
    pub movies: HashMap<i64, Movie>,
    pub genre_index: HashMap<String, Vec<i64>>, // Genre -> list of movie IDs
}

impl MovieData {
    pub fn load<P: AsRef<Path>>(pickle_path: P, csv_path: P) -> Result<Self> {
        info!("Loading movie data...");
        
        // Try to load from CSV first (more reliable in Rust)
        let movies = Self::load_from_csv(&csv_path)?;
        
        // Build genre index
        let genre_index = Self::build_genre_index(&movies);
        
        info!("Loaded {} movies with {} unique genres", 
              movies.len(), 
              genre_index.len());
        
        Ok(Self {
            movies,
            genre_index,
        })
    }
    
    fn load_from_csv<P: AsRef<Path>>(csv_path: P) -> Result<HashMap<i64, Movie>> {
        info!("Loading movies from CSV: {:?}", csv_path.as_ref());
        
        let mut reader = csv::Reader::from_path(csv_path)
            .context("Failed to open movies CSV file")?;
        
        let mut movies = HashMap::new();
        
        for result in reader.deserialize() {
            let record: MovieRecord = result
                .context("Failed to parse movie record")?;
            
            let movie = Movie {
                media_id: record.media_id,
                title: record.title.unwrap_or_else(|| "Unknown".to_string()),
                genres: record.genres.unwrap_or_else(|| "Unknown".to_string()),
            };
            
            movies.insert(record.media_id, movie);
        }
        
        Ok(movies)
    }
    
    fn build_genre_index(movies: &HashMap<i64, Movie>) -> HashMap<String, Vec<i64>> {
        let mut genre_index: HashMap<String, Vec<i64>> = HashMap::new();
        
        for (movie_id, movie) in movies {
            // Split genres by '|' as in the original Python code
            let genres: Vec<&str> = movie.genres.split('|').collect();
            
            for genre in genres {
                let genre = genre.trim().to_string();
                if !genre.is_empty() && genre != "Unknown" {
                    genre_index
                        .entry(genre)
                        .or_insert_with(Vec::new)
                        .push(*movie_id);
                }
            }
        }
        
        info!("Built genre index with {} genres", genre_index.len());
        genre_index
    }
    
    pub fn get_movie(&self, movie_id: i64) -> Option<&Movie> {
        self.movies.get(&movie_id)
    }
    
    pub fn get_movies_by_genre(&self, genre: &str) -> Vec<i64> {
        self.genre_index
            .get(genre)
            .cloned()
            .unwrap_or_default()
    }
    
    pub fn get_all_genres(&self) -> Vec<String> {
        self.genre_index.keys().cloned().collect()
    }
    
    pub fn search_movies(&self, query: &str) -> Vec<&Movie> {
        let query_lower = query.to_lowercase();
        
        self.movies
            .values()
            .filter(|movie| {
                movie.title.to_lowercase().contains(&query_lower) ||
                movie.genres.to_lowercase().contains(&query_lower)
            })
            .collect()
    }
    
    pub fn get_random_movies(&self, count: usize) -> Vec<&Movie> {
        use std::collections::HashSet;
        use std::hash::Hash;
        
        let mut selected = HashSet::new();
        let movie_ids: Vec<_> = self.movies.keys().collect();
        
        // Simple pseudo-random selection (in production, use a proper RNG)
        let mut result = Vec::new();
        let mut index = 0;
        
        while result.len() < count && result.len() < self.movies.len() {
            let movie_id = movie_ids[index % movie_ids.len()];
            if selected.insert(movie_id) {
                if let Some(movie) = self.movies.get(movie_id) {
                    result.push(movie);
                }
            }
            index = (index + 7) % movie_ids.len(); // Use prime number for better distribution
        }
        
        result
    }
}

#[derive(Debug, Deserialize)]
struct MovieRecord {
    media_id: i64,
    title: Option<String>,
    genres: Option<String>,
    // Optional fields that might be in the CSV
    #[serde(default)]
    #[allow(dead_code)]
    actor_id: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct GenreFeatures {
    features: Vec<f32>,
    genre_names: Vec<String>,
}

impl GenreFeatures {
    pub fn new(genres: Vec<String>) -> Self {
        Self {
            features: vec![0.0; genres.len()],
            genre_names: genres,
        }
    }
    
    pub fn from_movie(movie: &Movie, all_genres: &[String]) -> Self {
        let mut features = vec![0.0; all_genres.len()];
        
        // Split movie genres and set corresponding features to 1.0
        let movie_genres: Vec<&str> = movie.genres.split('|').collect();
        
        for (i, genre) in all_genres.iter().enumerate() {
            if movie_genres.contains(&genre.as_str()) {
                features[i] = 1.0;
            }
        }
        
        Self {
            features,
            genre_names: all_genres.to_vec(),
        }
    }
    
    pub fn from_genre_names(genre_names: &[String], all_genres: &[String]) -> Self {
        let mut features = vec![0.0; all_genres.len()];
        
        for (i, genre) in all_genres.iter().enumerate() {
            if genre_names.contains(genre) {
                features[i] = 1.0;
            }
        }
        
        Self {
            features,
            genre_names: all_genres.to_vec(),
        }
    }
    
    pub fn features(&self) -> &[f32] {
        &self.features
    }
    
    pub fn to_vec(self) -> Vec<f32> {
        self.features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_genre_features() {
        let all_genres = vec![
            "Action".to_string(),
            "Comedy".to_string(),
            "Drama".to_string(),
        ];
        
        let movie = Movie {
            media_id: 1,
            title: "Test Movie".to_string(),
            genres: "Action|Drama".to_string(),
        };
        
        let features = GenreFeatures::from_movie(&movie, &all_genres);
        assert_eq!(features.features(), &[1.0, 0.0, 1.0]);
    }
    
    #[test]
    fn test_genre_features_from_names() {
        let all_genres = vec![
            "Action".to_string(),
            "Comedy".to_string(),
            "Drama".to_string(),
        ];
        
        let selected_genres = vec!["Comedy".to_string()];
        let features = GenreFeatures::from_genre_names(&selected_genres, &all_genres);
        assert_eq!(features.features(), &[0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_load_movies_from_csv() -> Result<()> {
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, "media_id,title,genres")?;
        writeln!(temp_file, "1,Test Movie 1,Action|Comedy")?;
        writeln!(temp_file, "2,Test Movie 2,Drama")?;
        
        let movies = MovieData::load_from_csv(temp_file.path())?;
        
        assert_eq!(movies.len(), 2);
        assert!(movies.contains_key(&1));
        assert!(movies.contains_key(&2));
        
        let movie1 = &movies[&1];
        assert_eq!(movie1.title, "Test Movie 1");
        assert_eq!(movie1.genres, "Action|Comedy");
        
        Ok(())
    }
}