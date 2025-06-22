use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use rayon::prelude::*;
use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::Mutex;
use rand::seq::SliceRandom;
use rand::thread_rng;
use csv::ReaderBuilder;
use std::fs::File;
use std::io::BufReader;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MovieRating {
    pub user_id: u32,
    pub movie_id: u32, 
    pub rating: f32,
    pub timestamp: Option<u64>,
    pub genres: Vec<String>,
    pub title: String,
    pub year: Option<u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TVShowRating {
    pub user_id: u32,
    pub show_id: u32,
    pub rating: f32,
    pub timestamp: Option<u64>,
    pub genres: Vec<String>,
    pub title: String,
    pub year: Option<u16>,
    pub seasons: Option<u8>,
}

#[pyclass]
pub struct CineSyncDataLoader {
    movie_data: Arc<RwLock<Vec<MovieRating>>>,
    tv_data: Arc<RwLock<Vec<TVShowRating>>>,
    batch_size: usize,
    shuffle: bool,
    buffer_size: usize,
    current_epoch: Arc<Mutex<usize>>,
}

#[pymethods]
impl CineSyncDataLoader {
    #[new]
    pub fn new(batch_size: usize, shuffle: bool, buffer_size: usize) -> Self {
        Self {
            movie_data: Arc::new(RwLock::new(Vec::new())),
            tv_data: Arc::new(RwLock::new(Vec::new())),
            batch_size,
            shuffle,
            buffer_size,
            current_epoch: Arc::new(Mutex::new(0)),
        }
    }

    pub fn load_movies_csv(&mut self, file_path: &str) -> PyResult<usize> {
        let file = File::open(file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to open file: {}", e)))?;
        
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(BufReader::new(file));

        let mut movies = Vec::new();
        
        for result in reader.records() {
            let record = result
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("CSV parse error: {}", e)))?;
            
            // Flexible CSV parsing for different movie dataset formats
            let movie = if record.len() >= 4 {
                MovieRating {
                    user_id: record.get(0).unwrap_or("0").parse().unwrap_or(0),
                    movie_id: record.get(1).unwrap_or("0").parse().unwrap_or(0),
                    rating: record.get(2).unwrap_or("0.0").parse().unwrap_or(0.0),
                    timestamp: record.get(3).and_then(|s| s.parse().ok()),
                    title: record.get(4).unwrap_or("Unknown").to_string(),
                    genres: record.get(5)
                        .unwrap_or("")
                        .split('|')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect(),
                    year: record.get(6).and_then(|s| s.parse().ok()),
                }
            } else {
                continue; // Skip malformed records
            };
            
            movies.push(movie);
        }

        let count = movies.len();
        *self.movie_data.write().unwrap() = movies;
        
        Ok(count)
    }

    pub fn load_tv_shows_csv(&mut self, file_path: &str) -> PyResult<usize> {
        let file = File::open(file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to open file: {}", e)))?;
        
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(BufReader::new(file));

        let mut shows = Vec::new();
        
        for result in reader.records() {
            let record = result
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("CSV parse error: {}", e)))?;
            
            let show = if record.len() >= 4 {
                TVShowRating {
                    user_id: record.get(0).unwrap_or("0").parse().unwrap_or(0),
                    show_id: record.get(1).unwrap_or("0").parse().unwrap_or(0),
                    rating: record.get(2).unwrap_or("0.0").parse().unwrap_or(0.0),
                    timestamp: record.get(3).and_then(|s| s.parse().ok()),
                    title: record.get(4).unwrap_or("Unknown").to_string(),
                    genres: record.get(5)
                        .unwrap_or("")
                        .split('|')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect(),
                    year: record.get(6).and_then(|s| s.parse().ok()),
                    seasons: record.get(7).and_then(|s| s.parse().ok()),
                }
            } else {
                continue;
            };
            
            shows.push(show);
        }

        let count = shows.len();
        *self.tv_data.write().unwrap() = shows;
        
        Ok(count)
    }

    pub fn get_batch_count(&self) -> usize {
        let movie_count = self.movie_data.read().unwrap().len();
        let tv_count = self.tv_data.read().unwrap().len();
        let total = movie_count + tv_count;
        (total + self.batch_size - 1) / self.batch_size
    }

    pub fn create_batches(&self, py: Python) -> PyResult<PyObject> {
        let movie_data = self.movie_data.read().unwrap();
        let tv_data = self.tv_data.read().unwrap();
        
        // Combine and prepare all data
        let mut all_samples = Vec::new();
        
        // Add movie ratings
        for movie in movie_data.iter() {
            let sample = vec![
                movie.user_id as f32,
                movie.movie_id as f32, 
                movie.rating,
                0.0, // content_type: 0 = movie
                movie.year.unwrap_or(0) as f32,
                movie.genres.len() as f32,
            ];
            all_samples.push(sample);
        }
        
        // Add TV show ratings  
        for show in tv_data.iter() {
            let sample = vec![
                show.user_id as f32,
                show.show_id as f32,
                show.rating,
                1.0, // content_type: 1 = tv show
                show.year.unwrap_or(0) as f32,
                show.genres.len() as f32,
            ];
            all_samples.push(sample);
        }

        // Shuffle if requested
        if self.shuffle {
            let mut rng = thread_rng();
            all_samples.shuffle(&mut rng);
        }

        // Create batches
        let batches = all_samples
            .chunks(self.batch_size)
            .map(|chunk| {
                let batch_list = PyList::empty(py);
                for sample in chunk {
                    let sample_list = PyList::new(py, sample);
                    batch_list.append(sample_list)?;
                }
                Ok(batch_list.into())
            })
            .collect::<PyResult<Vec<PyObject>>>()?;

        Ok(PyList::new(py, batches).into())
    }

    pub fn get_performance_stats(&self, py: Python) -> PyResult<PyObject> {
        let movie_count = self.movie_data.read().unwrap().len();
        let tv_count = self.tv_data.read().unwrap().len();
        let total_samples = movie_count + tv_count;
        let epoch = *self.current_epoch.lock();
        
        let stats = PyDict::new(py);
        stats.set_item("total_samples", total_samples)?;
        stats.set_item("movie_samples", movie_count)?;
        stats.set_item("tv_samples", tv_count)?;
        stats.set_item("batch_size", self.batch_size)?;
        stats.set_item("current_epoch", epoch)?;
        stats.set_item("total_batches", self.get_batch_count())?;
        
        Ok(stats.into())
    }

    pub fn reset_epoch(&self) {
        let mut epoch = self.current_epoch.lock();
        *epoch += 1;
    }

    pub fn get_data_summary(&self, py: Python) -> PyResult<PyObject> {
        let movie_data = self.movie_data.read().unwrap();
        let tv_data = self.tv_data.read().unwrap();
        
        // Movie statistics
        let movie_users: std::collections::HashSet<u32> = movie_data.iter().map(|m| m.user_id).collect();
        let movie_items: std::collections::HashSet<u32> = movie_data.iter().map(|m| m.movie_id).collect();
        let movie_avg_rating = movie_data.iter().map(|m| m.rating).sum::<f32>() / movie_data.len().max(1) as f32;
        
        // TV statistics
        let tv_users: std::collections::HashSet<u32> = tv_data.iter().map(|t| t.user_id).collect();
        let tv_items: std::collections::HashSet<u32> = tv_data.iter().map(|t| t.show_id).collect();
        let tv_avg_rating = tv_data.iter().map(|t| t.rating).sum::<f32>() / tv_data.len().max(1) as f32;
        
        let summary = PyDict::new(py);
        summary.set_item("movies", PyDict::new(py))?;
        summary.set_item("tv_shows", PyDict::new(py))?;
        
        let movie_dict = summary.get_item("movies").unwrap().downcast::<PyDict>()?;
        movie_dict.set_item("total_ratings", movie_data.len())?;
        movie_dict.set_item("unique_users", movie_users.len())?;
        movie_dict.set_item("unique_items", movie_items.len())?;
        movie_dict.set_item("avg_rating", movie_avg_rating)?;
        
        let tv_dict = summary.get_item("tv_shows").unwrap().downcast::<PyDict>()?;
        tv_dict.set_item("total_ratings", tv_data.len())?;
        tv_dict.set_item("unique_users", tv_users.len())?;
        tv_dict.set_item("unique_items", tv_items.len())?;
        tv_dict.set_item("avg_rating", tv_avg_rating)?;
        
        Ok(summary.into())
    }
}

#[pymodule]
fn cine_sync_dataloader(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CineSyncDataLoader>()?;
    Ok(())
}