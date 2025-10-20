#!/usr/bin/env python3
"""
Script to export model metadata as JSON for better Rust compatibility.
Run this after training your model to create JSON versions of pickle files.
"""

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def export_metadata(models_path: Path):
    """Export model metadata from pickle to JSON"""
    logger.info("Exporting model metadata...")
    
    metadata_pickle_path = models_path / "model_metadata.pkl"
    metadata_json_path = models_path / "model_metadata.json"
    
    if not metadata_pickle_path.exists():
        logger.error(f"Metadata pickle file not found: {metadata_pickle_path}")
        return False
    
    try:
        with open(metadata_pickle_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Convert numpy types and ensure all values are JSON serializable
        metadata_clean = convert_numpy_types(metadata)
        
        # Convert any remaining non-serializable types
        for key, value in metadata_clean.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool, list, dict)):
                continue
            else:
                # Convert to string as fallback
                metadata_clean[key] = str(value)
        
        with open(metadata_json_path, 'w') as f:
            json.dump(metadata_clean, f, indent=2)
        
        logger.info(f"Exported metadata to {metadata_json_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting metadata: {e}")
        return False

def export_movie_lookup(models_path: Path):
    """Export movie lookup table from pickle to JSON"""
    logger.info("Exporting movie lookup table...")
    
    lookup_pickle_path = models_path / "movie_lookup.pkl"
    lookup_json_path = models_path / "movie_lookup.json"
    
    if not lookup_pickle_path.exists():
        logger.warning(f"Movie lookup pickle file not found: {lookup_pickle_path}")
        return False
    
    try:
        with open(lookup_pickle_path, 'rb') as f:
            movie_lookup = pickle.load(f)
        
        # Convert the lookup table to a JSON-serializable format
        # Convert integer keys to strings for JSON
        movie_lookup_clean = {}
        for movie_id, movie_data in movie_lookup.items():
            movie_lookup_clean[str(movie_id)] = convert_numpy_types(movie_data)
        
        with open(lookup_json_path, 'w') as f:
            json.dump(movie_lookup_clean, f, indent=2)
        
        logger.info(f"Exported movie lookup to {lookup_json_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting movie lookup: {e}")
        return False

def export_rating_scaler(models_path: Path):
    """Export rating scaler from pickle to JSON"""
    logger.info("Exporting rating scaler...")
    
    scaler_pickle_path = models_path / "rating_scaler.pkl"
    scaler_json_path = models_path / "rating_scaler.json"
    
    if not scaler_pickle_path.exists():
        logger.warning(f"Rating scaler pickle file not found: {scaler_pickle_path}")
        return False
    
    try:
        with open(scaler_pickle_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Extract scaler parameters (sklearn MinMaxScaler)
        scaler_data = {
            "min_": scaler.min_.tolist() if hasattr(scaler, 'min_') else None,
            "scale_": scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
            "data_min_": scaler.data_min_.tolist() if hasattr(scaler, 'data_min_') else None,
            "data_max_": scaler.data_max_.tolist() if hasattr(scaler, 'data_max_') else None,
            "data_range_": scaler.data_range_.tolist() if hasattr(scaler, 'data_range_') else None,
            "feature_range": getattr(scaler, 'feature_range', (0, 1)),
            "clip": getattr(scaler, 'clip', False)
        }
        
        with open(scaler_json_path, 'w') as f:
            json.dump(scaler_data, f, indent=2)
        
        logger.info(f"Exported rating scaler to {scaler_json_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting rating scaler: {e}")
        return False

def export_similarity_matrix(models_path: Path):
    """Export similarity matrix from pickle to JSON (if it exists)"""
    logger.info("Exporting similarity matrix...")
    
    similarity_pickle_path = models_path / "similarity_matrix.pkl"
    similarity_json_path = models_path / "similarity_matrix.json"
    
    if not similarity_pickle_path.exists():
        logger.warning(f"Similarity matrix pickle file not found: {similarity_pickle_path}")
        return False
    
    try:
        with open(similarity_pickle_path, 'rb') as f:
            similarity_matrix = pickle.load(f)
        
        # Convert to JSON-serializable format
        # Note: This could be very large, so we might want to compress or limit it
        similarity_clean = {}
        for movie_id, similarities in similarity_matrix.items():
            # Convert keys to strings and limit to top similarities to reduce size
            movie_similarities = dict(similarities)
            # Sort by similarity and keep only top 100 most similar movies
            top_similarities = dict(sorted(movie_similarities.items(), 
                                         key=lambda x: x[1], reverse=True)[:100])
            
            similarity_clean[str(movie_id)] = {
                str(k): float(v) for k, v in top_similarities.items()
            }
        
        with open(similarity_json_path, 'w') as f:
            json.dump(similarity_clean, f, indent=2)
        
        logger.info(f"Exported similarity matrix to {similarity_json_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting similarity matrix: {e}")
        return False

def export_training_history(models_path: Path):
    """Export training history from pickle to JSON"""
    logger.info("Exporting training history...")
    
    history_pickle_path = models_path / "training_history.pkl"
    history_json_path = models_path / "training_history.json"
    
    if not history_pickle_path.exists():
        logger.warning(f"Training history pickle file not found: {history_pickle_path}")
        return False
    
    try:
        with open(history_pickle_path, 'rb') as f:
            history = pickle.load(f)
        
        # Convert to JSON-serializable format
        history_clean = convert_numpy_types(history)
        
        with open(history_json_path, 'w') as f:
            json.dump(history_clean, f, indent=2)
        
        logger.info(f"Exported training history to {history_json_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting training history: {e}")
        return False

def validate_csv_files(models_path: Path):
    """Validate that CSV files are properly formatted for Rust"""
    logger.info("Validating CSV files...")
    
    movies_csv_path = models_path / "movies_data.csv"
    
    if not movies_csv_path.exists():
        logger.error(f"Movies CSV file not found: {movies_csv_path}")
        return False
    
    try:
        df = pd.read_csv(movies_csv_path)
        
        # Check required columns
        required_columns = ['media_id', 'title', 'genres']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns in movies CSV: {missing_columns}")
            return False
        
        # Check for null values in critical columns
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            logger.warning(f"Found null values in CSV: {null_counts.to_dict()}")
        
        logger.info(f"CSV validation passed. Found {len(df)} movies.")
        return True
        
    except Exception as e:
        logger.error(f"Error validating CSV files: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Export model artifacts to JSON for Rust compatibility")
    parser.add_argument("--models-path", type=Path, default="models", 
                       help="Path to the models directory")
    
    args = parser.parse_args()
    
    models_path = Path(args.models_path)
    
    if not models_path.exists():
        logger.error(f"Models directory not found: {models_path}")
        return 1
    
    logger.info(f"Exporting model artifacts from {models_path}")
    
    # Export all artifacts
    success_count = 0
    total_exports = 6
    
    if export_metadata(models_path):
        success_count += 1
    
    if export_movie_lookup(models_path):
        success_count += 1
    
    if export_rating_scaler(models_path):
        success_count += 1
    
    if export_similarity_matrix(models_path):
        success_count += 1
    
    if export_training_history(models_path):
        success_count += 1
    
    if validate_csv_files(models_path):
        success_count += 1
    
    logger.info(f"Export completed: {success_count}/{total_exports} successful")
    
    if success_count == total_exports:
        logger.info("All exports successful! Your model is ready for Rust deployment.")
        return 0
    else:
        logger.warning("Some exports failed. Check the logs above.")
        return 1

if __name__ == "__main__":
    exit(main())