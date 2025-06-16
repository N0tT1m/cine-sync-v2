"""
Simplified ID mapping utilities for movie recommendation system
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def create_id_mappings(ratings_df, content_df, content_id_col='media_id'):
    """
    Create contiguous ID mappings for embeddings
    
    Args:
        ratings_df: DataFrame with user and content ratings
        content_df: DataFrame with content metadata  
        content_id_col: Column name for content IDs
        
    Returns:
        Tuple of (updated_ratings_df, updated_content_df, mappings_dict)
    """
    logger.info("Creating ID mappings")
    
    # Get unique IDs
    unique_user_ids = sorted(ratings_df['userId'].unique())
    unique_content_ids = sorted(ratings_df['movieId'].unique())
    
    # Create mappings to contiguous indices
    user_mappings = {uid: idx for idx, uid in enumerate(unique_user_ids)}
    content_mappings = {cid: idx for idx, cid in enumerate(unique_content_ids)}
    
    # Create reverse mappings
    reverse_user_mappings = {idx: uid for uid, idx in user_mappings.items()}
    reverse_content_mappings = {idx: cid for cid, idx in content_mappings.items()}
    
    # Apply mappings to dataframes
    ratings_df = ratings_df.copy()
    ratings_df['user_idx'] = ratings_df['userId'].map(user_mappings)
    ratings_df['content_idx'] = ratings_df['movieId'].map(content_mappings)
    
    # Filter content to only include items with ratings
    content_df = content_df[content_df[content_id_col].isin(unique_content_ids)].copy()
    content_df['content_idx'] = content_df[content_id_col].map(content_mappings)
    
    # Validate mappings
    if ratings_df[['user_idx', 'content_idx']].isna().any().any():
        raise ValueError("Failed to map some IDs")
    
    mappings = {
        'user_id_to_idx': user_mappings,
        'content_id_to_idx': content_mappings,
        'idx_to_user_id': reverse_user_mappings,
        'idx_to_content_id': reverse_content_mappings,
        'num_users': len(unique_user_ids),
        'num_content': len(unique_content_ids)
    }
    
    logger.info(f"Created mappings: {len(unique_user_ids)} users, {len(unique_content_ids)} items")
    return ratings_df, content_df, mappings