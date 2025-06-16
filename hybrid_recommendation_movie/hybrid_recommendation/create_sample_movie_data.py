#!/usr/bin/env python3
"""
Create sample movie data for testing the Discord bot
This creates basic movie lookup and mapping files so the bot can function
"""

import pickle
import json
import os

def create_sample_movie_data():
    """Create sample movie data for testing"""
    
    # Sample real movies (popular films people would recognize)
    sample_movies = [
        (1, "The Shawshank Redemption", "Drama"),
        (2, "The Godfather", "Crime|Drama"), 
        (3, "The Dark Knight", "Action|Crime|Drama"),
        (4, "Pulp Fiction", "Crime|Drama"),
        (5, "The Lord of the Rings: The Return of the King", "Adventure|Drama|Fantasy"),
        (6, "Forrest Gump", "Drama|Romance"),
        (7, "Star Wars: Episode IV - A New Hope", "Adventure|Fantasy|Sci-Fi"),
        (8, "The Matrix", "Action|Sci-Fi|Thriller"),
        (9, "Goodfellas", "Crime|Drama"),
        (10, "The Silence of the Lambs", "Crime|Horror|Thriller"),
        (11, "Saving Private Ryan", "Drama|War"),
        (12, "Terminator 2: Judgment Day", "Action|Sci-Fi|Thriller"),
        (13, "Back to the Future", "Adventure|Comedy|Sci-Fi"),
        (14, "Raiders of the Lost Ark", "Action|Adventure"),
        (15, "Jurassic Park", "Adventure|Sci-Fi|Thriller"),
        (16, "Titanic", "Drama|Romance"),
        (17, "Avatar", "Action|Adventure|Sci-Fi"),
        (18, "Avengers: Endgame", "Action|Adventure|Sci-Fi"),
        (19, "Spider-Man: Into the Spider-Verse", "Action|Adventure|Animation"),
        (20, "Toy Story", "Adventure|Animation|Children|Comedy|Fantasy"),
        (21, "Finding Nemo", "Adventure|Animation|Children|Comedy"),
        (22, "The Lion King", "Adventure|Animation|Children|Drama|Musical"),
        (23, "Frozen", "Adventure|Animation|Children|Comedy|Family|Musical"),
        (24, "Inception", "Action|Crime|Drama|Mystery|Sci-Fi|Thriller"),
        (25, "Interstellar", "Adventure|Drama|Sci-Fi"),
        (26, "The Prestige", "Drama|Mystery|Sci-Fi|Thriller"),
        (27, "Fight Club", "Drama|Thriller"),
        (28, "Se7en", "Crime|Drama|Mystery|Thriller"),
        (29, "The Departed", "Crime|Drama|Thriller"),
        (30, "Casino Royale", "Action|Adventure|Thriller"),
        (31, "Mad Max: Fury Road", "Action|Adventure|Sci-Fi|Thriller"),
        (32, "John Wick", "Action|Crime|Thriller"),
        (33, "Die Hard", "Action|Thriller"),
        (34, "Aliens", "Action|Adventure|Horror|Sci-Fi|Thriller"),
        (35, "Blade Runner", "Drama|Sci-Fi|Thriller"),
        (36, "The Social Network", "Drama"),
        (37, "La La Land", "Comedy|Drama|Musical|Romance"),
        (38, "Parasite", "Comedy|Drama|Thriller"),
        (39, "Get Out", "Horror|Mystery|Thriller"),
        (40, "Black Panther", "Action|Adventure|Sci-Fi"),
        (41, "Wonder Woman", "Action|Adventure|Fantasy"),
        (42, "Captain America: The Winter Soldier", "Action|Adventure|Sci-Fi"),
        (43, "Iron Man", "Action|Adventure|Sci-Fi"),
        (44, "Thor: Ragnarok", "Action|Adventure|Comedy|Sci-Fi"),
        (45, "Guardians of the Galaxy", "Action|Adventure|Comedy|Sci-Fi"),
        (46, "The Incredibles", "Action|Adventure|Animation|Children|Comedy"),
        (47, "Up", "Adventure|Animation|Children|Drama"),
        (48, "WALL-E", "Adventure|Animation|Children|Romance|Sci-Fi"),
        (49, "Inside Out", "Adventure|Animation|Children|Comedy|Drama|Family"),
        (50, "Coco", "Adventure|Animation|Children|Comedy|Family|Musical"),
        (51, "Moana", "Adventure|Animation|Children|Comedy|Family|Musical"),
        (52, "Zootopia", "Adventure|Animation|Children|Comedy|Crime"),
        (53, "How to Train Your Dragon", "Adventure|Animation|Children|Comedy|Fantasy"),
        (54, "Shrek", "Adventure|Animation|Children|Comedy|Fantasy|Romance"),
        (55, "The Incredibles 2", "Action|Adventure|Animation|Children|Comedy"),
        (56, "Monsters, Inc.", "Adventure|Animation|Children|Comedy|Fantasy"),
        (57, "Cars", "Adventure|Animation|Children|Comedy|Family"),
        (58, "A Bug's Life", "Adventure|Animation|Children|Comedy|Family"),
        (59, "Ratatouille", "Adventure|Animation|Children|Comedy|Family"),
        (60, "Brave", "Adventure|Animation|Children|Comedy|Fantasy"),
        (61, "The Good, the Bad and the Ugly", "Western"),
        (62, "Casablanca", "Drama|Romance|War"),
        (63, "Citizen Kane", "Drama|Mystery"),
        (64, "Vertigo", "Mystery|Romance|Thriller"),
        (65, "Psycho", "Horror|Mystery|Thriller"),
        (66, "Apocalypse Now", "Drama|War"),
        (67, "2001: A Space Odyssey", "Mystery|Sci-Fi"),
        (68, "Taxi Driver", "Crime|Drama|Thriller"),
        (69, "Raging Bull", "Drama"),
        (70, "The Wizard of Oz", "Adventure|Children|Fantasy|Musical"),
        (71, "Lawrence of Arabia", "Adventure|Biography|Drama|War"),
        (72, "Some Like It Hot", "Comedy|Romance"),
        (73, "Singin' in the Rain", "Comedy|Musical|Romance"),
        (74, "Gone with the Wind", "Drama|Romance|War"),
        (75, "The Third Man", "Film-Noir|Mystery|Thriller"),
        (76, "Sunset Boulevard", "Drama|Film-Noir"),
        (77, "Dr. Strangelove", "Comedy|War"),
        (78, "North by Northwest", "Action|Adventure|Mystery|Romance|Thriller"),
        (79, "On the Waterfront", "Crime|Drama"),
        (80, "The Maltese Falcon", "Crime|Film-Noir|Mystery"),
        (81, "Chinatown", "Crime|Drama|Mystery|Thriller"),
        (82, "The Treasure of the Sierra Madre", "Adventure|Drama|Western"),
        (83, "Annie Hall", "Comedy|Romance"),
        (84, "The Godfather Part II", "Crime|Drama"),
        (85, "Rear Window", "Mystery|Thriller"),
        (86, "It's a Wonderful Life", "Drama|Fantasy"),
        (87, "Mr. Smith Goes to Washington", "Drama"),
        (88, "Sunset Blvd.", "Drama|Film-Noir"),
        (89, "The Bridge on the River Kwai", "Adventure|Drama|War"),
        (90, "12 Angry Men", "Crime|Drama"),
        (91, "Schindler's List", "Biography|Drama|History|War"),
        (92, "One Flew Over the Cuckoo's Nest", "Drama"),
        (93, "Singin' in the Rain", "Comedy|Musical|Romance"),
        (94, "Jaws", "Adventure|Drama|Thriller"),
        (95, "E.T. the Extra-Terrestrial", "Children|Drama|Fantasy|Sci-Fi"),
        (96, "Close Encounters of the Third Kind", "Drama|Sci-Fi"),
        (97, "All the President's Men", "Drama|History|Thriller"),
        (98, "The Deer Hunter", "Drama|War"),
        (99, "Manhattan", "Comedy|Drama|Romance"),
        (100, "Kramer vs. Kramer", "Drama")
    ]
    
    print(f"Creating sample data with {len(sample_movies)} movies...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # 1. Create movie lookup table
    movie_lookup = {}
    for movie_id, title, genres in sample_movies:
        movie_lookup[movie_id] = {
            'title': title,
            'genres': genres
        }
    
    # Save movie lookup
    with open('models/movie_lookup.pkl', 'wb') as f:
        pickle.dump(movie_lookup, f)
    print("✅ Created movie_lookup.pkl")
    
    # 2. Create ID mappings
    movie_id_to_idx = {movie_id: idx for idx, (movie_id, _, _) in enumerate(sample_movies)}
    idx_to_movie_id = {idx: movie_id for movie_id, idx in movie_id_to_idx.items()}
    
    # Create some sample user mappings
    user_id_to_idx = {i: i for i in range(1, 101)}  # 100 sample users
    idx_to_user_id = {i: i for i in range(1, 101)}
    
    mappings = {
        'num_users': 100,
        'num_movies': len(sample_movies),
        'user_id_to_idx': user_id_to_idx,
        'idx_to_user_id': idx_to_user_id,
        'movie_id_to_idx': movie_id_to_idx,
        'idx_to_movie_id': idx_to_movie_id
    }
    
    # Save mappings
    with open('models/id_mappings.pkl', 'wb') as f:
        pickle.dump(mappings, f)
    print("✅ Created id_mappings.pkl")
    
    # 3. Create movies_data.csv content
    import io
    
    # Get all unique genres
    all_genres = set()
    for _, _, genres in sample_movies:
        all_genres.update(genres.split('|'))
    
    genres_list = sorted(list(all_genres))
    print(f"Found {len(genres_list)} genres: {genres_list}")
    
    # Create CSV content
    csv_content = ['media_id,title,genres']
    for movie_id, title, genres in sample_movies:
        # Escape quotes in title
        title_escaped = title.replace('"', '""')
        csv_content.append(f'{movie_id},"{title_escaped}","{genres}"')
    
    # Add genre columns
    header = csv_content[0]
    for genre in genres_list:
        header += f',{genre}'
    csv_content[0] = header
    
    # Add genre values for each movie
    for i, (movie_id, title, genres) in enumerate(sample_movies):
        movie_genres = genres.split('|')
        for genre in genres_list:
            csv_content[i + 1] += f',{1 if genre in movie_genres else 0}'
    
    # Save CSV
    with open('models/movies_data.csv', 'w') as f:
        f.write('\n'.join(csv_content))
    print("✅ Created movies_data.csv")
    
    # 4. Create model metadata
    metadata = {
        'model_type': 'hybrid',
        'num_users': 100,
        'num_movies': len(sample_movies),
        'num_genres': len(genres_list),
        'embedding_size': 128,
        'created_for': 'discord_bot_testing'
    }
    
    with open('models/model_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    print("✅ Created model_metadata.pkl")
    
    # 5. Create dummy model file (just so the file exists)
    print("⚠️  Note: You'll need to train a real model and save it as models/best_model.pt")
    print("   For now, the bot will work in fallback mode without the trained model.")
    
    print(f"\n🎉 Sample movie data created successfully!")
    print(f"   {len(sample_movies)} real movies available for recommendation")
    print(f"   {len(genres_list)} genres available")
    print(f"   Movie IDs range from 1 to {max(movie_id for movie_id, _, _ in sample_movies)}")
    
    # Test the created data
    print("\n🧪 Testing created data...")
    
    # Load and verify
    with open('models/movie_lookup.pkl', 'rb') as f:
        test_lookup = pickle.load(f)
    
    with open('models/id_mappings.pkl', 'rb') as f:
        test_mappings = pickle.load(f)
    
    print(f"✅ Movie lookup loaded: {len(test_lookup)} movies")
    print(f"✅ ID mappings loaded: {test_mappings['num_movies']} movies, {test_mappings['num_users']} users")
    
    # Show sample movies
    print(f"\n📽️  Sample movies available:")
    for i, movie_id in enumerate(list(test_lookup.keys())[:10]):
        movie_info = test_lookup[movie_id]
        print(f"   {i+1:2d}. {movie_info['title']} ({movie_info['genres']})")
    
    print(f"\n✅ All data files created successfully!")
    print(f"✅ Your Discord bot should now recommend REAL movies!")

if __name__ == "__main__":
    create_sample_movie_data()