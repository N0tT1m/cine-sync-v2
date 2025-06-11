@echo off
echo Setting up PostgreSQL for CineSync Application...
echo.

REM Check if PostgreSQL is installed
where psql >nul 2>&1
if %errorlevel% neq 0 (
    echo PostgreSQL is not installed or not in PATH.
    echo Please install PostgreSQL and add it to your PATH.
    echo Download from: https://www.postgresql.org/download/windows/
    pause
    exit /b 1
)

echo PostgreSQL found. Proceeding with setup...
echo.

REM Set database configuration
set DB_NAME=cinesync
set DB_USER=postgres
set DB_HOST=localhost
set DB_PORT=5432

echo Creating database '%DB_NAME%'...
psql -h %DB_HOST% -p %DB_PORT% -U %DB_USER% -c "CREATE DATABASE %DB_NAME%;" 2>nul
if %errorlevel% equ 0 (
    echo Database '%DB_NAME%' created successfully.
) else (
    echo Database '%DB_NAME%' may already exist or creation failed.
)
echo.

echo Creating database schema...
psql -h %DB_HOST% -p %DB_PORT% -U %DB_USER% -d %DB_NAME% -c "
-- Create feedback table
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    username TEXT,
    feedback_type TEXT NOT NULL,
    movie_id INTEGER,
    movie_title TEXT,
    rating INTEGER,
    recommendation_method TEXT,
    original_query TEXT,
    feedback_text TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create user_ratings table  
CREATE TABLE IF NOT EXISTS user_ratings (
    id SERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    movie_id INTEGER NOT NULL,
    rating REAL NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, movie_id)
);

-- Create core movies table
CREATE TABLE IF NOT EXISTS movies (
    media_id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    genres TEXT,
    year INTEGER,
    tmdb_id INTEGER,
    imdb_id TEXT,
    overview TEXT,
    poster_path TEXT,
    backdrop_path TEXT,
    vote_average REAL,
    vote_count INTEGER,
    popularity REAL,
    release_date DATE,
    original_language TEXT,
    original_title TEXT,
    runtime INTEGER,
    budget BIGINT,
    revenue BIGINT,
    status TEXT,
    tagline TEXT
);

-- Create core ratings table
CREATE TABLE IF NOT EXISTS ratings (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    media_id INTEGER NOT NULL,
    rating REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, media_id)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_user_ratings_user_id ON user_ratings(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_user_id ON feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type);
CREATE INDEX IF NOT EXISTS idx_movies_title ON movies(title);
CREATE INDEX IF NOT EXISTS idx_movies_year ON movies(year);
CREATE INDEX IF NOT EXISTS idx_movies_genres ON movies(genres);
CREATE INDEX IF NOT EXISTS idx_ratings_user_id ON ratings(user_id);
CREATE INDEX IF NOT EXISTS idx_ratings_media_id ON ratings(media_id);

-- Grant permissions (optional, adjust as needed)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO %DB_USER%;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO %DB_USER%;
"

if %errorlevel% equ 0 (
    echo Database schema created successfully!
) else (
    echo Error creating database schema.
    pause
    exit /b 1
)

echo.
echo Setting up environment variables...
echo.

REM Create .env file for the application
echo Creating .env file...
(
echo DB_HOST=localhost
echo DB_NAME=%DB_NAME%
echo DB_USER=%DB_USER%
echo DB_PASSWORD=
echo DB_PORT=%DB_PORT%
echo DISCORD_TOKEN=
echo DEBUG=true
) > .env

echo .env file created. Please edit it to add your Discord bot token.
echo.

REM Create environment setup batch file
echo Creating set_env.bat for manual environment variable setup...
(
echo @echo off
echo set DB_HOST=localhost
echo set DB_NAME=%DB_NAME%
echo set DB_USER=%DB_USER%
echo set DB_PASSWORD=
echo set DB_PORT=%DB_PORT%
echo set DISCORD_TOKEN=
echo set DEBUG=true
echo echo Environment variables set for current session.
echo echo Please set DISCORD_TOKEN manually: set DISCORD_TOKEN=your_token_here
) > set_env.bat

echo.
echo ========================================
echo PostgreSQL Setup Complete!
echo ========================================
echo.
echo Database: %DB_NAME%
echo Host: %DB_HOST%
echo Port: %DB_PORT%
echo User: %DB_USER%
echo.
echo Next steps:
echo 1. Set your Discord bot token in .env file or run: set DISCORD_TOKEN=your_token_here
echo 2. If needed, set DB_PASSWORD in .env file
echo 3. Load your movie data into the 'movies' and 'ratings' tables
echo 4. Run your Python application
echo.
echo Files created:
echo - .env (environment configuration)
echo - set_env.bat (manual environment setup)
echo.
pause