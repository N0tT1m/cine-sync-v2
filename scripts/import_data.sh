#!/bin/bash
#
# CineSync v2 Data Import Script
# Convenience wrapper for the data import pipeline
#
# Usage:
#   ./scripts/import_data.sh                    # Full import (movies, TV, anime) from 2017
#   ./scripts/import_data.sh --anime-only       # Anime only (no API key needed)
#   ./scripts/import_data.sh --quick            # Quick mode
#   ./scripts/import_data.sh --from-year 2020   # Custom start year
#   ./scripts/import_data.sh --help             # Show all options
#
# Defaults:
#   - Date range: 2017-01-01 to today
#   - Quality filters: min 50 votes, min 2.0 popularity
#   - Anime: min 5.0 score, min 1000 members
#

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load .env file if it exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  CineSync v2 Data Import Pipeline${NC}"
echo -e "${BLUE}  Movies, TV Shows, and Anime${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check for TMDB API key
if [ -z "$TMDB_API_KEY" ]; then
    echo -e "${YELLOW}Warning: TMDB_API_KEY environment variable not set${NC}"
    echo -e "${YELLOW}Movies and TV import requires a TMDB API key${NC}"
    echo -e "${YELLOW}Get a free key at: https://www.themoviedb.org/settings/api${NC}"
    echo ""

    # Check if user wants to set it now
    if [[ ! "$*" == *"--anime-only"* ]] && [[ ! "$*" == *"--help"* ]]; then
        read -p "Enter TMDB API key (or press Enter to skip movies/TV): " api_key
        if [ -n "$api_key" ]; then
            export TMDB_API_KEY="$api_key"
            echo -e "${GREEN}API key set for this session${NC}"
        else
            echo -e "${YELLOW}Continuing with anime-only import...${NC}"
            set -- "$@" "--no-movies" "--no-tv"
        fi
    fi
    echo ""
fi

# Change to project root
cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}Activated virtual environment${NC}"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
    echo -e "${GREEN}Activated virtual environment${NC}"
fi

# Create necessary directories
mkdir -p data/imports
mkdir -p logs

# Run the import pipeline
echo -e "${BLUE}Starting data import...${NC}"
echo ""

python src/data/data_import_pipeline.py "$@"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}  Import completed successfully!${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo ""
    echo "Data saved to:"
    echo "  - Database: data/cinesync.db"
    echo "  - CSVs: data/imports/"
    echo "  - Logs: logs/"
else
    echo ""
    echo -e "${RED}================================================${NC}"
    echo -e "${RED}  Import failed with exit code $exit_code${NC}"
    echo -e "${RED}================================================${NC}"
fi

exit $exit_code
