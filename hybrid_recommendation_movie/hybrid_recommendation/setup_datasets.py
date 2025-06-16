#!/usr/bin/env python3
"""
CineSync Dataset Setup Script

This script attempts to automatically download and setup the CineSync training datasets.
If automatic download fails, it provides manual download links and instructions.

Usage:
    python setup_datasets.py
    python setup_datasets.py --check-only  # Just check what's missing
    python setup_datasets.py --force       # Re-download everything
"""

import os
import sys
import zipfile
import requests
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import shutil

class DatasetSetup:
    def __init__(self, force_download=False):
        self.project_root = Path(__file__).parent
        self.force_download = force_download
        
        # Dataset configurations
        self.datasets = {
            'movielens_32m': {
                'url': 'https://files.grouplens.org/datasets/movielens/ml-32m.zip',
                'destination': 'ml-32m',
                'files': ['ratings.csv', 'movies.csv', 'tags.csv', 'links.csv'],
                'size_mb': 265,
                'required': True,
                'description': 'MovieLens 32M dataset - Core collaborative filtering data'
            },
            'netflix_prize': {
                'url': None,  # No direct download - too large
                'destination': 'archive',
                'files': ['combined_data_1.txt', 'combined_data_2.txt', 'combined_data_3.txt', 'combined_data_4.txt', 'movie_titles.csv'],
                'size_mb': 2000,
                'required': False,
                'description': 'Netflix Prize dataset - Historic competition data',
                'manual_only': True
            }
        }
        
        # Manual download sources
        self.manual_sources = {
            'kaggle_complete': {
                'name': 'CineSync Complete Dataset',
                'url': 'https://kaggle.com/datasets/nott1m/cinesync-complete-training-dataset',
                'description': 'Complete dataset with all training data (recommended)',
                'size': '~3GB compressed'
            },
            'tmdb_tv': {
                'name': 'TMDB TV Shows Dataset',
                'url': 'https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata',
                'description': '150K TV shows with metadata',
                'size': '~32MB'
            },
            'myanimelist': {
                'name': 'MyAnimeList Dataset',
                'url': 'https://www.kaggle.com/datasets/azathoth42/myanimelist',
                'description': 'Anime and TV series ratings',
                'size': '~227MB'
            },
            'imdb_tv': {
                'name': 'IMDb TV Series Dataset',
                'url': 'https://www.kaggle.com/datasets/bourdier/imdb-tv-series-dataset',
                'description': 'TV series organized by genre',
                'size': '~28MB'
            }
        }

    def check_dataset_exists(self, dataset_name: str) -> Tuple[bool, List[str]]:
        """Check if a dataset exists and return missing files."""
        config = self.datasets[dataset_name]
        dataset_path = self.project_root / config['destination']
        
        if not dataset_path.exists():
            return False, config['files']
        
        missing_files = []
        for file_name in config['files']:
            if not (dataset_path / file_name).exists():
                missing_files.append(file_name)
        
        return len(missing_files) == 0, missing_files

    def download_with_progress(self, url: str, destination: Path) -> bool:
        """Download a file with progress indicator."""
        try:
            print(f"Downloading from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rProgress: {percent:.1f}% ({downloaded // 1024 // 1024}MB)", end='')
            
            print(f"\n✅ Downloaded successfully: {destination.name}")
            return True
            
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            return False

    def extract_zip(self, zip_path: Path, extract_to: Path) -> bool:
        """Extract a zip file to destination."""
        try:
            print(f"Extracting {zip_path.name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to.parent)
            
            # Remove zip file after extraction
            zip_path.unlink()
            print(f"✅ Extracted successfully")
            return True
            
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return False

    def setup_movielens(self) -> bool:
        """Setup MovieLens 32M dataset."""
        print("\n📊 Setting up MovieLens 32M dataset...")
        
        exists, missing = self.check_dataset_exists('movielens_32m')
        if exists and not self.force_download:
            print("✅ MovieLens 32M already exists and is complete")
            return True
        
        dataset_path = self.project_root / 'ml-32m'
        zip_path = self.project_root / 'ml-32m.zip'
        
        # Download
        if not self.download_with_progress(self.datasets['movielens_32m']['url'], zip_path):
            return False
        
        # Extract
        if not self.extract_zip(zip_path, dataset_path):
            return False
        
        # Verify extraction
        exists, missing = self.check_dataset_exists('movielens_32m')
        if exists:
            print("✅ MovieLens 32M setup complete")
            return True
        else:
            print(f"❌ Setup incomplete, missing files: {missing}")
            return False

    def check_all_datasets(self) -> Dict[str, Tuple[bool, List[str]]]:
        """Check status of all datasets."""
        results = {}
        
        # Check configured datasets
        for name in self.datasets:
            results[name] = self.check_dataset_exists(name)
        
        # Check for additional folders
        additional_checks = {
            'tmdb': (['actor_filmography_data.csv'], 'TMDB actor data'),
            'tv': (['netflix_tv_shows.csv'], 'TV show data'),
            'models': ([], 'Model storage (created during training)')
        }
        
        for folder, (files, desc) in additional_checks.items():
            folder_path = self.project_root / folder
            if folder_path.exists():
                if files:
                    missing = [f for f in files if not (folder_path / f).exists()]
                    results[f'{folder}_data'] = (len(missing) == 0, missing)
                else:
                    results[f'{folder}_folder'] = (True, [])
            else:
                results[f'{folder}_folder'] = (False, [folder])
        
        return results

    def print_status_report(self):
        """Print comprehensive status of all datasets."""
        print("\n" + "="*60)
        print("🎬 CineSync Dataset Status Report")
        print("="*60)
        
        results = self.check_all_datasets()
        
        # Required datasets
        print("\n📊 Required Datasets:")
        for name, config in self.datasets.items():
            if config['required']:
                exists, missing = results.get(name, (False, []))
                status = "✅ Complete" if exists else f"❌ Missing: {missing}"
                print(f"  {config['description']}: {status}")
        
        # Optional datasets
        print("\n📺 Optional Datasets:")
        for name, config in self.datasets.items():
            if not config['required']:
                exists, missing = results.get(name, (False, []))
                status = "✅ Available" if exists else "⚠️  Not installed"
                print(f"  {config['description']}: {status}")
        
        # Additional data
        print("\n🗂️ Additional Data:")
        additional_items = ['tmdb_data', 'tv_data', 'models_folder']
        for item in additional_items:
            if item in results:
                exists, missing = results[item]
                status = "✅ Available" if exists else "⚠️  Not found"
                print(f"  {item.replace('_', ' ').title()}: {status}")

    def print_manual_download_guide(self):
        """Print manual download instructions."""
        print("\n" + "="*60)
        print("📥 Manual Download Instructions")
        print("="*60)
        
        print("\n🚀 Recommended: Complete Dataset")
        kaggle = self.manual_sources['kaggle_complete']
        print(f"📦 {kaggle['name']}")
        print(f"🔗 URL: {kaggle['url']}")
        print(f"📊 Size: {kaggle['size']}")
        print(f"📝 Description: {kaggle['description']}")
        print("\nSteps:")
        print("1. Download and extract the complete dataset")
        print("2. Run: python organize_datasets.py")
        print("3. Verify: python setup_datasets.py --check-only")
        
        print("\n📺 Individual TV Show Datasets:")
        tv_datasets = ['tmdb_tv', 'myanimelist', 'imdb_tv']
        for name in tv_datasets:
            source = self.manual_sources[name]
            print(f"\n📦 {source['name']} ({source['size']})")
            print(f"🔗 {source['url']}")
            print(f"📝 {source['description']}")
        
        print("\n💡 Pro Tips:")
        print("- Create a free Kaggle account to access datasets")
        print("- Use the complete dataset for easiest setup")
        print("- Individual datasets give you more control")
        print("- All datasets are research-grade and well-documented")

    def create_directory_structure(self):
        """Create necessary directories."""
        directories = ['models', 'tv', 'movies']
        for dir_name in directories:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(exist_ok=True)
            print(f"📁 Created directory: {dir_name}")

    def run_setup(self, check_only=False):
        """Run the complete setup process."""
        print("🎬 CineSync Dataset Setup")
        print("=" * 40)
        
        # Always show status first
        self.print_status_report()
        
        if check_only:
            print("\n✅ Check complete. Use 'python setup_datasets.py' to download missing datasets.")
            return
        
        # Create directories
        print("\n📁 Creating directory structure...")
        self.create_directory_structure()
        
        # Attempt automatic downloads
        success_count = 0
        total_auto = sum(1 for config in self.datasets.values() if config.get('url') and not config.get('manual_only'))
        
        if total_auto > 0:
            print(f"\n🚀 Attempting automatic downloads ({total_auto} datasets)...")
            
            # Try MovieLens download
            if self.datasets['movielens_32m']['url']:
                if self.setup_movielens():
                    success_count += 1
        
        # Show results and manual instructions
        print(f"\n📊 Automatic download results: {success_count}/{total_auto} successful")
        
        if success_count < total_auto or any(not self.check_dataset_exists(name)[0] for name in self.datasets if self.datasets[name]['required']):
            print("\n📥 Some datasets require manual download:")
            self.print_manual_download_guide()
        else:
            print("\n✅ All required datasets are available!")
            print("🚀 You can now run training with: python main.py")
        
        # Final status
        print("\n" + "="*60)
        print("📋 Final Setup Status")
        print("="*60)
        self.print_status_report()


def main():
    parser = argparse.ArgumentParser(description='Setup CineSync training datasets')
    parser.add_argument('--check-only', action='store_true', help='Only check dataset status, do not download')
    parser.add_argument('--force', action='store_true', help='Force re-download of existing datasets')
    
    args = parser.parse_args()
    
    setup = DatasetSetup(force_download=args.force)
    setup.run_setup(check_only=args.check_only)


if __name__ == '__main__':
    main()