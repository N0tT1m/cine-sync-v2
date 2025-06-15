#!/usr/bin/env python3
"""
CineSync Dataset Organization Script

This script organizes downloaded datasets into the correct folder structure.
Use this after manually downloading datasets or the complete Kaggle dataset.

Usage:
    python organize_datasets.py                    # Auto-detect and organize
    python organize_datasets.py --source kaggle    # Organize Kaggle complete dataset
    python organize_datasets.py --source individual # Organize individual downloads
    python organize_datasets.py --dry-run          # Show what would be done without actually doing it
"""

import os
import sys
import shutil
import zipfile
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import json

class DatasetOrganizer:
    def __init__(self, dry_run=False):
        self.project_root = Path(__file__).parent
        self.dry_run = dry_run
        
        # Expected folder structure
        self.target_structure = {
            'ml-32m': {
                'files': ['ratings.csv', 'movies.csv', 'tags.csv', 'links.csv', 'README.txt'],
                'description': 'MovieLens 32M dataset'
            },
            'archive': {
                'files': ['combined_data_1.txt', 'combined_data_2.txt', 'combined_data_3.txt', 
                         'combined_data_4.txt', 'movie_titles.csv', 'probe.txt', 'qualifying.txt'],
                'description': 'Netflix Prize archive dataset'
            },
            'tmdb': {
                'files': ['actor_filmography_data.csv', 'actor_filmography_data.json',
                         'actor_filmography_data_movies.csv', 'actor_filmography_data_tv.csv'],
                'description': 'TMDB actor and movie data'
            },
            'tv': {
                'files': ['netflix_tv_shows.csv', '*.zip'],
                'description': 'TV show datasets and zip files'
            },
            'movies': {
                'files': ['netflix_movies.csv', '*.zip'],
                'description': 'Movie datasets and zip files'
            }
        }
        
        # Common source patterns for auto-detection
        self.source_patterns = {
            'kaggle_complete': {
                'indicators': ['kaggle_complete_dataset', 'movie_datasets', 'tv_datasets'],
                'priority': 1
            },
            'individual_zips': {
                'indicators': ['ml-32m.zip', 'archive_netflix_prize.zip', 'tmdb_actors_movies.zip'],
                'priority': 2
            },
            'extracted_folders': {
                'indicators': ['ml-32m', 'archive', 'tmdb'],
                'priority': 3
            }
        }

    def log_action(self, action: str, source: Path = None, destination: Path = None):
        """Log an action that will be or has been performed."""
        if self.dry_run:
            print(f"[DRY RUN] {action}")
            if source and destination:
                print(f"  From: {source}")
                print(f"  To: {destination}")
        else:
            print(f"✅ {action}")
            if source and destination:
                print(f"  {source} → {destination}")

    def find_files_recursively(self, root_dir: Path, patterns: List[str]) -> List[Path]:
        """Find files matching patterns recursively."""
        found_files = []
        for pattern in patterns:
            if '*' in pattern:
                found_files.extend(root_dir.rglob(pattern))
            else:
                found_files.extend(root_dir.rglob(pattern))
        return found_files

    def detect_source_type(self) -> Optional[str]:
        """Auto-detect the type of dataset organization needed."""
        print("🔍 Auto-detecting dataset organization needed...")
        
        for source_type, config in self.source_patterns.items():
            found_indicators = 0
            for indicator in config['indicators']:
                if '*' in indicator:
                    matches = list(self.project_root.glob(indicator))
                    if matches:
                        found_indicators += 1
                        print(f"  Found: {matches[0].name}")
                else:
                    path = self.project_root / indicator
                    if path.exists():
                        found_indicators += 1
                        print(f"  Found: {indicator}")
            
            if found_indicators > 0:
                print(f"✅ Detected source type: {source_type}")
                return source_type
        
        print("❌ Could not auto-detect source type")
        return None

    def organize_kaggle_complete(self):
        """Organize the complete Kaggle dataset."""
        print("\n📦 Organizing Kaggle complete dataset...")
        
        # Look for kaggle_complete_dataset folder
        kaggle_folder = None
        possible_names = ['kaggle_complete_dataset', 'cinesync-complete-training-dataset']
        
        for name in possible_names:
            path = self.project_root / name
            if path.exists():
                kaggle_folder = path
                break
        
        if not kaggle_folder:
            print("❌ Could not find Kaggle complete dataset folder")
            print("Expected folder names: kaggle_complete_dataset or cinesync-complete-training-dataset")
            return False
        
        print(f"📁 Found Kaggle dataset in: {kaggle_folder.name}")
        
        # Organize each component
        success_count = 0
        
        # Handle ml-32m folder
        ml32m_source = kaggle_folder / 'ml-32m'
        if ml32m_source.exists():
            ml32m_dest = self.project_root / 'ml-32m'
            if not self.dry_run:
                if ml32m_dest.exists():
                    shutil.rmtree(ml32m_dest)
                shutil.copytree(ml32m_source, ml32m_dest)
            self.log_action(f"Copied MovieLens 32M dataset", ml32m_source, ml32m_dest)
            success_count += 1
        
        # Handle archive folder
        archive_source = kaggle_folder / 'archive'
        if archive_source.exists():
            archive_dest = self.project_root / 'archive'
            if not self.dry_run:
                if archive_dest.exists():
                    shutil.rmtree(archive_dest)
                shutil.copytree(archive_source, archive_dest)
            self.log_action(f"Copied Netflix Prize archive", archive_source, archive_dest)
            success_count += 1
        
        # Handle tmdb folder
        tmdb_source = kaggle_folder / 'tmdb'
        if tmdb_source.exists():
            tmdb_dest = self.project_root / 'tmdb'
            if not self.dry_run:
                if tmdb_dest.exists():
                    shutil.rmtree(tmdb_dest)
                shutil.copytree(tmdb_source, tmdb_dest)
            self.log_action(f"Copied TMDB actor data", tmdb_source, tmdb_dest)
            success_count += 1
        
        # Handle compressed datasets
        datasets_folders = ['movie_datasets', 'tv_datasets', 'additional_movie_data']
        for folder_name in datasets_folders:
            folder_source = kaggle_folder / folder_name
            if folder_source.exists():
                # Create appropriate destination folders
                if 'tv' in folder_name:
                    dest_folder = self.project_root / 'tv'
                else:
                    dest_folder = self.project_root / 'movies'
                
                dest_folder.mkdir(exist_ok=True)
                
                # Copy all files from this folder
                for file_path in folder_source.iterdir():
                    if file_path.is_file():
                        dest_file = dest_folder / file_path.name
                        if not self.dry_run:
                            shutil.copy2(file_path, dest_file)
                        self.log_action(f"Copied {file_path.name}", file_path, dest_file)
                
                success_count += 1
        
        print(f"\n✅ Kaggle dataset organization complete: {success_count} components processed")
        return success_count > 0

    def organize_individual_zips(self):
        """Organize individual dataset zip files."""
        print("\n📦 Organizing individual dataset zip files...")
        
        zip_mappings = {
            'ml-32m.zip': 'ml-32m',
            'ml32m_movielens.zip': 'ml-32m',
            'archive_netflix_prize.zip': 'archive',
            'tmdb_actors_movies.zip': 'tmdb',
            'CineSync-Training-Dataset.zip': 'movies'
        }
        
        success_count = 0
        
        for zip_name, dest_folder in zip_mappings.items():
            zip_path = self.project_root / zip_name
            if zip_path.exists():
                dest_path = self.project_root / dest_folder
                
                print(f"📦 Extracting {zip_name}...")
                if not self.dry_run:
                    # Remove existing destination
                    if dest_path.exists():
                        shutil.rmtree(dest_path)
                    
                    # Extract zip
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(self.project_root)
                    
                    # Handle nested folder structures
                    extracted_items = list(self.project_root.glob(f"{dest_folder}*"))
                    if extracted_items:
                        # If extraction created a nested folder, move contents up
                        extracted_path = extracted_items[0]
                        if extracted_path.is_dir() and extracted_path != dest_path:
                            if dest_path.exists():
                                shutil.rmtree(dest_path)
                            extracted_path.rename(dest_path)
                
                self.log_action(f"Extracted {zip_name} to {dest_folder}/", zip_path, dest_path)
                success_count += 1
        
        # Handle TV show zip files
        tv_zips = list(self.project_root.glob("*tv*.zip")) + list(self.project_root.glob("*anime*.zip")) + list(self.project_root.glob("*imdb*.zip"))
        if tv_zips:
            tv_folder = self.project_root / 'tv'
            tv_folder.mkdir(exist_ok=True)
            
            for zip_path in tv_zips:
                dest_file = tv_folder / zip_path.name
                if not self.dry_run:
                    shutil.move(str(zip_path), str(dest_file))
                self.log_action(f"Moved {zip_path.name} to tv/", zip_path, dest_file)
                success_count += 1
        
        print(f"\n✅ Individual zip organization complete: {success_count} files processed")
        return success_count > 0

    def organize_extracted_folders(self):
        """Organize already extracted folders."""
        print("\n📁 Organizing extracted folders...")
        
        success_count = 0
        
        # Check for common extracted folder patterns
        folder_mappings = {
            'ml-32m': 'ml-32m',
            'archive': 'archive',
            'tmdb': 'tmdb',
            'netflix_prize': 'archive',
            'movielens': 'ml-32m'
        }
        
        for source_pattern, dest_folder in folder_mappings.items():
            # Look for folders matching the pattern
            matching_folders = [p for p in self.project_root.iterdir() 
                              if p.is_dir() and source_pattern.lower() in p.name.lower()]
            
            if matching_folders:
                source_folder = matching_folders[0]  # Take the first match
                dest_path = self.project_root / dest_folder
                
                if source_folder != dest_path:  # Only move if they're different
                    if not self.dry_run:
                        if dest_path.exists():
                            shutil.rmtree(dest_path)
                        shutil.move(str(source_folder), str(dest_path))
                    
                    self.log_action(f"Moved {source_folder.name} to {dest_folder}/", source_folder, dest_path)
                    success_count += 1
        
        print(f"\n✅ Extracted folder organization complete: {success_count} folders processed")
        return success_count > 0

    def verify_organization(self):
        """Verify that the organization was successful."""
        print("\n🔍 Verifying dataset organization...")
        
        verification_results = {}
        
        for folder, config in self.target_structure.items():
            folder_path = self.project_root / folder
            if not folder_path.exists():
                verification_results[folder] = {'exists': False, 'files': []}
                continue
            
            found_files = []
            for file_pattern in config['files']:
                if '*' in file_pattern:
                    matches = list(folder_path.glob(file_pattern))
                    found_files.extend([m.name for m in matches])
                else:
                    file_path = folder_path / file_pattern
                    if file_path.exists():
                        found_files.append(file_pattern)
            
            verification_results[folder] = {
                'exists': True,
                'files': found_files,
                'expected': len(config['files']),
                'found': len(found_files)
            }
        
        # Print verification results
        print("\n📊 Organization Verification Results:")
        for folder, result in verification_results.items():
            config = self.target_structure[folder]
            if result['exists']:
                status = "✅" if result['found'] > 0 else "⚠️"
                print(f"{status} {folder}/: {result['found']} files found")
                print(f"    {config['description']}")
                if result['files']:
                    print(f"    Files: {', '.join(result['files'][:3])}{'...' if len(result['files']) > 3 else ''}")
            else:
                print(f"❌ {folder}/: Folder not found")
                print(f"    {config['description']}")
        
        return verification_results

    def run_organization(self, source_type: Optional[str] = None):
        """Run the complete organization process."""
        print("🎬 CineSync Dataset Organization")
        print("=" * 40)
        
        if self.dry_run:
            print("🔍 DRY RUN MODE - No files will be moved")
            print("-" * 40)
        
        # Auto-detect source type if not specified
        if not source_type:
            source_type = self.detect_source_type()
            if not source_type:
                print("\n❌ Could not determine how to organize datasets")
                print("Please specify --source manually or check that datasets are downloaded")
                return False
        
        # Run appropriate organization method
        success = False
        if source_type == 'kaggle_complete' or source_type == 'kaggle':
            success = self.organize_kaggle_complete()
        elif source_type == 'individual_zips' or source_type == 'individual':
            success = self.organize_individual_zips()
        elif source_type == 'extracted_folders' or source_type == 'extracted':
            success = self.organize_extracted_folders()
        else:
            print(f"❌ Unknown source type: {source_type}")
            return False
        
        if not success:
            print("❌ Organization failed or no datasets found to organize")
            return False
        
        # Verify organization
        verification_results = self.verify_organization()
        
        # Create models directory if it doesn't exist
        models_dir = self.project_root / 'models'
        models_dir.mkdir(exist_ok=True)
        self.log_action("Created models directory", destination=models_dir)
        
        print("\n" + "=" * 60)
        print("✅ Dataset organization complete!")
        print("=" * 60)
        print("\n🚀 Next steps:")
        print("1. Verify datasets: python setup_datasets.py --check-only")
        print("2. Start training: python main.py")
        print("3. Or try simplified training: python train_simple.py")
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Organize CineSync training datasets')
    parser.add_argument('--source', choices=['kaggle', 'individual', 'extracted', 'auto'], 
                       default='auto', help='Source type to organize from')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be done without actually doing it')
    
    args = parser.parse_args()
    
    source_type = None if args.source == 'auto' else args.source
    
    organizer = DatasetOrganizer(dry_run=args.dry_run)
    success = organizer.run_organization(source_type=source_type)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()