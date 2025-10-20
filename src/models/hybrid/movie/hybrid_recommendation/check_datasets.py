#!/usr/bin/env python3
"""
CineSync Dataset Verification Script

This script checks the integrity and completeness of CineSync training datasets.
Provides detailed information about what's available and what's missing.

Usage:
    python check_datasets.py              # Full check with recommendations
    python check_datasets.py --quick      # Quick check, just show status
    python check_datasets.py --detailed   # Detailed file-by-file analysis
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv
import json
from datetime import datetime

class DatasetChecker:
    def __init__(self):
        self.project_root = Path(__file__).parent
        
        # Dataset specifications
        self.datasets = {
            'ml-32m': {
                'name': 'MovieLens 32M',
                'required': True,
                'priority': 'High',
                'files': {
                    'ratings.csv': {'min_size_mb': 800, 'required': True, 'description': '32M+ user ratings'},
                    'movies.csv': {'min_size_mb': 3, 'required': True, 'description': 'Movie metadata'},
                    'tags.csv': {'min_size_mb': 60, 'required': True, 'description': 'User-generated tags'},
                    'links.csv': {'min_size_mb': 1, 'required': True, 'description': 'IMDB/TMDB links'},
                    'README.txt': {'min_size_mb': 0, 'required': False, 'description': 'Dataset documentation'}
                },
                'total_min_size_mb': 850,
                'description': 'Core collaborative filtering dataset - required for training'
            },
            'archive': {
                'name': 'Netflix Prize Archive',
                'required': False,
                'priority': 'Medium',
                'files': {
                    'combined_data_1.txt': {'min_size_mb': 450, 'required': True, 'description': 'Netflix ratings part 1'},
                    'combined_data_2.txt': {'min_size_mb': 500, 'required': True, 'description': 'Netflix ratings part 2'},
                    'combined_data_3.txt': {'min_size_mb': 420, 'required': True, 'description': 'Netflix ratings part 3'},
                    'combined_data_4.txt': {'min_size_mb': 500, 'required': True, 'description': 'Netflix ratings part 4'},
                    'movie_titles.csv': {'min_size_mb': 0.5, 'required': True, 'description': 'Netflix movie catalog'},
                    'probe.txt': {'min_size_mb': 5, 'required': False, 'description': 'Test set'},
                    'qualifying.txt': {'min_size_mb': 40, 'required': False, 'description': 'Qualifying set'}
                },
                'total_min_size_mb': 1900,
                'description': 'Historic Netflix Prize data - adds 100M+ additional ratings'
            },
            'tmdb': {
                'name': 'TMDB Actor Data',
                'required': False,
                'priority': 'Medium',
                'files': {
                    'actor_filmography_data.csv': {'min_size_mb': 1000, 'required': True, 'description': 'Actor filmography'},
                    'actor_filmography_data.json': {'min_size_mb': 2000, 'required': False, 'description': 'Detailed actor data'},
                    'actor_filmography_data_movies.csv': {'min_size_mb': 900, 'required': False, 'description': 'Actor movie appearances'},
                    'actor_filmography_data_tv.csv': {'min_size_mb': 80, 'required': False, 'description': 'Actor TV appearances'}
                },
                'total_min_size_mb': 1000,
                'description': 'Actor and filmography data for content-based filtering'
            },
            'tv': {
                'name': 'TV Show Data',
                'required': False,
                'priority': 'Medium',
                'files': {
                    'netflix_tv_shows.csv': {'min_size_mb': 0.8, 'required': False, 'description': 'Netflix TV catalog'},
                    'tmdb-tv.zip': {'min_size_mb': 30, 'required': False, 'description': 'TMDB TV dataset'},
                    'anime-dataset.zip': {'min_size_mb': 200, 'required': False, 'description': 'MyAnimeList data'},
                    'imdb-tv.zip': {'min_size_mb': 25, 'required': False, 'description': 'IMDb TV series'},
                    'netflix-movies-and-tv.zip': {'min_size_mb': 1, 'required': False, 'description': 'Netflix catalog'}
                },
                'total_min_size_mb': 0,  # All optional
                'description': 'TV show datasets for multi-modal recommendations'
            },
            'movies': {
                'name': 'Additional Movie Data',
                'required': False,
                'priority': 'Low',
                'files': {
                    'netflix_movies.csv': {'min_size_mb': 2, 'required': False, 'description': 'Netflix movie catalog'},
                    'CineSync-Training-Dataset.zip': {'min_size_mb': 1300, 'required': False, 'description': 'Additional training data'},
                    'archive_netflix_prize.zip': {'min_size_mb': 600, 'required': False, 'description': 'Compressed Netflix Prize'},
                    'ml32m_movielens.zip': {'min_size_mb': 200, 'required': False, 'description': 'Compressed MovieLens'},
                    'tmdb_actors_movies.zip': {'min_size_mb': 450, 'required': False, 'description': 'Compressed TMDB data'}
                },
                'total_min_size_mb': 0,  # All optional
                'description': 'Additional movie datasets and compressed versions'
            }
        }

    def get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB."""
        if not file_path.exists():
            return 0
        return file_path.stat().st_size / (1024 * 1024)

    def check_file(self, file_path: Path, file_spec: Dict) -> Dict:
        """Check a single file against specifications."""
        result = {
            'exists': file_path.exists(),
            'size_mb': 0,
            'size_ok': False,
            'readable': False,
            'status': 'missing'
        }
        
        if result['exists']:
            try:
                result['size_mb'] = self.get_file_size_mb(file_path)
                result['size_ok'] = result['size_mb'] >= file_spec['min_size_mb']
                
                # Test readability
                if file_path.suffix.lower() == '.csv':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        f.readline()  # Try to read first line
                        result['readable'] = True
                elif file_path.suffix.lower() in ['.txt', '.json']:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        f.read(1024)  # Try to read first 1KB
                        result['readable'] = True
                else:
                    # For other files, just check if we can open them
                    with open(file_path, 'rb') as f:
                        f.read(1024)
                        result['readable'] = True
                
                if result['size_ok'] and result['readable']:
                    result['status'] = 'ok'
                elif result['readable']:
                    result['status'] = 'small'
                else:
                    result['status'] = 'corrupted'
                    
            except Exception as e:
                result['status'] = 'error'
                result['error'] = str(e)
        
        return result

    def check_dataset(self, dataset_name: str) -> Dict:
        """Check a complete dataset."""
        spec = self.datasets[dataset_name]
        dataset_path = self.project_root / dataset_name
        
        result = {
            'name': spec['name'],
            'required': spec['required'],
            'priority': spec['priority'],
            'description': spec['description'],
            'folder_exists': dataset_path.exists(),
            'total_size_mb': 0,
            'files': {},
            'required_files_ok': 0,
            'total_required_files': 0,
            'optional_files_ok': 0,
            'total_optional_files': 0,
            'status': 'missing'
        }
        
        if not result['folder_exists']:
            return result
        
        # Check each file
        for file_name, file_spec in spec['files'].items():
            file_path = dataset_path / file_name
            file_result = self.check_file(file_path, file_spec)
            file_result['required'] = file_spec['required']
            file_result['description'] = file_spec['description']
            result['files'][file_name] = file_result
            
            if file_result['exists']:
                result['total_size_mb'] += file_result['size_mb']
            
            # Count file status
            if file_spec['required']:
                result['total_required_files'] += 1
                if file_result['status'] == 'ok':
                    result['required_files_ok'] += 1
            else:
                result['total_optional_files'] += 1
                if file_result['status'] == 'ok':
                    result['optional_files_ok'] += 1
        
        # Determine overall dataset status
        if result['total_required_files'] == 0:
            # No required files, check if any files exist
            result['status'] = 'optional' if result['total_size_mb'] > 0 else 'empty'
        elif result['required_files_ok'] == result['total_required_files']:
            result['status'] = 'complete'
        elif result['required_files_ok'] > 0:
            result['status'] = 'partial'
        else:
            result['status'] = 'missing'
        
        return result

    def check_all_datasets(self) -> Dict[str, Dict]:
        """Check all datasets."""
        results = {}
        for dataset_name in self.datasets:
            results[dataset_name] = self.check_dataset(dataset_name)
        return results

    def print_quick_summary(self, results: Dict[str, Dict]):
        """Print a quick summary of dataset status."""
        print("\n🎬 CineSync Dataset Status Summary")
        print("=" * 50)
        
        required_ok = 0
        required_total = 0
        optional_available = 0
        
        for dataset_name, result in results.items():
            if result['required']:
                required_total += 1
                if result['status'] in ['complete', 'partial']:
                    required_ok += 1
            
            status_emoji = {
                'complete': '✅',
                'partial': '⚠️',
                'optional': '📦',
                'empty': '📁',
                'missing': '❌'
            }.get(result['status'], '❓')
            
            size_str = f"{result['total_size_mb']:.1f}MB" if result['total_size_mb'] > 0 else "0MB"
            
            print(f"{status_emoji} {result['name']}: {result['status'].upper()} ({size_str})")
            
            if result['status'] not in ['missing', 'empty']:
                optional_available += 1
        
        print(f"\n📊 Summary:")
        print(f"   Required datasets: {required_ok}/{required_total} available")
        print(f"   Optional datasets: {optional_available} have data")
        
        if required_ok == required_total:
            print("✅ All required datasets are available - ready for training!")
        else:
            print("⚠️  Some required datasets are missing - see detailed report below")

    def print_detailed_report(self, results: Dict[str, Dict]):
        """Print detailed dataset analysis."""
        print("\n" + "=" * 70)
        print("📊 Detailed Dataset Analysis")
        print("=" * 70)
        
        for dataset_name, result in results.items():
            print(f"\n📁 {result['name']} ({dataset_name}/)")
            print(f"   Priority: {result['priority']} | Required: {'Yes' if result['required'] else 'No'}")
            print(f"   Description: {result['description']}")
            print(f"   Status: {result['status'].upper()} | Total Size: {result['total_size_mb']:.1f}MB")
            
            if not result['folder_exists']:
                print("   ❌ Folder does not exist")
                continue
            
            # Group files by status
            ok_files = []
            problem_files = []
            missing_files = []
            
            for file_name, file_result in result['files'].items():
                if file_result['status'] == 'ok':
                    ok_files.append((file_name, file_result))
                elif file_result['exists']:
                    problem_files.append((file_name, file_result))
                else:
                    missing_files.append((file_name, file_result))
            
            # Print file status
            if ok_files:
                print("   ✅ Good files:")
                for file_name, file_result in ok_files:
                    req_str = "Required" if file_result['required'] else "Optional"
                    print(f"      {file_name} ({file_result['size_mb']:.1f}MB) - {req_str}")
            
            if problem_files:
                print("   ⚠️  Problem files:")
                for file_name, file_result in problem_files:
                    status_desc = {
                        'small': 'File too small',
                        'corrupted': 'File corrupted/unreadable',
                        'error': 'Access error'
                    }.get(file_result['status'], 'Unknown issue')
                    req_str = "Required" if file_result['required'] else "Optional"
                    print(f"      {file_name} ({file_result['size_mb']:.1f}MB) - {status_desc} ({req_str})")
            
            if missing_files:
                print("   ❌ Missing files:")
                for file_name, file_result in missing_files:
                    req_str = "Required" if file_result['required'] else "Optional"
                    print(f"      {file_name} - {file_result['description']} ({req_str})")

    def print_recommendations(self, results: Dict[str, Dict]):
        """Print recommendations based on dataset status."""
        print("\n" + "=" * 70)
        print("💡 Recommendations")
        print("=" * 70)
        
        # Check if training is possible
        required_datasets = [name for name, result in results.items() 
                           if result['required'] and result['status'] in ['complete', 'partial']]
        
        if required_datasets:
            print("\n✅ Training is possible with current datasets!")
            print("\n🚀 Next steps:")
            print("   1. Start training: python main.py")
            print("   2. Or try simplified training: python train_simple.py")
            
            # Check for missing optional datasets
            missing_optional = [result['name'] for result in results.values() 
                              if not result['required'] and result['status'] == 'missing']
            
            if missing_optional:
                print(f"\n📈 To improve recommendations, consider adding:")
                for dataset_name in missing_optional[:3]:  # Show top 3
                    print(f"   • {dataset_name}")
                
                print("\n   Download with: python setup_datasets.py")
        else:
            print("\n❌ Training is not possible - missing required datasets")
            print("\n📥 To get started:")
            print("   1. Download datasets: python setup_datasets.py")
            print("   2. Or get complete dataset from:")
            print("      https://kaggle.com/datasets/YOUR_USERNAME/cinesync-complete-training-dataset")
            print("   3. Then organize: python organize_datasets.py")
        
        # Disk space warning
        total_size = sum(result['total_size_mb'] for result in results.values())
        if total_size > 10000:  # 10GB
            print(f"\n💾 Disk space note: Current datasets use {total_size/1024:.1f}GB")
            print("   Consider removing compressed zip files after extraction")

    def save_report(self, results: Dict[str, Dict], output_file: str = "dataset_report.json"):
        """Save detailed report to JSON file."""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'summary': {
                'total_datasets': len(results),
                'required_complete': sum(1 for r in results.values() if r['required'] and r['status'] == 'complete'),
                'required_total': sum(1 for r in results.values() if r['required']),
                'total_size_mb': sum(r['total_size_mb'] for r in results.values())
            },
            'datasets': results
        }
        
        output_path = self.project_root / output_file
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {output_file}")

    def run_check(self, quick=False, detailed=False, save_report=False):
        """Run the complete dataset check."""
        print("🔍 CineSync Dataset Checker")
        print("=" * 40)
        
        # Check all datasets
        results = self.check_all_datasets()
        
        # Print appropriate level of detail
        if quick:
            self.print_quick_summary(results)
        else:
            self.print_quick_summary(results)
            if detailed:
                self.print_detailed_report(results)
            self.print_recommendations(results)
        
        # Save report if requested
        if save_report:
            self.save_report(results)
        
        # Return True if ready for training
        required_ok = sum(1 for r in results.values() 
                         if r['required'] and r['status'] in ['complete', 'partial'])
        required_total = sum(1 for r in results.values() if r['required'])
        
        return required_ok == required_total


def main():
    parser = argparse.ArgumentParser(description='Check CineSync dataset integrity and completeness')
    parser.add_argument('--quick', action='store_true', help='Quick check, just show status')
    parser.add_argument('--detailed', action='store_true', help='Detailed file-by-file analysis')
    parser.add_argument('--save-report', action='store_true', help='Save detailed report to JSON file')
    
    args = parser.parse_args()
    
    checker = DatasetChecker()
    ready_for_training = checker.run_check(
        quick=args.quick, 
        detailed=args.detailed, 
        save_report=args.save_report
    )
    
    # Exit with appropriate code
    sys.exit(0 if ready_for_training else 1)


if __name__ == '__main__':
    main()