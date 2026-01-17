#!/usr/bin/env python3
"""
CineSync v2 Dataset Refresh Script

Downloads and updates static datasets to ensure we have the latest data.

Datasets supported:
- MovieLens: Latest version from GroupLens
- TMDB: Daily exports (requires account)
- IMDB: Non-commercial datasets

Usage:
    python scripts/refresh_datasets.py --all
    python scripts/refresh_datasets.py --movielens
    python scripts/refresh_datasets.py --imdb
"""

import argparse
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import zipfile
import gzip
from datetime import datetime
from pathlib import Path
from typing import Optional
import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Dataset URLs and info
DATASETS = {
    "movielens_32m": {
        "name": "MovieLens 32M",
        "url": "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
        "dest": DATA_DIR / "movies" / "cinesync" / "ml-32m",
        "description": "32 million ratings from MovieLens users",
        "size_mb": 900,
    },
    "movielens_latest_small": {
        "name": "MovieLens Latest Small",
        "url": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
        "dest": DATA_DIR / "movies" / "movielens-latest-small",
        "description": "100,000 ratings (frequently updated)",
        "size_mb": 1,
    },
    "movielens_latest": {
        "name": "MovieLens Latest Full",
        "url": "https://files.grouplens.org/datasets/movielens/ml-latest.zip",
        "dest": DATA_DIR / "movies" / "movielens-latest",
        "description": "33+ million ratings (frequently updated)",
        "size_mb": 350,
    },
    "imdb_title_basics": {
        "name": "IMDB Title Basics",
        "url": "https://datasets.imdbws.com/title.basics.tsv.gz",
        "dest": DATA_DIR / "imdb" / "title.basics.tsv",
        "description": "Basic title information (type, title, year, runtime, genres)",
        "size_mb": 150,
        "compressed": True,
    },
    "imdb_title_ratings": {
        "name": "IMDB Title Ratings",
        "url": "https://datasets.imdbws.com/title.ratings.tsv.gz",
        "dest": DATA_DIR / "imdb" / "title.ratings.tsv",
        "description": "IMDb ratings and votes",
        "size_mb": 7,
        "compressed": True,
    },
    "imdb_title_episode": {
        "name": "IMDB Episodes",
        "url": "https://datasets.imdbws.com/title.episode.tsv.gz",
        "dest": DATA_DIR / "imdb" / "title.episode.tsv",
        "description": "TV episode information",
        "size_mb": 50,
        "compressed": True,
    },
    "imdb_name_basics": {
        "name": "IMDB Names",
        "url": "https://datasets.imdbws.com/name.basics.tsv.gz",
        "dest": DATA_DIR / "imdb" / "name.basics.tsv",
        "description": "Cast and crew information",
        "size_mb": 250,
        "compressed": True,
    },
    "imdb_title_crew": {
        "name": "IMDB Crew",
        "url": "https://datasets.imdbws.com/title.crew.tsv.gz",
        "dest": DATA_DIR / "imdb" / "title.crew.tsv",
        "description": "Directors and writers",
        "size_mb": 80,
        "compressed": True,
    },
    "imdb_title_principals": {
        "name": "IMDB Principals",
        "url": "https://datasets.imdbws.com/title.principals.tsv.gz",
        "dest": DATA_DIR / "imdb" / "title.principals.tsv",
        "description": "Principal cast/crew per title",
        "size_mb": 400,
        "compressed": True,
    },
    "imdb_title_akas": {
        "name": "IMDB AKAs",
        "url": "https://datasets.imdbws.com/title.akas.tsv.gz",
        "dest": DATA_DIR / "imdb" / "title.akas.tsv",
        "description": "Alternative titles/languages",
        "size_mb": 200,
        "compressed": True,
    },
}

# Dataset groups for convenience
DATASET_GROUPS = {
    "movielens": ["movielens_latest"],  # Use latest full by default
    "movielens_all": ["movielens_32m", "movielens_latest_small", "movielens_latest"],
    "imdb": ["imdb_title_basics", "imdb_title_ratings", "imdb_title_episode"],
    "imdb_full": list(k for k in DATASETS.keys() if k.startswith("imdb_")),
    "essential": ["movielens_latest", "imdb_title_basics", "imdb_title_ratings"],
    "all": list(DATASETS.keys()),
}


def download_file(url: str, dest: Path, desc: str = None) -> bool:
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Download to temp file first
        temp_dest = dest.with_suffix(dest.suffix + '.tmp')

        with open(temp_dest, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc or dest.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Move to final location
        shutil.move(temp_dest, dest)
        return True

    except Exception as e:
        logger.error(f"Download failed: {e}")
        if temp_dest.exists():
            temp_dest.unlink()
        return False


def extract_zip(zip_path: Path, dest_dir: Path) -> bool:
    """Extract a zip file"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Get the top-level directory in the zip
            namelist = zf.namelist()
            top_dirs = set(n.split('/')[0] for n in namelist if '/' in n)

            if len(top_dirs) == 1:
                # Extract contents of single top directory to dest
                top_dir = list(top_dirs)[0]
                for member in tqdm(namelist, desc="Extracting"):
                    if member.startswith(top_dir + '/'):
                        # Remove top directory from path
                        new_name = member[len(top_dir) + 1:]
                        if new_name:
                            source = zf.open(member)
                            target_path = dest_dir / new_name
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            if not member.endswith('/'):
                                with open(target_path, 'wb') as target:
                                    shutil.copyfileobj(source, target)
            else:
                # Extract directly
                zf.extractall(dest_dir)

        return True

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


def decompress_gzip(gz_path: Path, dest: Path) -> bool:
    """Decompress a gzip file"""
    try:
        with gzip.open(gz_path, 'rb') as f_in:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return True
    except Exception as e:
        logger.error(f"Decompression failed: {e}")
        return False


def download_dataset(dataset_id: str, force: bool = False) -> bool:
    """Download and extract a single dataset"""
    if dataset_id not in DATASETS:
        logger.error(f"Unknown dataset: {dataset_id}")
        return False

    info = DATASETS[dataset_id]
    dest = Path(info["dest"])
    url = info["url"]

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Dataset: {info['name']}")
    logger.info(f"Description: {info['description']}")
    logger.info(f"Estimated size: {info['size_mb']} MB")
    logger.info(f"Destination: {dest}")
    logger.info(f"{'=' * 50}")

    # Check if already exists
    if dest.exists() and not force:
        logger.info(f"Dataset already exists. Use --force to re-download.")
        return True

    # Create temp directory for downloads
    temp_dir = DATA_DIR / ".temp"
    temp_dir.mkdir(exist_ok=True)

    # Download
    if url.endswith('.zip'):
        temp_file = temp_dir / f"{dataset_id}.zip"
        logger.info(f"Downloading {url}...")
        if not download_file(url, temp_file, info['name']):
            return False

        # Extract
        logger.info("Extracting...")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not extract_zip(temp_file, dest):
            return False

        # Cleanup
        temp_file.unlink()

    elif url.endswith('.gz'):
        temp_file = temp_dir / Path(url).name
        logger.info(f"Downloading {url}...")
        if not download_file(url, temp_file, info['name']):
            return False

        # Decompress
        logger.info("Decompressing...")
        if not decompress_gzip(temp_file, dest):
            return False

        # Cleanup
        temp_file.unlink()

    else:
        # Direct download
        logger.info(f"Downloading {url}...")
        if not download_file(url, dest, info['name']):
            return False

    logger.info(f"Successfully downloaded: {info['name']}")
    return True


def download_datasets(dataset_ids: list, force: bool = False) -> dict:
    """Download multiple datasets"""
    results = {}
    for dataset_id in dataset_ids:
        results[dataset_id] = download_dataset(dataset_id, force)
    return results


def list_datasets():
    """List available datasets and their status"""
    print("\n" + "=" * 70)
    print("AVAILABLE DATASETS")
    print("=" * 70)

    for dataset_id, info in DATASETS.items():
        dest = Path(info["dest"])
        status = "INSTALLED" if dest.exists() else "NOT INSTALLED"
        size = info["size_mb"]
        print(f"\n{dataset_id}")
        print(f"  Name: {info['name']}")
        print(f"  Description: {info['description']}")
        print(f"  Size: ~{size} MB")
        print(f"  Status: {status}")
        print(f"  Path: {dest}")

    print("\n" + "=" * 70)
    print("DATASET GROUPS")
    print("=" * 70)

    for group_name, datasets in DATASET_GROUPS.items():
        total_size = sum(DATASETS[d]["size_mb"] for d in datasets)
        print(f"\n--{group_name}")
        print(f"  Datasets: {', '.join(datasets)}")
        print(f"  Total size: ~{total_size} MB")

    print("\n")


def get_dataset_status() -> dict:
    """Get status of all datasets"""
    status = {}
    for dataset_id, info in DATASETS.items():
        dest = Path(info["dest"])
        status[dataset_id] = {
            "installed": dest.exists(),
            "path": str(dest),
            "name": info["name"],
        }
        if dest.exists():
            if dest.is_file():
                status[dataset_id]["size_bytes"] = dest.stat().st_size
                status[dataset_id]["modified"] = datetime.fromtimestamp(
                    dest.stat().st_mtime
                ).isoformat()
            elif dest.is_dir():
                total_size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
                status[dataset_id]["size_bytes"] = total_size
    return status


def main():
    parser = argparse.ArgumentParser(
        description='CineSync v2 Dataset Refresh Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available datasets
  python scripts/refresh_datasets.py --list

  # Download essential datasets (MovieLens + IMDB basics)
  python scripts/refresh_datasets.py --essential

  # Download only MovieLens
  python scripts/refresh_datasets.py --movielens

  # Download only IMDB data
  python scripts/refresh_datasets.py --imdb

  # Download everything
  python scripts/refresh_datasets.py --all

  # Force re-download
  python scripts/refresh_datasets.py --movielens --force
        """
    )

    parser.add_argument('--list', action='store_true',
                        help='List available datasets')
    parser.add_argument('--status', action='store_true',
                        help='Show status of installed datasets')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download even if exists')

    # Dataset groups
    for group_name in DATASET_GROUPS:
        parser.add_argument(f'--{group_name}', action='store_true',
                            help=f'Download {group_name} datasets')

    # Individual datasets
    parser.add_argument('--dataset', type=str, action='append',
                        help='Download specific dataset by ID')

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return 0

    if args.status:
        status = get_dataset_status()
        print("\n" + "=" * 70)
        print("DATASET STATUS")
        print("=" * 70)
        for dataset_id, info in status.items():
            status_str = "INSTALLED" if info["installed"] else "NOT INSTALLED"
            print(f"\n{dataset_id}: {status_str}")
            if info["installed"]:
                size_mb = info.get("size_bytes", 0) / (1024 * 1024)
                print(f"  Size: {size_mb:.1f} MB")
                print(f"  Modified: {info.get('modified', 'N/A')}")
        print("\n")
        return 0

    # Collect datasets to download
    datasets_to_download = []

    for group_name, datasets in DATASET_GROUPS.items():
        if getattr(args, group_name.replace('-', '_'), False):
            datasets_to_download.extend(datasets)

    if args.dataset:
        datasets_to_download.extend(args.dataset)

    # Remove duplicates while preserving order
    seen = set()
    datasets_to_download = [d for d in datasets_to_download
                            if not (d in seen or seen.add(d))]

    if not datasets_to_download:
        parser.print_help()
        print("\nNo datasets specified. Use --list to see available datasets.")
        return 1

    # Calculate total size
    total_size = sum(DATASETS[d]["size_mb"] for d in datasets_to_download if d in DATASETS)
    print(f"\nWill download {len(datasets_to_download)} dataset(s), ~{total_size} MB total")
    print(f"Datasets: {', '.join(datasets_to_download)}")

    # Download
    results = download_datasets(datasets_to_download, args.force)

    # Summary
    print("\n" + "=" * 50)
    print("DOWNLOAD SUMMARY")
    print("=" * 50)
    success_count = sum(1 for v in results.values() if v)
    fail_count = len(results) - success_count

    for dataset_id, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {dataset_id}: {status}")

    print(f"\nTotal: {success_count} succeeded, {fail_count} failed")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
