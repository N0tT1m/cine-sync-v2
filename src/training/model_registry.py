#!/usr/bin/env python3
"""
Local Model Registry for CineSync AI Models

A lightweight, file-based model registry that runs entirely on local infrastructure.
No cloud services, no costs - just organized model versioning and deployment.

Features:
- Versioned model storage with metadata
- Automatic "latest" symlink management
- Model metrics and training history tracking
- Simple publish/pull interface
- CLI for model management

Registry Structure:
/path/to/registry/
├── registry.json              # Central index
├── movies/
│   ├── movie_actor_collaboration/
│   │   ├── v1/
│   │   │   ├── checkpoint.pt
│   │   │   └── metadata.json
│   │   ├── v2/
│   │   │   └── ...
│   │   └── latest -> v2/      # Symlink
│   └── ...
├── tv/
│   └── ...
└── unified/
    └── ...
"""

import os
import sys
import json
import shutil
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import torch

logger = logging.getLogger(__name__)

# Default registry path - shared model registry for deployment
# Script is at: cine-sync-v2/src/training/model_registry.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_REGISTRY_PATH = os.getenv(
    'CINESYNC_MODEL_REGISTRY',
    str(_PROJECT_ROOT / '..' / '..' / 'public-projects' / 'mommy-milk-me-v2')
)


@dataclass
class ModelMetadata:
    """Metadata for a published model version"""
    model_name: str
    version: str
    category: str  # movies, tv, unified
    content_type: str  # movie, tv, both
    published_at: str
    trained_at: Optional[str] = None
    epochs: Optional[int] = None
    metrics: Optional[Dict[str, float]] = None
    training_config: Optional[Dict[str, Any]] = None
    checkpoint_hash: Optional[str] = None
    description: Optional[str] = None
    source_path: Optional[str] = None
    gpu_profile: Optional[str] = None
    file_size_mb: Optional[float] = None


@dataclass
class RegistryIndex:
    """Central registry index"""
    created_at: str
    updated_at: str
    total_models: int
    models: Dict[str, Dict[str, Any]]  # model_name -> {latest_version, versions, category}


class LocalModelRegistry:
    """
    Local file-based model registry for CineSync AI models.

    Usage:
        # Publishing (from training)
        registry = LocalModelRegistry()
        registry.publish_model(
            model_name='movie_actor_collaboration',
            checkpoint_path='/path/to/checkpoint.pt',
            category='movies',
            metrics={'mse': 0.006, 'accuracy': 0.92}
        )

        # Pulling (from inference)
        registry = LocalModelRegistry()
        checkpoint_path = registry.get_model_path('movie_actor_collaboration')
        # or specific version
        checkpoint_path = registry.get_model_path('movie_actor_collaboration', version='v2')
    """

    def __init__(self, registry_path: str = None):
        """
        Initialize the model registry.

        Args:
            registry_path: Path to registry directory. Defaults to CINESYNC_MODEL_REGISTRY env var
                          or ./model-registry (relative to project root)
        """
        self.registry_path = Path(registry_path or DEFAULT_REGISTRY_PATH)
        self.index_path = self.registry_path / 'registry.json'
        self._ensure_registry_exists()
        self._index = self._load_index()

    def _ensure_registry_exists(self):
        """Create registry directory structure if it doesn't exist"""
        self.registry_path.mkdir(parents=True, exist_ok=True)
        for category in ['movies', 'tv', 'unified']:
            (self.registry_path / category).mkdir(exist_ok=True)

    def _load_index(self) -> RegistryIndex:
        """Load or create the registry index"""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r') as f:
                    data = json.load(f)
                return RegistryIndex(**data)
            except Exception as e:
                logger.warning(f"Failed to load index, creating new: {e}")

        # Create new index
        now = datetime.now().isoformat()
        return RegistryIndex(
            created_at=now,
            updated_at=now,
            total_models=0,
            models={}
        )

    def _save_index(self):
        """Save the registry index"""
        self._index.updated_at = datetime.now().isoformat()
        with open(self.index_path, 'w') as f:
            json.dump(asdict(self._index), f, indent=2)

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]  # Short hash

    def _get_next_version(self, model_name: str) -> str:
        """Get the next version number for a model"""
        if model_name not in self._index.models:
            return 'v1'

        versions = self._index.models[model_name].get('versions', [])
        if not versions:
            return 'v1'

        # Extract version numbers and find max
        max_version = 0
        for v in versions:
            try:
                num = int(v.replace('v', ''))
                max_version = max(max_version, num)
            except ValueError:
                continue

        return f'v{max_version + 1}'

    def publish_model(
        self,
        model_name: str,
        checkpoint_path: str,
        category: str,
        content_type: str = 'both',
        metrics: Dict[str, float] = None,
        training_config: Dict[str, Any] = None,
        description: str = None,
        version: str = None,
        trained_at: str = None,
        epochs: int = None,
        gpu_profile: str = None
    ) -> str:
        """
        Publish a trained model to the registry.

        Args:
            model_name: Name of the model (e.g., 'movie_actor_collaboration')
            checkpoint_path: Path to the checkpoint file (.pt)
            category: Model category ('movies', 'tv', 'unified')
            content_type: Content type ('movie', 'tv', 'both')
            metrics: Training/validation metrics
            training_config: Training configuration used
            description: Human-readable description
            version: Specific version (auto-increments if not provided)
            trained_at: Training timestamp
            epochs: Number of epochs trained
            gpu_profile: GPU profile used for training

        Returns:
            Version string of the published model
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Validate category
        if category not in ['movies', 'tv', 'unified']:
            raise ValueError(f"Invalid category: {category}. Must be movies, tv, or unified")

        # Determine version
        if version is None:
            version = self._get_next_version(model_name)

        # Create model directory structure
        model_dir = self.registry_path / category / model_name
        version_dir = model_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copy checkpoint
        dest_checkpoint = version_dir / 'checkpoint.pt'
        shutil.copy2(checkpoint_path, dest_checkpoint)
        logger.info(f"Copied checkpoint to {dest_checkpoint}")

        # Compute hash and file size
        file_hash = self._compute_file_hash(dest_checkpoint)
        file_size_mb = dest_checkpoint.stat().st_size / (1024 * 1024)

        # Create metadata
        metadata = ModelMetadata(
            model_name=model_name,
            version=version,
            category=category,
            content_type=content_type,
            published_at=datetime.now().isoformat(),
            trained_at=trained_at,
            epochs=epochs,
            metrics=metrics,
            training_config=training_config,
            checkpoint_hash=file_hash,
            description=description,
            source_path=str(checkpoint_path),
            gpu_profile=gpu_profile,
            file_size_mb=round(file_size_mb, 2)
        )

        # Save metadata
        metadata_path = version_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)

        # Update latest symlink
        latest_link = model_dir / 'latest'
        if latest_link.is_symlink():
            latest_link.unlink()
        elif latest_link.exists():
            shutil.rmtree(latest_link)
        latest_link.symlink_to(version, target_is_directory=True)

        # Update index
        if model_name not in self._index.models:
            self._index.models[model_name] = {
                'category': category,
                'content_type': content_type,
                'versions': [],
                'latest_version': None
            }

        if version not in self._index.models[model_name]['versions']:
            self._index.models[model_name]['versions'].append(version)
        self._index.models[model_name]['latest_version'] = version
        self._index.models[model_name]['latest_metrics'] = metrics
        self._index.total_models = len(self._index.models)
        self._save_index()

        logger.info(f"Published {model_name} {version} to registry ({file_size_mb:.1f} MB)")
        return version

    def get_model_path(
        self,
        model_name: str,
        version: str = 'latest'
    ) -> Optional[Path]:
        """
        Get the path to a model checkpoint.

        Args:
            model_name: Name of the model
            version: Version to get ('latest' or specific like 'v2')

        Returns:
            Path to checkpoint file, or None if not found
        """
        if model_name not in self._index.models:
            logger.warning(f"Model not found in registry: {model_name}")
            return None

        category = self._index.models[model_name]['category']
        model_dir = self.registry_path / category / model_name

        if version == 'latest':
            checkpoint_path = model_dir / 'latest' / 'checkpoint.pt'
        else:
            checkpoint_path = model_dir / version / 'checkpoint.pt'

        if checkpoint_path.exists():
            return checkpoint_path

        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return None

    def get_model_metadata(
        self,
        model_name: str,
        version: str = 'latest'
    ) -> Optional[ModelMetadata]:
        """Get metadata for a specific model version"""
        if model_name not in self._index.models:
            return None

        category = self._index.models[model_name]['category']
        model_dir = self.registry_path / category / model_name

        if version == 'latest':
            metadata_path = model_dir / 'latest' / 'metadata.json'
        else:
            metadata_path = model_dir / version / 'metadata.json'

        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                data = json.load(f)
            return ModelMetadata(**data)

        return None

    def list_models(self, category: str = None) -> List[Dict[str, Any]]:
        """
        List all models in the registry.

        Args:
            category: Filter by category ('movies', 'tv', 'unified') or None for all

        Returns:
            List of model info dicts
        """
        models = []
        for name, info in self._index.models.items():
            if category and info['category'] != category:
                continue
            models.append({
                'name': name,
                'category': info['category'],
                'content_type': info['content_type'],
                'latest_version': info['latest_version'],
                'version_count': len(info['versions']),
                'versions': info['versions'],
                'latest_metrics': info.get('latest_metrics')
            })
        return models

    def list_versions(self, model_name: str) -> List[str]:
        """List all versions of a model"""
        if model_name not in self._index.models:
            return []
        return self._index.models[model_name]['versions']

    def rollback(self, model_name: str, version: str) -> bool:
        """
        Rollback a model to a specific version (update 'latest' symlink).

        Args:
            model_name: Name of the model
            version: Version to rollback to

        Returns:
            True if successful
        """
        if model_name not in self._index.models:
            logger.error(f"Model not found: {model_name}")
            return False

        if version not in self._index.models[model_name]['versions']:
            logger.error(f"Version not found: {version}")
            return False

        category = self._index.models[model_name]['category']
        model_dir = self.registry_path / category / model_name
        version_dir = model_dir / version

        if not version_dir.exists():
            logger.error(f"Version directory not found: {version_dir}")
            return False

        # Update symlink
        latest_link = model_dir / 'latest'
        if latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(version, target_is_directory=True)

        # Update index
        self._index.models[model_name]['latest_version'] = version
        metadata = self.get_model_metadata(model_name, version)
        if metadata and metadata.metrics:
            self._index.models[model_name]['latest_metrics'] = metadata.metrics
        self._save_index()

        logger.info(f"Rolled back {model_name} to {version}")
        return True

    def delete_version(self, model_name: str, version: str) -> bool:
        """Delete a specific version of a model (cannot delete latest)"""
        if model_name not in self._index.models:
            return False

        info = self._index.models[model_name]
        if info['latest_version'] == version:
            logger.error("Cannot delete the latest version. Rollback first.")
            return False

        if version not in info['versions']:
            return False

        category = info['category']
        version_dir = self.registry_path / category / model_name / version

        if version_dir.exists():
            shutil.rmtree(version_dir)

        info['versions'].remove(version)
        self._save_index()

        logger.info(f"Deleted {model_name} {version}")
        return True

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get overall registry statistics"""
        stats = {
            'total_models': self._index.total_models,
            'registry_path': str(self.registry_path),
            'created_at': self._index.created_at,
            'updated_at': self._index.updated_at,
            'by_category': {
                'movies': 0,
                'tv': 0,
                'unified': 0
            },
            'total_versions': 0,
            'total_size_mb': 0
        }

        for name, info in self._index.models.items():
            stats['by_category'][info['category']] += 1
            stats['total_versions'] += len(info['versions'])

        # Calculate total size
        for category in ['movies', 'tv', 'unified']:
            category_path = self.registry_path / category
            if category_path.exists():
                for checkpoint in category_path.rglob('checkpoint.pt'):
                    stats['total_size_mb'] += checkpoint.stat().st_size / (1024 * 1024)

        stats['total_size_mb'] = round(stats['total_size_mb'], 2)
        return stats


# Singleton instance
_registry_instance: Optional[LocalModelRegistry] = None


def get_model_registry(registry_path: str = None) -> LocalModelRegistry:
    """Get or create the global registry instance"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = LocalModelRegistry(registry_path)
    return _registry_instance


# =============================================================================
# CLI Interface
# =============================================================================
def main():
    """CLI for model registry management"""
    import argparse

    parser = argparse.ArgumentParser(description='CineSync Local Model Registry')
    parser.add_argument('--registry-path', default=DEFAULT_REGISTRY_PATH,
                       help='Path to model registry')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # List command
    list_parser = subparsers.add_parser('list', help='List models')
    list_parser.add_argument('--category', choices=['movies', 'tv', 'unified'],
                            help='Filter by category')

    # Info command
    info_parser = subparsers.add_parser('info', help='Get model info')
    info_parser.add_argument('model_name', help='Model name')
    info_parser.add_argument('--version', default='latest', help='Version')

    # Publish command
    publish_parser = subparsers.add_parser('publish', help='Publish a model')
    publish_parser.add_argument('model_name', help='Model name')
    publish_parser.add_argument('checkpoint', help='Path to checkpoint file')
    publish_parser.add_argument('--category', required=True,
                               choices=['movies', 'tv', 'unified'])
    publish_parser.add_argument('--content-type', default='both',
                               choices=['movie', 'tv', 'both'])
    publish_parser.add_argument('--description', help='Model description')

    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback to version')
    rollback_parser.add_argument('model_name', help='Model name')
    rollback_parser.add_argument('version', help='Version to rollback to')

    # Stats command
    subparsers.add_parser('stats', help='Show registry statistics')

    # Path command
    path_parser = subparsers.add_parser('path', help='Get model checkpoint path')
    path_parser.add_argument('model_name', help='Model name')
    path_parser.add_argument('--version', default='latest', help='Version')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # Create registry
    registry = LocalModelRegistry(args.registry_path)

    if args.command == 'list':
        models = registry.list_models(args.category)
        if not models:
            print("No models in registry")
            return

        print(f"\n{'Model':<35} {'Category':<10} {'Latest':<8} {'Versions':<10}")
        print("-" * 70)
        for m in models:
            print(f"{m['name']:<35} {m['category']:<10} {m['latest_version']:<8} {m['version_count']}")
        print(f"\nTotal: {len(models)} models")

    elif args.command == 'info':
        metadata = registry.get_model_metadata(args.model_name, args.version)
        if not metadata:
            print(f"Model not found: {args.model_name}")
            return

        print(f"\n=== {metadata.model_name} ({metadata.version}) ===")
        print(f"Category: {metadata.category}")
        print(f"Content Type: {metadata.content_type}")
        print(f"Published: {metadata.published_at}")
        if metadata.trained_at:
            print(f"Trained: {metadata.trained_at}")
        if metadata.epochs:
            print(f"Epochs: {metadata.epochs}")
        if metadata.metrics:
            print(f"Metrics: {json.dumps(metadata.metrics, indent=2)}")
        if metadata.file_size_mb:
            print(f"Size: {metadata.file_size_mb} MB")
        if metadata.checkpoint_hash:
            print(f"Hash: {metadata.checkpoint_hash}")

    elif args.command == 'publish':
        version = registry.publish_model(
            model_name=args.model_name,
            checkpoint_path=args.checkpoint,
            category=args.category,
            content_type=args.content_type,
            description=args.description
        )
        print(f"Published {args.model_name} as {version}")

    elif args.command == 'rollback':
        if registry.rollback(args.model_name, args.version):
            print(f"Rolled back {args.model_name} to {args.version}")
        else:
            print("Rollback failed")
            sys.exit(1)

    elif args.command == 'stats':
        stats = registry.get_registry_stats()
        print(f"\n=== Model Registry Stats ===")
        print(f"Path: {stats['registry_path']}")
        print(f"Total Models: {stats['total_models']}")
        print(f"Total Versions: {stats['total_versions']}")
        print(f"Total Size: {stats['total_size_mb']} MB")
        print(f"\nBy Category:")
        for cat, count in stats['by_category'].items():
            print(f"  {cat}: {count}")

    elif args.command == 'path':
        path = registry.get_model_path(args.model_name, args.version)
        if path:
            print(path)
        else:
            print(f"Model not found: {args.model_name}", file=sys.stderr)
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
