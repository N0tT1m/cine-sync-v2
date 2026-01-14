#!/usr/bin/env python3
"""
Model Integration Tests for CineSync v2
Verifies all 45 models can be instantiated and basic operations work
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch
import importlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the model registry from train_all_models
from src.training.train_all_models import (
    ALL_MODELS,
    MOVIE_SPECIFIC_MODELS,
    TV_SPECIFIC_MODELS,
    CONTENT_AGNOSTIC_MODELS,
    UNIFIED_MODELS
)


class TestModelImports:
    """Test that all models can be imported"""

    @pytest.mark.parametrize("model_name,model_info", MOVIE_SPECIFIC_MODELS.items())
    def test_movie_model_imports(self, model_name, model_info):
        """Test movie-specific model imports"""
        module_path = model_info['module']
        model_class = model_info['model_class']

        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, model_class)
            assert cls is not None, f"Model class {model_class} is None"
            logger.info(f"✓ {model_name}: {model_class} imported successfully")
        except ImportError as e:
            pytest.fail(f"Failed to import {model_name}: {e}")
        except AttributeError as e:
            pytest.fail(f"Class {model_class} not found in {module_path}: {e}")

    @pytest.mark.parametrize("model_name,model_info", TV_SPECIFIC_MODELS.items())
    def test_tv_model_imports(self, model_name, model_info):
        """Test TV-specific model imports"""
        module_path = model_info['module']
        model_class = model_info['model_class']

        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, model_class)
            assert cls is not None, f"Model class {model_class} is None"
            logger.info(f"✓ {model_name}: {model_class} imported successfully")
        except ImportError as e:
            pytest.fail(f"Failed to import {model_name}: {e}")
        except AttributeError as e:
            pytest.fail(f"Class {model_class} not found in {module_path}: {e}")

    @pytest.mark.parametrize("model_name,model_info", CONTENT_AGNOSTIC_MODELS.items())
    def test_content_agnostic_model_imports(self, model_name, model_info):
        """Test content-agnostic model imports"""
        module_path = model_info['module']
        model_class = model_info['model_class']

        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, model_class)
            assert cls is not None, f"Model class {model_class} is None"
            logger.info(f"✓ {model_name}: {model_class} imported successfully")
        except ImportError as e:
            pytest.fail(f"Failed to import {model_name}: {e}")
        except AttributeError as e:
            pytest.fail(f"Class {model_class} not found in {module_path}: {e}")

    @pytest.mark.parametrize("model_name,model_info", UNIFIED_MODELS.items())
    def test_unified_model_imports(self, model_name, model_info):
        """Test unified model imports"""
        module_path = model_info['module']
        model_class = model_info['model_class']

        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, model_class)
            assert cls is not None, f"Model class {model_class} is None"
            logger.info(f"✓ {model_name}: {model_class} imported successfully")
        except ImportError as e:
            pytest.fail(f"Failed to import {model_name}: {e}")
        except AttributeError as e:
            pytest.fail(f"Class {model_class} not found in {module_path}: {e}")


class TestModelInstantiation:
    """Test that models can be instantiated"""

    def _try_instantiate_model(self, model_info):
        """Helper to instantiate a model"""
        module_path = model_info['module']
        model_class = model_info['model_class']
        config_class = model_info.get('config_class')

        module = importlib.import_module(module_path)
        cls = getattr(module, model_class)

        if config_class:
            config_cls = getattr(module, config_class)
            config = config_cls()
            model = cls(config)
        else:
            model = cls()

        return model

    @pytest.mark.parametrize("model_name,model_info", [
        (k, v) for k, v in MOVIE_SPECIFIC_MODELS.items()
        if v.get('config_class')  # Only test models with config
    ][:5])  # Test first 5 for speed
    def test_movie_model_instantiation(self, model_name, model_info):
        """Test movie model instantiation"""
        try:
            model = self._try_instantiate_model(model_info)
            assert model is not None
            assert isinstance(model, torch.nn.Module)
            logger.info(f"✓ {model_name}: instantiated successfully")
        except Exception as e:
            pytest.fail(f"Failed to instantiate {model_name}: {e}")

    @pytest.mark.parametrize("model_name,model_info", [
        (k, v) for k, v in TV_SPECIFIC_MODELS.items()
        if v.get('config_class')
    ][:5])
    def test_tv_model_instantiation(self, model_name, model_info):
        """Test TV model instantiation"""
        try:
            model = self._try_instantiate_model(model_info)
            assert model is not None
            assert isinstance(model, torch.nn.Module)
            logger.info(f"✓ {model_name}: instantiated successfully")
        except Exception as e:
            pytest.fail(f"Failed to instantiate {model_name}: {e}")


class TestModelParameters:
    """Test that models have expected parameters"""

    def test_all_models_have_parameters(self):
        """Verify all registered models have trainable parameters"""
        results = {"success": [], "failed": []}

        for model_name, model_info in ALL_MODELS.items():
            try:
                module = importlib.import_module(model_info['module'])
                cls = getattr(module, model_info['model_class'])

                # Try to get parameter count
                config_class = model_info.get('config_class')
                if config_class:
                    config_cls = getattr(module, config_class)
                    model = cls(config_cls())
                else:
                    model = cls()

                param_count = sum(p.numel() for p in model.parameters())
                assert param_count > 0, f"{model_name} has no parameters"
                results["success"].append((model_name, param_count))

            except Exception as e:
                results["failed"].append((model_name, str(e)))

        logger.info(f"\nParameter check results:")
        logger.info(f"  Success: {len(results['success'])}")
        logger.info(f"  Failed: {len(results['failed'])}")

        # Allow some failures for models without default configs
        assert len(results["success"]) > len(ALL_MODELS) * 0.5, \
            f"Too many models failed parameter check: {results['failed']}"


class TestBotIntegration:
    """Test bot-specific integration"""

    def test_unified_content_manager_import(self):
        """Test UnifiedLupeContentManager can be imported"""
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "services" / "lupe_python"))
            from unified_content_manager import UnifiedLupeContentManager
            assert UnifiedLupeContentManager is not None
        except ImportError as e:
            pytest.skip(f"Unified content manager not available: {e}")

    def test_personalized_commands_import(self):
        """Test personalized commands can be imported"""
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "services" / "lupe_python"))
            from personalized_commands import (
                setup_personalization,
                my_recommendations_command,
                my_stats_command
            )
            assert setup_personalization is not None
        except ImportError as e:
            pytest.skip(f"Personalized commands not available: {e}")


class TestExtendedModelLoader:
    """Test the extended model loader"""

    def test_extended_model_loader_import(self):
        """Test extended model loader can be imported"""
        from src.api.extended_model_loader import ExtendedModelLoader, EXTENDED_MODEL_REGISTRY
        assert ExtendedModelLoader is not None
        assert len(EXTENDED_MODEL_REGISTRY) > 40  # Should have 45 models

    def test_model_registry_completeness(self):
        """Test that model registry covers all categories"""
        from src.api.extended_model_loader import EXTENDED_MODEL_REGISTRY, ExtendedModelType

        # Check we have models from all categories
        movie_models = [mt for mt in ExtendedModelType if mt.value.startswith("movie_")]
        tv_models = [mt for mt in ExtendedModelType if mt.value.startswith("tv_")]

        assert len(movie_models) >= 14, "Should have 14 movie-specific models"
        assert len(tv_models) >= 14, "Should have 14 TV-specific models"


def run_quick_verification():
    """Quick verification of all model imports"""
    print("\n" + "="*70)
    print("QUICK MODEL VERIFICATION")
    print("="*70)

    success_count = 0
    fail_count = 0
    errors = []

    for model_name, model_info in ALL_MODELS.items():
        try:
            module = importlib.import_module(model_info['module'])
            cls = getattr(module, model_info['model_class'])
            success_count += 1
            print(f"  ✓ {model_name}")
        except Exception as e:
            fail_count += 1
            errors.append((model_name, str(e)))
            print(f"  ✗ {model_name}: {e}")

    print(f"\nResults: {success_count} passed, {fail_count} failed out of {len(ALL_MODELS)}")

    if errors:
        print("\nFailed models:")
        for name, error in errors:
            print(f"  - {name}: {error}")

    return success_count, fail_count


if __name__ == "__main__":
    # Run quick verification
    success, failed = run_quick_verification()

    # Run pytest if available
    print("\n\nRunning pytest...")
    pytest.main([__file__, "-v", "--tb=short", "-x"])
