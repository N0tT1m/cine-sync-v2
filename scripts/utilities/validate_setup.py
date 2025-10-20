#!/usr/bin/env python3
"""
Validation script to check CineSync v2 setup
Verifies that all models have W&B integration and the data pipeline is ready
"""

import os
import sys
from pathlib import Path
import json

def check_file_exists(file_path, description):
    """Check if file exists and report status"""
    path = Path(file_path)
    if path.exists():
        size = path.stat().st_size
        print(f"✓ {description}: {file_path} ({size:,} bytes)")
        return True
    else:
        print(f"✗ {description}: {file_path} - NOT FOUND")
        return False

def check_directory_exists(dir_path, description):
    """Check if directory exists and report status"""
    path = Path(dir_path)
    if path.exists() and path.is_dir():
        files = list(path.glob("*"))
        print(f"✓ {description}: {dir_path} ({len(files)} files)")
        return True
    else:
        print(f"✗ {description}: {dir_path} - NOT FOUND")
        return False

def validate_wandb_integration():
    """Validate W&B integration files"""
    print("\n=== Weights & Biases Integration ===")
    
    wandb_files = [
        ("wandb_config.py", "W&B Configuration"),
        ("wandb_training_integration.py", "W&B Training Integration"),
        ("wandb_experiment_manager.py", "W&B Experiment Manager"),
        ("wandb_inference_monitor.py", "W&B Inference Monitor"),
    ]
    
    all_exist = True
    for file_path, description in wandb_files:
        if not check_file_exists(file_path, description):
            all_exist = False
    
    return all_exist

def validate_model_training_scripts():
    """Validate that all models have W&B training scripts"""
    print("\n=== Model Training Scripts with W&B ===")
    
    training_scripts = [
        ("neural_collaborative_filtering/train_with_wandb.py", "NCF W&B Training"),
        ("sequential_models/train_with_wandb.py", "Sequential Model W&B Training"),
        ("two_tower_model/train_with_wandb.py", "Two-Tower W&B Training"),
        ("hybrid_recommendation/train_with_wandb.py", "Hybrid Model W&B Training"),
    ]
    
    all_exist = True
    for file_path, description in training_scripts:
        if not check_file_exists(file_path, description):
            all_exist = False
    
    return all_exist

def validate_model_files():
    """Validate core model implementation files"""
    print("\n=== Core Model Files ===")
    
    model_files = [
        ("neural_collaborative_filtering/src/model.py", "NCF Model"),
        ("sequential_models/src/model.py", "Sequential Model"),
        ("two_tower_model/src/model.py", "Two-Tower Model"),
        ("advanced_models/enhanced_two_tower.py", "Enhanced Two-Tower Model"),
        ("hybrid_recommendation/models/tv_recommender.py", "TV Recommender Model"),
    ]
    
    all_exist = True
    for file_path, description in model_files:
        if not check_file_exists(file_path, description):
            all_exist = False
    
    return all_exist

def validate_data_pipeline():
    """Validate data import pipeline"""
    print("\n=== Data Import Pipeline ===")
    
    pipeline_files = [
        ("data_import_pipeline.py", "Main Data Import Pipeline"),
    ]
    
    all_exist = True
    for file_path, description in pipeline_files:
        if not check_file_exists(file_path, description):
            all_exist = False
    
    return all_exist

def validate_requirements():
    """Check requirements files"""
    print("\n=== Requirements Files ===")
    
    requirements_files = [
        ("neural_collaborative_filtering/requirements.txt", "NCF Requirements"),
        ("sequential_models/requirements.txt", "Sequential Requirements"),
        ("two_tower_model/requirements.txt", "Two-Tower Requirements"),
        ("hybrid_recommendation/requirements.txt", "Hybrid Requirements"),
    ]
    
    all_exist = True
    for file_path, description in requirements_files:
        if not check_file_exists(file_path, description):
            all_exist = False
    
    return all_exist

def check_wandb_in_requirements():
    """Check if wandb is in requirements files"""
    print("\n=== W&B in Requirements ===")
    
    requirements_files = [
        "neural_collaborative_filtering/requirements.txt",
        "sequential_models/requirements.txt", 
        "two_tower_model/requirements.txt",
        "hybrid_recommendation/requirements.txt",
    ]
    
    for req_file in requirements_files:
        path = Path(req_file)
        if path.exists():
            content = path.read_text()
            if 'wandb' in content.lower():
                print(f"✓ {req_file}: contains wandb")
            else:
                print(f"⚠ {req_file}: wandb not found")
        else:
            print(f"✗ {req_file}: file not found")

def validate_test_files():
    """Check test files"""
    print("\n=== Test Files ===")
    
    test_files = [
        ("test_all_models.py", "All Models Test"),
        ("model_memory_profiler.py", "Memory Profiler"),
    ]
    
    all_exist = True
    for file_path, description in test_files:
        if not check_file_exists(file_path, description):
            all_exist = False
    
    return all_exist

def generate_usage_summary():
    """Generate usage summary"""
    print("\n=== Usage Summary ===")
    print("""
To use the CineSync v2 system:

1. Install Dependencies:
   cd neural_collaborative_filtering && pip install -r requirements.txt
   cd ../sequential_models && pip install -r requirements.txt
   cd ../two_tower_model && pip install -r requirements.txt
   cd ../hybrid_recommendation && pip install -r requirements.txt

2. Set up W&B:
   wandb login  # Login to your W&B account
   export WANDB_PROJECT="cinesync-v2"

3. Train Models with W&B:
   python neural_collaborative_filtering/train_with_wandb.py
   python sequential_models/train_with_wandb.py
   python two_tower_model/train_with_wandb.py
   python hybrid_recommendation/train_with_wandb.py

4. Import New Data (post-2022):
   export TMDB_API_KEY="your_api_key_here"
   python data_import_pipeline.py --start-date 2022-01-01

5. Test All Models:
   python test_all_models.py

6. Monitor in W&B:
   - Training metrics and losses
   - Model comparisons
   - System performance
   - Data import statistics
""")

def main():
    """Main validation function"""
    print("CineSync v2 Setup Validation")
    print("=" * 50)
    
    # Run all validations
    results = {}
    results['wandb_integration'] = validate_wandb_integration()
    results['model_training'] = validate_model_training_scripts()
    results['model_files'] = validate_model_files()
    results['data_pipeline'] = validate_data_pipeline()
    results['requirements'] = validate_requirements()
    results['test_files'] = validate_test_files()
    
    check_wandb_in_requirements()
    
    # Summary
    print("\n=== Validation Summary ===")
    all_passed = all(results.values())
    
    for component, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{component.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall Status: {'✓ ALL CHECKS PASSED' if all_passed else '✗ SOME CHECKS FAILED'}")
    
    if all_passed:
        print("\n🎉 CineSync v2 is ready!")
        print("- All 4 models have W&B integration")
        print("- Data import pipeline is ready")
        print("- All core files are present")
        generate_usage_summary()
    else:
        print("\n⚠️  Some components are missing. Please check the failed items above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())