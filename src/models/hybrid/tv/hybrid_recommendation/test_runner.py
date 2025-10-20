#!/usr/bin/env python3
"""
Simple test runner to check if our test files are syntactically correct
and can be imported without running the full test suite.
"""

import sys
import os
import importlib.util

def check_test_file(file_path):
    """Check if a test file can be imported successfully"""
    try:
        spec = importlib.util.spec_from_file_location("test_module", file_path)
        if spec is None:
            return False, f"Could not load spec for {file_path}"
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True, f"✓ {os.path.basename(file_path)} - Import successful"
    except Exception as e:
        return False, f"✗ {os.path.basename(file_path)} - Import failed: {e}"

def main():
    """Run basic import checks on test files"""
    test_dir = "tests"
    if not os.path.exists(test_dir):
        print(f"Test directory '{test_dir}' not found")
        return 1
    
    test_files = [f for f in os.listdir(test_dir) if f.startswith('test_') and f.endswith('.py')]
    
    if not test_files:
        print("No test files found")
        return 1
    
    print("Checking test file imports...")
    print("-" * 50)
    
    success_count = 0
    total_count = len(test_files)
    
    for test_file in sorted(test_files):
        file_path = os.path.join(test_dir, test_file)
        success, message = check_test_file(file_path)
        print(message)
        if success:
            success_count += 1
    
    print("-" * 50)
    print(f"Results: {success_count}/{total_count} test files passed import check")
    
    if success_count == total_count:
        print("✓ All test files can be imported successfully!")
        print("\nTo run the actual tests, install pytest:")
        print("  python3 -m pip install pytest")
        print("  python3 -m pytest tests/ -v")
        return 0
    else:
        print("✗ Some test files have import issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())