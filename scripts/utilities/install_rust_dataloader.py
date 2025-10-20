#!/usr/bin/env python3
"""
Installation script for CineSync Rust DataLoader
Automatically installs Rust, Maturin, and builds the high-performance data loading extension
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def run_command(command, cwd=None, check=True):
    """Run a command and return the result."""
    try:
        print(f"Running: {command}")
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd=cwd,
            check=check
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"Error: {result.stderr}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return e

def check_rust_installed():
    """Check if Rust is installed."""
    try:
        result = run_command("rustc --version", check=False)
        if result.returncode == 0:
            print("✅ Rust is already installed")
            print(result.stdout.strip())
            return True
        else:
            print("❌ Rust not found")
            return False
    except:
        print("❌ Rust not found")
        return False

def install_rust():
    """Install Rust using rustup."""
    print("📦 Installing Rust...")
    
    system = platform.system().lower()
    
    if system == "windows":
        print("Please install Rust manually from https://rustup.rs/")
        print("Then run this script again.")
        return False
    else:
        # Unix-like systems (macOS, Linux)
        command = 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y'
        result = run_command(command, check=False)
        
        if result.returncode == 0:
            # Source the cargo environment
            cargo_env = os.path.expanduser("~/.cargo/env")
            if os.path.exists(cargo_env):
                run_command(f"source {cargo_env}", check=False)
            
            # Add cargo to PATH for this session
            cargo_bin = os.path.expanduser("~/.cargo/bin")
            if cargo_bin not in os.environ["PATH"]:
                os.environ["PATH"] = f"{cargo_bin}:{os.environ['PATH']}"
            
            print("✅ Rust installed successfully")
            return True
        else:
            print("❌ Failed to install Rust")
            return False

def check_maturin_installed():
    """Check if Maturin is installed."""
    try:
        result = run_command("maturin --version", check=False)
        if result.returncode == 0:
            print("✅ Maturin is already installed")
            print(result.stdout.strip())
            return True
        else:
            print("❌ Maturin not found")
            return False
    except:
        print("❌ Maturin not found")
        return False

def install_maturin():
    """Install Maturin for building Python extensions."""
    print("📦 Installing Maturin...")
    
    # Try pip install first
    result = run_command("pip install maturin", check=False)
    if result.returncode == 0:
        print("✅ Maturin installed via pip")
        return True
    
    # Fallback to cargo install
    result = run_command("cargo install maturin", check=False)
    if result.returncode == 0:
        print("✅ Maturin installed via cargo")
        return True
    else:
        print("❌ Failed to install Maturin")
        return False

def build_rust_extension():
    """Build the Rust extension using Maturin."""
    print("🔨 Building CineSync Rust DataLoader...")
    
    rust_dir = Path(__file__).parent / "rust_dataloader"
    
    if not rust_dir.exists():
        print(f"❌ Rust dataloader directory not found: {rust_dir}")
        return False
    
    # Change to rust directory
    original_dir = os.getcwd()
    os.chdir(rust_dir)
    
    try:
        # Build in development mode first
        print("Building in development mode...")
        result = run_command("maturin develop --release", check=False)
        
        if result.returncode == 0:
            print("✅ Rust extension built successfully!")
            
            # Test the import
            print("🧪 Testing import...")
            test_result = run_command("python -c \"import cine_sync_dataloader; print('Import successful!')\"", check=False)
            
            if test_result.returncode == 0:
                print("✅ Rust dataloader import test passed!")
                return True
            else:
                print("❌ Import test failed")
                return False
        else:
            print("❌ Failed to build Rust extension")
            return False
            
    finally:
        os.chdir(original_dir)

def test_dataloader():
    """Test the Rust dataloader functionality."""
    print("🧪 Testing CineSync Rust DataLoader functionality...")
    
    test_code = '''
import cine_sync_dataloader
import tempfile
import csv
import os

# Create a temporary CSV file for testing
with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'movie_id', 'rating', 'timestamp', 'title', 'genres', 'year'])
    
    # Write some test data
    for i in range(1000):
        writer.writerow([
            i % 100,  # user_id
            i % 500,  # movie_id
            3.5 + (i % 5) * 0.5,  # rating
            1234567890 + i,  # timestamp
            f"Movie {i}",  # title
            "Action|Drama",  # genres
            2000 + (i % 20)  # year
        ])
    
    temp_file = f.name

try:
    # Test the loader
    loader = cine_sync_dataloader.CineSyncDataLoader(batch_size=32, shuffle=True, buffer_size=64)
    
    # Load test data
    count = loader.load_movies_csv(temp_file)
    print(f"Loaded {count} movie ratings")
    
    # Test batch creation
    batches = loader.create_batches()
    print(f"Created {len(batches)} batches")
    
    # Test performance stats
    stats = loader.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    # Test data summary
    summary = loader.get_data_summary()
    print(f"Data summary: {summary}")
    
    print("✅ All tests passed!")
    
finally:
    # Clean up
    os.unlink(temp_file)
'''
    
    result = run_command(f"python -c \"{test_code}\"", check=False)
    
    if result.returncode == 0:
        print("✅ CineSync Rust DataLoader functionality test passed!")
        return True
    else:
        print("❌ Functionality test failed")
        return False

def install_python_dependencies():
    """Install required Python dependencies."""
    print("📦 Installing Python dependencies...")
    
    dependencies = [
        "numpy",
        "pandas", 
        "torch",
        "requests",
        "psutil",
        "gputil",
    ]
    
    for dep in dependencies:
        print(f"Installing {dep}...")
        result = run_command(f"pip install {dep}", check=False)
        if result.returncode != 0:
            print(f"⚠️  Warning: Failed to install {dep}")

def setup_environment():
    """Setup environment variables and configuration."""
    print("⚙️  Setting up environment...")
    
    # Create a simple config file
    config = {
        "rust_dataloader": {
            "enabled": True,
            "default_batch_size": 256,
            "default_buffer_size": 1024,
            "num_threads": 8
        },
        "training": {
            "mixed_precision": True,
            "compile_model": True,
            "gradient_checkpointing": True
        }
    }
    
    import json
    config_path = Path(__file__).parent / "dataloader_config.json"
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration saved to {config_path}")

def main():
    """Main installation function."""
    print("🚀 CineSync Rust DataLoader Installation")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Install Python dependencies first
    install_python_dependencies()
    
    # Check and install Rust
    if not check_rust_installed():
        if not install_rust():
            print("❌ Failed to install Rust. Please install manually from https://rustup.rs/")
            sys.exit(1)
    
    # Check and install Maturin
    if not check_maturin_installed():
        if not install_maturin():
            print("❌ Failed to install Maturin")
            sys.exit(1)
    
    # Build the Rust extension
    if not build_rust_extension():
        print("❌ Failed to build Rust extension")
        print("\n🔄 Falling back to Python dataloader")
        print("The system will use the optimized Python backend instead.")
        return
    
    # Test the dataloader
    if not test_dataloader():
        print("⚠️  Rust dataloader built but tests failed")
        print("The system will fall back to Python dataloader if needed.")
    
    # Setup environment
    setup_environment()
    
    print("\n" + "=" * 50)
    print("🎉 Installation completed successfully!")
    print("\nYou can now use the CineSync optimized training pipeline:")
    print("python optimized_training_with_rust.py --movies-path your_movies.csv --tv-path your_tv.csv")
    print("\nFor Discord notifications, add: --discord-webhook YOUR_WEBHOOK_URL")
    print("For SMS alerts, add: --alert-phone +1234567890")

if __name__ == "__main__":
    main()