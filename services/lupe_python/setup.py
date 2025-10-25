#!/usr/bin/env python3
"""
Setup script for Lupe Discord Bot
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"❌ Python 3.9+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def setup_environment():
    """Set up the virtual environment and install dependencies"""
    print("🚀 Setting up Lupe Discord Bot Environment")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create virtual environment if it doesn't exist
    if not os.path.exists(".venv"):
        if not run_command(f"{sys.executable} -m venv .venv", "Creating virtual environment"):
            return False
    else:
        print("✅ Virtual environment already exists")
    
    # Determine activation script based on OS
    if os.name == 'nt':  # Windows
        activate_script = ".venv\\Scripts\\activate"
        pip_cmd = ".venv\\Scripts\\pip"
        python_cmd = ".venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        activate_script = ".venv/bin/activate"
        pip_cmd = ".venv/bin/pip"
        python_cmd = ".venv/bin/python"
    
    # Install dependencies
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return False
    
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies"):
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Set up your Discord bot token in .env file:")
    print("   DISCORD_TOKEN=your_discord_bot_token_here")
    print("\n2. Ensure PostgreSQL database is running:")
    print("   - Host: localhost:5432")
    print("   - Database: cinesync")
    print("   - User: postgres")
    print("   - Password: (set in .env file)")
    print("\n3. Start the bot:")
    if os.name == 'nt':
        print("   .venv\\Scripts\\python main.py")
    else:
        print("   .venv/bin/python main.py")
    
    return True

def create_env_file():
    """Create a .env file template if it doesn't exist"""
    env_file = Path(".env")
    if not env_file.exists():
        env_content = """# Discord Bot Configuration
DISCORD_TOKEN=your_discord_bot_token_here

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=cinesync
DB_USER=postgres
DB_PASSWORD=your_secure_database_password_here

# Model Configuration
MODELS_DIR=../models
DEVICE=auto

# Debug Mode
DEBUG=true
"""
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ Created .env template file")
        print("⚠️  Please edit .env and add your Discord bot token")
    else:
        print("✅ .env file already exists")

if __name__ == "__main__":
    print("🤖 Lupe Discord Bot Setup")
    print("========================")
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    # Create .env file
    create_env_file()
    
    # Setup environment
    if setup_environment():
        print("\n🎯 Lupe is ready to go!")
    else:
        print("\n💥 Setup failed. Please check the errors above.")
        sys.exit(1)