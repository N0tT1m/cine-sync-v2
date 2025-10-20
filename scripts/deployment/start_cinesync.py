#!/usr/bin/env python3
"""
CineSync v2 - Complete Startup Script
Initializes all 6 AI models + 2 hybrid recommenders and launches admin interface
"""

import os
import sys
import asyncio
import logging
import signal
import subprocess
from pathlib import Path
import time

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from unified_model_manager import model_manager, initialize_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cinesync_startup.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("CineSyncStartup")

class CineSyncLauncher:
    """Complete launcher for CineSync v2 with all AI models"""
    
    def __init__(self):
        self.processes = []
        self.models_loaded = False
        
    async def initialize_system(self):
        """Initialize the complete CineSync system"""
        
        logger.info("🚀 Starting CineSync v2 - AI Recommendation System")
        logger.info("=" * 60)
        
        # Step 1: Check dependencies
        logger.info("📋 Checking dependencies...")
        if not self.check_dependencies():
            logger.error("❌ Dependency check failed!")
            return False
            
        # Step 2: Setup directories
        logger.info("📁 Setting up directories...")
        self.setup_directories()
        
        # Step 3: Initialize models
        logger.info("🤖 Initializing AI models...")
        try:
            model_status = await initialize_models()
            self.models_loaded = True
            
            loaded_count = sum(model_status.values())
            total_count = len(model_status)
            
            logger.info(f"✅ Loaded {loaded_count}/{total_count} models successfully")
            
            if loaded_count == 0:
                logger.warning("⚠️  No models loaded successfully - check model files")
            
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            self.models_loaded = False
            
        # Step 4: Launch admin interface
        logger.info("🌐 Starting admin interface...")
        self.start_admin_interface()
        
        logger.info("=" * 60)
        logger.info("🎉 CineSync v2 startup complete!")
        logger.info("📊 Admin Dashboard: http://localhost:5001")
        logger.info("🔑 Default login: admin / admin123")
        logger.info("=" * 60)
        
        return True
        
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available"""
        
        required_packages = [
            'torch', 'numpy', 'pandas', 'flask', 'plotly'
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"  ✅ {package}")
            except ImportError:
                missing.append(package)
                logger.error(f"  ❌ {package}")
                
        if missing:
            logger.error(f"Missing packages: {', '.join(missing)}")
            logger.error("Install with: pip install torch numpy pandas flask plotly")
            return False
            
        return True
        
    def setup_directories(self):
        """Setup required directories"""
        
        directories = [
            'models',
            'models/weights', 
            'models/configs',
            'templates',
            'static',
            'logs'
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"  📁 {dir_path}")
            
    def start_admin_interface(self):
        """Start the Flask admin interface"""
        
        try:
            # Set environment variables
            os.environ.setdefault('ADMIN_SECRET_KEY', 'cinesync-v2-admin-key')
            os.environ.setdefault('ADMIN_USERS', 'admin')
            os.environ.setdefault('ADMIN_PASSWORD', 'admin123')
            
            # Start admin interface in background
            admin_process = subprocess.Popen([
                sys.executable, 'admin_interface.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes.append(admin_process)
            logger.info("  🌐 Admin interface started (PID: {})".format(admin_process.pid))
            
        except Exception as e:
            logger.error(f"Failed to start admin interface: {e}")
            
    def create_sample_config(self):
        """Create sample configuration files"""
        
        # Create model config
        model_config = {
            "models": {
                "bert4rec": {
                    "enabled": True,
                    "priority": 5,
                    "content_type": "both"
                },
                "sentence_bert_two_tower": {
                    "enabled": True,
                    "priority": 4,
                    "content_type": "both"
                },
                "graphsage": {
                    "enabled": True,
                    "priority": 3,
                    "content_type": "both"
                }
            },
            "training": {
                "auto_retrain": True,
                "min_feedback_threshold": 100,
                "excluded_genres": [],
                "quality_filters": ["4K", "2K", "1080p"]
            }
        }
        
        import json
        with open('cinesync_config.json', 'w') as f:
            json.dump(model_config, f, indent=2)
            
        logger.info("📋 Created sample configuration")
        
    def show_status(self):
        """Show current system status"""
        
        print("\n" + "=" * 60)
        print("🎬 CineSync v2 - System Status")
        print("=" * 60)
        
        # Model status
        if self.models_loaded:
            model_status = model_manager.get_model_status()
            print(f"🤖 Models: {model_status['loaded_count']}/{model_status['total_models']} loaded")
            
            for model_name, is_loaded in model_status['loaded_models'].items():
                status = "✅" if is_loaded else "❌"
                print(f"   {status} {model_name.replace('_', ' ').title()}")
        else:
            print("🤖 Models: Not initialized")
            
        # Service status
        print(f"\n🌐 Services:")
        for i, process in enumerate(self.processes):
            if process.poll() is None:
                print(f"   ✅ Admin Interface (PID: {process.pid})")
            else:
                print(f"   ❌ Admin Interface (Stopped)")
                
        print(f"\n📊 Admin Dashboard: http://localhost:5001")
        print(f"🔑 Login: admin / admin123")
        print("=" * 60 + "\n")
        
    def shutdown(self):
        """Gracefully shutdown all services"""
        
        logger.info("🛑 Shutting down CineSync v2...")
        
        for process in self.processes:
            if process.poll() is None:
                logger.info(f"Stopping process {process.pid}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                    logger.info(f"Process {process.pid} stopped gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing process {process.pid}")
                    process.kill()
                    
        logger.info("✅ Shutdown complete")
        
    def run_interactive(self):
        """Run in interactive mode with status updates"""
        
        try:
            while True:
                command = input("\nCineSync> ").strip().lower()
                
                if command in ['quit', 'exit', 'q']:
                    break
                elif command in ['status', 's']:
                    self.show_status()
                elif command in ['help', 'h']:
                    self.show_help()
                elif command == 'models':
                    self.show_model_details()
                elif command == 'logs':
                    self.show_recent_logs()
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()
            
    def show_help(self):
        """Show available commands"""
        
        print("\n📋 Available Commands:")
        print("  status, s     - Show system status")
        print("  models        - Show detailed model information")  
        print("  logs          - Show recent logs")
        print("  help, h       - Show this help")
        print("  quit, exit, q - Shutdown CineSync")
        print("\n🌐 Web Interface: http://localhost:5001")
        
    def show_model_details(self):
        """Show detailed model information"""
        
        if not self.models_loaded:
            print("❌ Models not loaded yet")
            return
            
        model_status = model_manager.get_model_status()
        
        print("\n🤖 Model Details:")
        print("-" * 40)
        
        for model_name, config in model_manager.model_configs.items():
            is_loaded = model_status['loaded_models'].get(model_name, False)
            status = "✅ Online" if is_loaded else "❌ Offline"
            
            print(f"\n{config.name}")
            print(f"  Status: {status}")
            print(f"  Type: {config.content_type}")
            print(f"  Priority: {config.priority}")
            print(f"  Enabled: {'Yes' if config.enabled else 'No'}")
            
            if model_name in model_status['model_errors']:
                print(f"  Error: {model_status['model_errors'][model_name]}")
                
    def show_recent_logs(self):
        """Show recent log entries"""
        
        try:
            with open('cinesync_startup.log', 'r') as f:
                lines = f.readlines()
                
            print("\n📋 Recent Logs (last 10 lines):")
            print("-" * 40)
            for line in lines[-10:]:
                print(line.rstrip())
                
        except FileNotFoundError:
            print("❌ No log file found")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal")
    if hasattr(signal_handler, 'launcher'):
        signal_handler.launcher.shutdown()
    sys.exit(0)

async def main():
    """Main entry point"""
    
    launcher = CineSyncLauncher()
    signal_handler.launcher = launcher
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize system
    success = await launcher.initialize_system()
    
    if not success:
        logger.error("❌ Failed to initialize CineSync v2")
        return 1
        
    # Check if running in interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == '--daemon':
        logger.info("Running in daemon mode...")
        try:
            # Keep running until interrupted
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        # Run interactive mode
        launcher.show_status()
        launcher.run_interactive()
        
    return 0

if __name__ == "__main__":
    # Check if we need to create sample config
    if not os.path.exists('cinesync_config.json'):
        launcher = CineSyncLauncher()
        launcher.create_sample_config()
        
    # Run the main launcher
    exit_code = asyncio.run(main())
    sys.exit(exit_code)