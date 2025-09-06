#!/usr/bin/env python3
"""
Setup script for AI-powered traffic light system
Automatically downloads models and sets up the ML environment
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run shell command with error handling"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ Error in {description}: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Exception in {description}: {e}")
        return False

def setup_environment():
    """Setup the AI environment"""
    print("🚀 Setting up AI-Powered Traffic Light System")
    print("=" * 50)

    # Check Python version
    import sys
    if sys.version_info >= (3, 13):
        print("⚠️  Python 3.13 detected - some packages may have compatibility issues")
        print("💡 Consider using Python 3.10-3.12 for better compatibility")
        return False

    # Install dependencies with fallback
    success = False

    # First try the full requirements
    if run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        success = True
    else:
        print("🔄 Trying alternative installation method...")
        # Try installing core packages first
        if run_command("pip install opencv-python numpy pillow python-dotenv requests", "Installing core dependencies"):
            print("📦 Core dependencies installed - some advanced features may be limited")
            success = True
        else:
            print("❌ Failed to install core dependencies")
            return False

    # Create necessary directories
    Path("src/models").mkdir(exist_ok=True)
    Path("src/Data/images").mkdir(parents=True, exist_ok=True)

    # Download ML models
    if run_command("python src/models/download_models.py", "Downloading AI models"):
        print("🤖 AI models downloaded successfully")
    else:
        print("⚠️  Model download failed - will use CV fallback")

    print("\n🎉 AI Setup Complete!")
    print("\nTo test the system:")
    print("1. python main.py")
    print("2. Or use python -m src.training.capture_training_data --mode interactive")
    print("\nThe system will automatically use AI when models are available!")

    return True

if __name__ == "__main__":
    success = setup_environment()
    sys.exit(0 if success else 1)