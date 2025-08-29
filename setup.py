#!/usr/bin/env python3
"""
Setup script for CFINet SKU-110K training environment
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8 or higher is required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n=== Installing Dependencies ===")
    
    # Upgrade pip
    if not run_command("python -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install PyTorch (with CUDA support if available)
    try:
        import torch
        if torch.cuda.is_available():
            print("✓ CUDA is available - PyTorch with CUDA support will be installed")
            if not run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118", "Installing PyTorch with CUDA"):
                return False
        else:
            print("✓ CUDA not available - Installing CPU-only PyTorch")
            if not run_command("pip install torch torchvision torchaudio", "Installing PyTorch (CPU)"):
                return False
    except ImportError:
        print("Installing PyTorch...")
        if not run_command("pip install torch torchvision torchaudio", "Installing PyTorch"):
            return False
    
    # Install other dependencies
    if not run_command("pip install -r requirements.txt", "Installing other dependencies"):
        return False
    
    return True

def verify_installation():
    """Verify that all dependencies are installed correctly"""
    print("\n=== Verifying Installation ===")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} installed")
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available - training will be slower on CPU")
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__} installed")
    except ImportError:
        print("✗ OpenCV not installed")
        return False
    
    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__} installed")
    except ImportError:
        print("✗ NumPy not installed")
        return False
    
    try:
        import tqdm
        print(f"✓ tqdm installed")
    except ImportError:
        print("✗ tqdm not installed")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print("\n=== Creating Directories ===")
    
    directories = ['outputs', 'inference_results', 'logs']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory already exists: {directory}")
    
    return True

def main():
    """Main setup function"""
    print("=== CFINet SKU-110K Setup ===")
    print("This script will set up the environment for training CFINet on SKU-110K dataset")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n✗ Setup failed during dependency installation")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n✗ Setup failed during verification")
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("\n✗ Setup failed during directory creation")
        sys.exit(1)
    
    print("\n=== Setup Completed Successfully! ===")
    print("\nNext steps:")
    print("1. Download the SKU-110K dataset")
    print("2. Organize the dataset according to the README.md structure")
    print("3. Run training with: python train_sku110k.py --data_root /path/to/sku110k")
    print("\nFor detailed instructions, see README.md")

if __name__ == "__main__":
    main() 