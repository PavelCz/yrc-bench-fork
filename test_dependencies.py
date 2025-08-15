#!/usr/bin/env python3
"""Test script to check if dependencies are available for coinrun analysis."""

import sys
import os

def check_imports():
    """Check if required imports are available."""
    missing = []
    
    try:
        import numpy as np
        print("✓ numpy available")
    except ImportError:
        print("✗ numpy not available")
        missing.append("numpy")
    
    try:
        import torch
        print("✓ torch available")
    except ImportError:
        print("✗ torch not available") 
        missing.append("torch")
    
    try:
        import cv2
        print("✓ opencv-python available")
    except ImportError:
        print("✗ opencv-python not available")
        missing.append("opencv-python")
    
    # Check YRC components
    sys.path.append(".")
    try:
        from YRC.core.configs.global_configs import set_global_variable
        print("✓ YRC core available")
    except ImportError as e:
        print(f"✗ YRC core not available: {e}")
        missing.append("YRC.core")
    
    # Check procgen
    try:
        from lib.procgenAISC.procgen import ProcgenEnv
        print("✓ ProcgenEnv available")
    except ImportError as e:
        print(f"✗ ProcgenEnv not available: {e}")
        missing.append("procgen")
    
    return missing

def check_checkpoints():
    """Check if weak agent checkpoint exists."""
    checkpoint_path = "YRC/checkpoints/procgen/coinrun/weak/model_80019456.pth"
    if os.path.exists(checkpoint_path):
        print(f"✓ Weak agent checkpoint found: {checkpoint_path}")
        return True
    else:
        print(f"✗ Weak agent checkpoint not found: {checkpoint_path}")
        return False

if __name__ == "__main__":
    print("=== Dependency Check ===")
    missing = check_imports()
    
    print("\n=== Checkpoint Check ===")
    checkpoint_exists = check_checkpoints()
    
    print("\n=== Summary ===")
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("To install missing dependencies, you may need to:")
        print("pip install numpy torch opencv-python")
        print("pip install -e lib/procgenAISC")
    else:
        print("All dependencies available!")
    
    if not checkpoint_exists:
        print("Weak agent checkpoint is required for analysis.")
        print("Please ensure the checkpoint file exists or update the path.")
    
    if not missing and checkpoint_exists:
        print("✓ Ready to run coinrun counterfactual analysis!")
        sys.exit(0)
    else:
        print("✗ Setup incomplete")
        sys.exit(1)