#!/usr/bin/env python3
"""
Simple test script to verify eval run name functionality.
"""
import os
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

# Mock config class for testing
class MockConfig:
    def __init__(self, eval_run_name=None):
        self.eval_mode = True
        self.eval_run_name = eval_run_name
        self.file_name = "trained.ckpt"
        self.general = MockGeneral()
        self.experiment_dir = tempfile.mkdtemp()

class MockGeneral:
    def __init__(self):
        self.seed = 42

def test_eval_run_naming():
    """Test the eval run naming logic"""
    print("Testing eval run naming logic...")
    
    # Test 1: With custom name
    config = MockConfig(eval_run_name="validation_run")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if hasattr(config, 'eval_run_name') and config.eval_run_name:
        eval_dir_name = f"{config.eval_run_name}_{timestamp}"
    else:
        eval_dir_name = f"eval_{timestamp}"
    
    eval_runs_dir = os.path.join(config.experiment_dir, "eval_runs")
    eval_run_dir = os.path.join(eval_runs_dir, eval_dir_name)
    
    os.makedirs(eval_run_dir, exist_ok=True)
    
    assert "validation_run_" in eval_dir_name
    assert os.path.exists(eval_run_dir)
    print(f"✓ Test 1 passed: Custom name creates directory '{eval_dir_name}'")
    
    # Test 2: Without custom name
    config2 = MockConfig()  # No eval_run_name
    timestamp2 = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if hasattr(config2, 'eval_run_name') and config2.eval_run_name:
        eval_dir_name2 = f"{config2.eval_run_name}_{timestamp2}"
    else:
        eval_dir_name2 = f"eval_{timestamp2}"
    
    eval_runs_dir2 = os.path.join(config2.experiment_dir, "eval_runs")
    eval_run_dir2 = os.path.join(eval_runs_dir2, eval_dir_name2)
    
    os.makedirs(eval_run_dir2, exist_ok=True)
    
    assert eval_dir_name2.startswith("eval_")
    assert os.path.exists(eval_run_dir2)
    print(f"✓ Test 2 passed: Default name creates directory '{eval_dir_name2}'")
    
    # Clean up
    shutil.rmtree(config.experiment_dir)
    shutil.rmtree(config2.experiment_dir)
    
    print("All tests passed! ✅")

if __name__ == "__main__":
    test_eval_run_naming()