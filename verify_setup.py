#!/usr/bin/env python3
"""
================================================================================
SETUP VERIFICATION SCRIPT
================================================================================
Purpose: Verify that the environment and data are set up correctly for pipeline execution.

Checks:
  - Python version
  - Required dependencies installed
  - Data files present
  - Output directories writable

Usage:
  python verify_setup.py
================================================================================
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def check_python_version():
    """Verify Python version >= 3.10."""
    print("[CHECKING] Python Version")
    version = sys.version_info
    print(f"  Current: Python {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 10:
        print("  ✓ PASS: Python 3.10+ detected")
        return True
    else:
        print("  ✗ FAIL: Requires Python 3.10+")
        return False


def check_dependencies():
    """Verify required packages are installed."""
    print("\n[CHECKING] Python Dependencies")
    required = {
        'pandas': 'Data processing',
        'numpy': 'Numerical operations',
        'sklearn': 'Machine learning (scikit-learn)',
        'xgboost': 'Gradient boosting',
        'matplotlib': 'Plotting',
        'joblib': 'Model serialization'
    }
    
    all_found = True
    for pkg, purpose in required.items():
        try:
            __import__(pkg)
            print(f"  ✓ {pkg:15s} - {purpose}")
        except ImportError:
            print(f"  ✗ {pkg:15s} - {purpose} (NOT FOUND)")
            all_found = False
    
    if not all_found:
        print("\n  Install missing packages:")
        print("    pip install -r requirements.txt")
    
    return all_found


def check_data_files():
    """Verify required data files exist."""
    print("\n[CHECKING] Data Files")
    required_files = [
        'data/matches_with_features.csv',
        'data/matches_with_features_ucl_enriched.csv',
        'data/team_features.csv'
    ]
    
    all_found = True
    for f in required_files:
        if os.path.exists(f):
            size_mb = os.path.getsize(f) / (1024*1024)
            print(f"  ✓ {f:45s} ({size_mb:6.1f} MB)")
        else:
            print(f"  ✗ {f:45s} (NOT FOUND)")
            all_found = False
    
    return all_found


def check_output_directories():
    """Verify output directories are writable."""
    print("\n[CHECKING] Output Directories")
    dirs = ['models', 'reports', 'logs']
    
    all_writable = True
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        if os.path.isdir(d) and os.access(d, os.W_OK):
            print(f"  ✓ {d:20s} - writable")
        else:
            print(f"  ✗ {d:20s} - not writable")
            all_writable = False
    
    return all_writable


def check_venv():
    """Check if running in a virtual environment."""
    print("\n[CHECKING] Virtual Environment")
    in_venv = (hasattr(sys, 'real_prefix') or 
               (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
    
    if in_venv:
        print(f"  ✓ Running in virtual environment: {sys.prefix}")
        return True
    else:
        print("  ⚠ WARNING: Not running in virtual environment")
        print("    Recommended: source .venv/bin/activate")
        return True  # Not fatal, just a warning


def main():
    """Run all checks."""
    print_header("ENVIRONMENT VERIFICATION")
    
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_venv),
        ("Dependencies", check_dependencies),
        ("Data Files", check_data_files),
        ("Output Directories", check_output_directories)
    ]
    
    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"  ✗ Exception during check: {str(e)}")
            results[name] = False
    
    # Summary
    print_header("SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:7s} - {name}")
    
    print(f"\n  Result: {passed}/{total} checks passed\n")
    
    if passed == total:
        print("  ✓ Environment is ready! Run: python run_pipeline.py")
        return 0
    else:
        print("  ✗ Please fix the issues above and try again")
        return 1


if __name__ == '__main__':
    sys.exit(main())
