#!/usr/bin/env python3
"""
================================================================================
MASTER PIPELINE RUNNER - UCL MATCH OUTCOME PREDICTOR
================================================================================
Purpose: Execute the complete UEFA Champions League match prediction pipeline
         in sequential order without any manual intervention.

Pipeline Steps:
  1. Train baseline models (Logistic Regression + XGBoost)
  2. Generate predictions on UCL matches
  3. Compute model metrics
  4. Generate confusion matrix visualization
  5. Run QA checks on data

Features:
  - One-command execution of entire pipeline
  - Real-time progress tracking with timestamps
  - Automatic time estimation and elapsed time reporting
  - Graceful error handling with detailed error messages
  - Step-by-step status indicators

Usage:
  python run_pipeline.py [--quick]
  
  --quick: Skip non-essential steps (QA, explanations) for faster runtime

Expected Total Runtime: 10-15 minutes

Output:
  Models:  models/logreg_pipeline.pkl, models/xgb_pipeline.pkl
  Metrics: reports/model_metrics.csv, reports/model_report.md
  Visuals: reports/confusion_matrix.png
  Reports: reports/qa_report.md, reports/qa_summary.csv

================================================================================
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime, timedelta


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def print_step(step_num: int, total_steps: int, title: str):
    """Print a step indicator."""
    print(f"\n[STEP {step_num}/{total_steps}] {title}")
    print("-" * 80)


def print_progress(msg: str, level: str = "INFO"):
    """Print timestamped progress message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level:7s}] {msg}")
    sys.stdout.flush()


def run_command(cmd: list, description: str) -> bool:
    """
    Execute a shell command and capture output.
    
    Args:
        cmd (list): Command to execute (as list for subprocess)
        description (str): Human-readable description of the step
        
    Returns:
        bool: True if successful, False otherwise
    """
    print_progress(f"Starting: {description}")
    try:
        start = time.time()
        result = subprocess.run(cmd, capture_output=False, text=True, check=True)
        elapsed = time.time() - start
        print_progress(f"✓ Completed: {description} ({elapsed:.1f}s)")
        return True
    except subprocess.CalledProcessError as e:
        print_progress(f"✗ Failed: {description}", level="ERROR")
        print_progress(f"  Error: {str(e)}", level="ERROR")
        return False
    except Exception as e:
        print_progress(f"✗ Exception: {str(e)}", level="ERROR")
        return False


def verify_data_files() -> bool:
    """Verify that required data files exist."""
    print_progress("Verifying data files...")
    required_files = [
        'data/matches_with_features.csv',
        'data/matches_with_features_ucl_enriched.csv',
        'data/team_features.csv'
    ]
    
    missing = []
    for f in required_files:
        if not os.path.exists(f):
            missing.append(f)
        else:
            print_progress(f"  ✓ {f}")
    
    if missing:
        print_progress(f"✗ Missing required files: {missing}", level="ERROR")
        return False
    
    print_progress("✓ All data files verified")
    return True


def verify_environment() -> bool:
    """Verify Python environment and dependencies."""
    print_progress("Verifying Python environment...")
    required_packages = ['pandas', 'numpy', 'sklearn', 'xgboost', 'matplotlib']
    
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg if pkg != 'sklearn' else 'sklearn')
            print_progress(f"  ✓ {pkg}")
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print_progress(f"✗ Missing packages: {missing}", level="ERROR")
        print_progress("  Install with: pip install -r requirements.txt", level="ERROR")
        return False
    
    print_progress("✓ All dependencies verified")
    return True


def main():
    """Execute complete pipeline."""
    print_header("UCL MATCH OUTCOME PREDICTOR - PIPELINE RUNNER")
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run complete UCL prediction pipeline")
    parser.add_argument('--quick', action='store_true', help='Skip non-essential steps')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training (use existing models)')
    parser.add_argument('--skip-predictions', action='store_true', help='Skip predictions (use existing predictions)')
    args = parser.parse_args()
    
    # Overall timer
    overall_start = time.time()
    
    # Verification phase
    print_header("PHASE 1: PRE-FLIGHT CHECKS")
    
    if not verify_environment():
        print_progress("✗ Environment verification failed", level="ERROR")
        sys.exit(1)
    
    if not verify_data_files():
        print_progress("✗ Data verification failed", level="ERROR")
        sys.exit(1)
    
    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('reports/figs', exist_ok=True)
    print_progress("✓ Output directories ready")
    
    # Define pipeline steps
    steps = []
    if not args.skip_training:
        steps.append((
            "Training Baseline Models (Logistic Regression + XGBoost)",
            ["python", "src/models/train_baseline.py", "--input", "data/matches_with_features.csv"]
        ))
    
    if not args.skip_predictions:
        steps.append((
            "Generating Predictions on UCL Matches (Logistic Regression)",
            ["python", "src/models/predict.py", "--model", "logreg", 
             "--input", "data/matches_with_features_ucl_enriched.csv",
             "--features", "data/model_features.csv",
             "--out", "data/predictions_ucl_logreg.csv"]
        ))
    
    steps.append((
        "Computing Model Metrics",
        ["python", "scripts/model_metrics.py"]
    ))
    
    steps.append((
        "Generating Confusion Matrix Visualization",
        ["python", "scripts/save_confusion_matrix.py"]
    ))
    
    if not args.quick:
        steps.append((
            "Running Data Quality Checks",
            ["python", "scripts/qa_csvs.py"]
        ))
    
    # Execute pipeline
    print_header(f"PHASE 2: PIPELINE EXECUTION ({len(steps)} STEPS)")
    
    failed_steps = []
    for i, (description, cmd) in enumerate(steps, 1):
        print_step(i, len(steps), description)
        
        if not run_command(cmd, description):
            failed_steps.append(description)
        
        # Progress percentage
        progress_pct = (i / len(steps)) * 100
        print_progress(f"Pipeline progress: {progress_pct:.0f}%")
    
    # Results summary
    print_header("PHASE 3: RESULTS SUMMARY")
    
    overall_elapsed = time.time() - overall_start
    overall_str = str(timedelta(seconds=int(overall_elapsed)))
    
    if not failed_steps:
        print_progress("="*80, level="")
        print_progress("✓ PIPELINE EXECUTION SUCCESSFUL", level="SUCCESS")
        print_progress("="*80, level="")
        print_progress(f"Total execution time: {overall_str}", level="")
        print_progress("", level="")
        print_progress("Output files:", level="")
        print_progress("  Models:", level="")
        if os.path.exists('models/logreg_pipeline.pkl'):
            print_progress("    ✓ models/logreg_pipeline.pkl", level="")
        if os.path.exists('models/xgb_pipeline.pkl'):
            print_progress("    ✓ models/xgb_pipeline.pkl", level="")
        print_progress("  Reports:", level="")
        if os.path.exists('reports/model_report.md'):
            print_progress("    ✓ reports/model_report.md", level="")
        if os.path.exists('reports/model_metrics.csv'):
            print_progress("    ✓ reports/model_metrics.csv", level="")
        if os.path.exists('reports/confusion_matrix.png'):
            print_progress("    ✓ reports/confusion_matrix.png", level="")
        if os.path.exists('reports/qa_report.md'):
            print_progress("    ✓ reports/qa_report.md", level="")
        print_progress("", level="")
        return 0
    else:
        print_progress("="*80, level="ERROR")
        print_progress("✗ PIPELINE EXECUTION FAILED", level="ERROR")
        print_progress("="*80, level="ERROR")
        print_progress(f"Failed steps ({len(failed_steps)}):", level="ERROR")
        for step in failed_steps:
            print_progress(f"  - {step}", level="ERROR")
        print_progress(f"Partial execution time: {overall_str}", level="ERROR")
        return 1


if __name__ == '__main__':
    sys.exit(main())
