#!/bin/bash
# QUICKSTART SCRIPT - UEFA Champions League Match Predictor
# ============================================================================
# This script performs all setup steps and runs the complete pipeline
# Usage: bash quickstart.sh
# ============================================================================

set -e  # Exit on error

echo "================================================================================="
echo "  UEFA CHAMPIONS LEAGUE MATCH OUTCOME PREDICTOR - QUICKSTART"
echo "================================================================================="
echo ""

# Step 1: Check if already in venv
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "[STEP 1/5] Setting up Python virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "  ✓ Virtual environment activated"
else
    echo "[STEP 1/5] Virtual environment already active"
fi

# Step 2: Install dependencies
echo ""
echo "[STEP 2/5] Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "  ✓ Dependencies installed"

# Step 3: Verify setup
echo ""
echo "[STEP 3/5] Verifying environment..."
python verify_setup.py || exit 1

# Step 4: Create output directories
echo ""
echo "[STEP 4/5] Creating output directories..."
mkdir -p models reports/figs logs
echo "  ✓ Directories ready"

# Step 5: Run pipeline
echo ""
echo "[STEP 5/5] Running complete pipeline..."
echo ""
python run_pipeline.py "$@"

exit_code=$?

echo ""
echo "================================================================================="
if [ $exit_code -eq 0 ]; then
    echo "  ✓ PIPELINE COMPLETED SUCCESSFULLY"
    echo ""
    echo "  Output files:"
    echo "    Models:  models/logreg_pipeline.pkl, models/xgb_pipeline.pkl"
    echo "    Metrics: reports/model_metrics.csv, reports/model_report.md"
    echo "    Visuals: reports/confusion_matrix.png"
    echo "    Reports: reports/qa_report.md, reports/qa_summary.csv"
else
    echo "  ✗ PIPELINE FAILED (exit code: $exit_code)"
    echo ""
    echo "  Troubleshooting:"
    echo "    1. Check Python version: python --version"
    echo "    2. Verify dependencies: pip list | grep -E 'pandas|sklearn|xgboost'"
    echo "    3. Check data files: ls -la data/*.csv"
    echo "    4. Run verification: python verify_setup.py"
fi
echo "================================================================================="

exit $exit_code
