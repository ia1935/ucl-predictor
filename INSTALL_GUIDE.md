# SUBMISSION GUIDE - UEFA Champions League Match Outcome Predictor

## Project Overview

This is a complete machine learning pipeline for predicting UEFA Champions League match outcomes using team statistics. The project includes training, inference, evaluation, and reporting components with full progress tracking and comprehensive documentation.

---

## What's Included

### Code Files
```
✓ run_pipeline.py              Master runner for sequential execution
✓ src/models/train_baseline.py   Model training (Logistic Regression, XGBoost)
✓ src/models/predict.py          Inference on UCL matches
✓ scripts/model_metrics.py        Compute accuracy and F1 scores
✓ scripts/save_confusion_matrix.py Visualization
✓ scripts/qa_csvs.py             Data quality checks
✓ scripts/explain_predictions.py  Prediction explanation
✓ verify_setup.py                Environment verification
✓ quickstart.sh                  Automated setup and launch
```

### Documentation
```
✓ README.md                    Comprehensive guide (2000+ words)
✓ SUBMISSION_CHECKLIST.md      Requirements verification
✓ requirements.txt             Dependencies with versions
✓ INSTALL_GUIDE.md             This file
```

### Data Files (Included for Testing)
```
✓ data/matches_with_features.csv              Training data
✓ data/matches_with_features_ucl_enriched.csv  Prediction data
✓ data/team_features.csv                       Team statistics
```

---

## Quick Start (Recommended)

### Fastest Method (Automated)
```bash
cd /path/to/UclPredictor
bash quickstart.sh
```

This automatically:
1. Creates Python virtual environment
2. Installs all dependencies
3. Verifies setup
4. Runs complete pipeline
5. Produces all outputs

**Expected time**: ~15 minutes

---

## Manual Setup (Alternative)

### Step 1: Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate    # On macOS/Linux
# (Windows: .venv\Scripts\activate)
```

### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
python verify_setup.py
```

Expected output: `Result: 5/5 checks passed`

### Step 4: Run Pipeline
```bash
python run_pipeline.py
```

Or with options:
```bash
python run_pipeline.py --quick              # Skip QA checks
python run_pipeline.py --skip-training      # Use existing models
```

---

## What the Pipeline Does

### Step 1: Train Models (2-5 minutes)
- Loads match data with team statistics
- Builds delta features (home advantage metrics)
- Trains Logistic Regression and XGBoost
- Saves pipelines to `models/`
- Generates training report

### Step 2: Generate Predictions (1-2 minutes)
- Loads trained model
- Applies feature engineering to UCL matches
- Generates win/draw/loss probabilities
- Saves predictions to `data/predictions_ucl_logreg.csv`

### Step 3: Compute Metrics (1 minute)
- Calculates accuracy and F1 scores
- Saves to `reports/model_metrics.csv`

### Step 4: Visualize Confusion Matrix (1 minute)
- Creates 3x3 prediction confusion matrix heatmap
- Saves to `reports/confusion_matrix.png`

### Step 5: Data Quality Checks (2 minutes, optional)
- Analyzes missing values
- Generates summary report
- Creates null value visualizations

---

## Output Files

After execution, you'll find:

### Models
```
models/
├── logreg_pipeline.pkl         Binary serialized Logistic Regression model
└── xgb_pipeline.pkl            Binary serialized XGBoost model
```

### Reports
```
reports/
├── model_report.md             Training metrics and summary
├── model_metrics.csv           Accuracy and F1 scores per model
├── confusion_matrix.png        3x3 confusion matrix heatmap
├── qa_report.md                Data quality analysis (optional)
├── qa_summary.csv              QA metrics (optional)
└── figs/                       Null value distribution plots (optional)
```

### Predictions
```
data/
├── model_features.csv          Features used in training
├── predictions_ucl_logreg.csv  Match predictions with probabilities
└── [input data files]
```

---

## System Requirements

- **Python**: 3.10 or higher
- **OS**: Linux, macOS, or Unix-like system
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Disk**: 500MB for data + models
- **Shell**: bash or zsh

---

## Real-Time Progress Example

```
[2025-12-04 10:30:45] [INFO   ] ================================================================================
[2025-12-04 10:30:45] [INFO   ]   UCL MATCH OUTCOME PREDICTOR - PIPELINE RUNNER
[2025-12-04 10:30:45] [INFO   ] ================================================================================

[STEP 1/5] Train Baseline Models (Logistic Regression + XGBoost)
[2025-12-04 10:30:50] [INFO   ] Starting: Training Baseline Models...
[2025-12-04 10:30:51] [INFO   ] Loading data from data/matches_with_features.csv
[2025-12-04 10:30:51] [INFO   ] Loaded 2000 rows, 25 columns
[2025-12-04 10:30:52] [INFO   ] Building delta features...
[2025-12-04 10:35:20] [INFO   ] ✓ Completed: Training Baseline Models (284.5s)

[STEP 2/5] Generate Predictions on UCL Matches (Logistic Regression)
...

[2025-12-04 10:45:20] [SUCCESS] ✓ PIPELINE EXECUTION SUCCESSFUL
[2025-12-04 10:45:20] [INFO   ] Total execution time: 0:14:35
```

---

## Troubleshooting

### "Python version not found"
```bash
python3 --version  # Should be 3.10+
# If not, install Python 3.10+ from python.org or use system package manager
```

### "Module not found" error
```bash
pip install -r requirements.txt
# Ensure virtual environment is activated
source .venv/bin/activate
```

### "Data files not found"
```bash
ls -la data/*.csv  # Verify data files exist
```

### "Permission denied" for scripts
```bash
chmod +x quickstart.sh verify_setup.py run_pipeline.py
```

### Pipeline runs but produces no output
```bash
mkdir -p models reports/figs logs
python run_pipeline.py --quick  # Try quick mode first
```

---

## Code Quality Features

### Progress Tracking
- ✓ Timestamps on every log message
- ✓ Step completion indicators
- ✓ Percentage progress through pipeline
- ✓ Elapsed time per step
- ✓ Total execution time

### Documentation
- ✓ File-level docstrings (purpose, I/O, runtime)
- ✓ Function docstrings (args, returns, purpose)
- ✓ Inline comments for complex logic
- ✓ 2000+ word README with examples
- ✓ Comprehensive API documentation

### Error Handling
- ✓ Try/except blocks with meaningful messages
- ✓ File existence checks before operations
- ✓ Dependency verification before execution
- ✓ Graceful failure with actionable guidance

---

## For Grading

### Suggested Workflow

1. **Extract and navigate**:
   ```bash
   unzip/tar -xf project.zip
   cd UclPredictor
   ```

2. **Run verification**:
   ```bash
   python verify_setup.py
   ```

3. **Run pipeline**:
   ```bash
   python run_pipeline.py
   ```
   Or automated: `bash quickstart.sh`

4. **Review outputs**:
   ```bash
   cat reports/model_report.md           # Training results
   cat reports/model_metrics.csv         # Accuracy/F1 scores
   open reports/confusion_matrix.png     # Visualization
   ```

5. **Verify reproducibility**:
   - All steps executed automatically
   - Progress messages displayed in real-time
   - Results generated in expected locations
   - Reports are complete and readable

### Success Indicators

✓ Program runs without errors  
✓ Models trained successfully  
✓ Predictions generated  
✓ Metrics computed  
✓ Confusion matrix visualized  
✓ Reports generated  
✓ Real-time progress displayed  
✓ Total time ~10-15 minutes  

---

## Time Estimates

| Component | Duration | Notes |
|-----------|----------|-------|
| Setup (venv + pip) | 5 min | One-time only |
| Verification | 1 min | Checks environment |
| Model Training | 2-5 min | Depends on data size |
| Predictions | 1-2 min | Batch inference |
| Metrics | 1 min | Evaluation computation |
| Visualizations | 1 min | Plot generation |
| QA Checks | 2 min | Optional |
| **TOTAL PIPELINE** | **10-15 min** | Full end-to-end |

---

## Project Structure

```
UclPredictor/
├── run_pipeline.py                  Main entry point
├── verify_setup.py                  Setup verification
├── quickstart.sh                    Automated setup
├── requirements.txt                 Dependencies
├── README.md                        Comprehensive guide
├── SUBMISSION_CHECKLIST.md          Requirements verification
│
├── src/
│   ├── models/
│   │   ├── train_baseline.py       Model training
│   │   └── predict.py              Inference
│   ├── features/
│   │   ├── engineering.py          Feature engineering
│   │   └── aggregate_features.py   Feature aggregation
│   └── utils/
│
├── scripts/
│   ├── model_metrics.py            Metrics computation
│   ├── save_confusion_matrix.py    Visualization
│   ├── qa_csvs.py                  QA checks
│   ├── explain_predictions.py      Explanation
│   └── merge_ucl_matches.py        Data preprocessing
│
├── data/                           Data directory
│   ├── matches_with_features.csv
│   ├── matches_with_features_ucl_enriched.csv
│   └── team_features.csv
│
├── models/                         Output models (created)
│   ├── logreg_pipeline.pkl
│   └── xgb_pipeline.pkl
│
├── reports/                        Output reports (created)
│   ├── model_report.md
│   ├── model_metrics.csv
│   ├── confusion_matrix.png
│   └── figs/
│
└── logs/                          Log files (created)
```

---

## Contact & Support

For questions about the code, refer to:
- Docstrings in each function
- Inline comments in complex sections
- README.md for architecture overview
- SUBMISSION_CHECKLIST.md for requirements

---

**Ready to grade?** Start with: `bash quickstart.sh`

Expected completion: ~15 minutes  
Result: All outputs in `reports/` and `models/` directories
