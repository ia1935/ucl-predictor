# UEFA Champions League Match Outcome Predictor

A lightweight, end-to-end machine learning pipeline for predicting UEFA Champions League match outcomes using team statistics and historical form. Includes model training, inference, and comprehensive evaluation tools.

## 📋 Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Full Pipeline Execution](#full-pipeline-execution)
- [Individual Components](#individual-components)
- [Data Specification](#data-specification)
- [Output Files](#output-files)
- [Pipeline Architecture](#pipeline-architecture)
- [Dependencies](#dependencies)

---

## Overview

This project trains machine learning models to predict UEFA Champions League match outcomes (Home Win, Draw, Away Win) based on team features. The pipeline includes:

- **Feature Engineering**: Delta-based features (home_stat - away_stat) capturing team advantages
- **Model Training**: Logistic Regression and XGBoost with time-aware train/test split
- **Inference**: Batch predictions on UCL matches with probability estimates
- **Evaluation**: Metrics computation, confusion matrices, and data quality checks
- **Monitoring**: Real-time progress tracking, time estimation, and detailed logging

**Estimated Total Runtime**: 10-15 minutes (full pipeline)

---

## System Requirements

- **Python**: 3.10 or higher
- **Operating System**: Unix-like (Linux, macOS) with bash/zsh
- **Disk Space**: ~500MB (data + models)
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Shell**: bash or zsh (not cmd/PowerShell)

---

## Installation

### Step 1: Clone or Navigate to Repository
```bash
cd /path/to/UclPredictor
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# (On Windows use: .venv\Scripts\activate)
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import pandas, numpy, sklearn, xgboost; print('✓ All dependencies installed')"
```

---

## Quick Start

### Run Complete Pipeline (Recommended)
Execute the entire pipeline in one command with automatic progress tracking:

```bash
# Full pipeline (training + inference + evaluation)
python run_pipeline.py

# Quick mode (skip optional QA checks)
python run_pipeline.py --quick

# Resume from predictions (skip training)
python run_pipeline.py --skip-training
```

This will:
1. Train Logistic Regression and XGBoost models
2. Generate predictions on UCL matches
3. Compute metrics (accuracy, F1 score)
4. Create confusion matrix visualization
5. Run data quality checks (optional)
6. Output timing and results summary

**Expected Output**:
```
[2025-12-04 10:30:45] [INFO   ] ================================================================================
[2025-12-04 10:30:45] [INFO   ]   UCL MATCH OUTCOME PREDICTOR - PIPELINE RUNNER
[2025-12-04 10:30:45] [INFO   ] ================================================================================
...
[2025-12-04 10:45:20] [SUCCESS] ✓ PIPELINE EXECUTION SUCCESSFUL
[2025-12-04 10:45:20] [INFO   ] Total execution time: 0:14:35
```

---

## Full Pipeline Execution

### Detailed Step-by-Step Workflow

If you prefer to run individual components manually:

#### Step 1: Train Baseline Models (~2-5 minutes)
```bash
python src/models/train_baseline.py \
  --input data/matches_with_features.csv

# Outputs:
#   - models/logreg_pipeline.pkl      (Logistic Regression model)
#   - models/xgb_pipeline.pkl         (XGBoost model)
#   - data/model_features.csv         (Feature list for inference)
#   - reports/model_report.md         (Training report)
```

**What it does**:
- Loads match data with team statistics
- Builds delta features (home_stat - away_stat)
- Performs time-aware train/test split
- Trains two models: Logistic Regression and XGBoost
- Evaluates both models on held-out test set
- Saves trained pipelines (model + imputer + scaler)

#### Step 2: Generate Predictions (~1-2 minutes)
```bash
python src/models/predict.py \
  --model logreg \
  --input data/matches_with_features_ucl_enriched.csv \
  --features data/model_features.csv \
  --out data/predictions_ucl_logreg.csv

# Outputs:
#   - data/predictions_ucl_logreg.csv (Probabilities for each match)
```

**What it does**:
- Loads pre-trained model pipeline
- Applies delta feature engineering to new matches
- Generates class probability predictions (Away, Draw, Home)
- Saves predictions with match identifiers

#### Step 3: Compute Metrics (~1 minute)
```bash
python scripts/model_metrics.py

# Outputs:
#   - reports/model_metrics.csv (Accuracy, F1 score per model)
```

**What it does**:
- Loads ground truth labels (actual match outcomes)
- Compares predictions against actual outcomes
- Computes accuracy and F1 scores
- Generates summary CSV

#### Step 4: Generate Confusion Matrix (~1 minute)
```bash
python scripts/save_confusion_matrix.py

# Outputs:
#   - reports/confusion_matrix.png (Heatmap visualization)
```

**What it does**:
- Creates 3x3 confusion matrix (Away/Draw/Home vs Away/Draw/Home)
- Generates heatmap PNG with counts
- Shows prediction accuracy per class

#### Step 5: Quality Assurance Checks (~2 minutes, optional)
```bash
python scripts/qa_csvs.py

# Outputs:
#   - reports/qa_report.md   (Detailed QA report)
#   - reports/qa_summary.csv (QA metrics)
#   - reports/figs/*_nulls.png (Null value distributions)
```

**What it does**:
- Analyzes data completeness
- Reports missing values
- Detects data types and range issues
- Generates visualizations

---

## Individual Components

### 1. Model Training
```bash
python src/models/train_baseline.py --input <CSV_PATH>
```

**Arguments**:
- `--input`: Path to CSV with match features (default: `data/matches_with_features.csv`)

**Features Used**:
- Delta features: `delta_goals`, `delta_assists`, `delta_passes_completed`, etc.
- Raw features: `home_team_id`, `away_team_id`

**Models**:
- Logistic Regression (multiclass, LBFGS solver)
- XGBoost (multiclass softmax)

### 2. Batch Predictions
```bash
python src/models/predict.py \
  --model <logreg|xgb> \
  --input <CSV_PATH> \
  --features <FEATURE_CSV> \
  --out <OUTPUT_CSV>
```

**Arguments**:
- `--model`: Model type to use (logreg, xgb)
- `--input`: Match features CSV
- `--features`: Feature list from training (data/model_features.csv)
- `--out`: Output predictions CSV path

### 3. Metric Computation
```bash
python scripts/model_metrics.py
```

Computes accuracy and F1 scores across all trained models.

### 4. Confusion Matrix
```bash
python scripts/save_confusion_matrix.py
```

Generates visualization of prediction confusion matrix.

### 5. Data QA
```bash
python scripts/qa_csvs.py
```

Generates data quality report (completeness, distributions, samples).

### 6. Prediction Explanation (Optional)
```bash
# Explain specific match
python scripts/explain_predictions.py --match-id 523968

# Explain top-5 home win predictions
python scripts/explain_predictions.py --top 5
```

Shows per-feature contributions to predictions.

---

## Data Specification

### Required Input Files

The pipeline expects CSV files in the `data/` directory:

#### 1. **matches_with_features.csv** (Training data)
Main training dataset with match outcomes and team statistics.

**Required Columns**:
- `fulltime_home` (int): Goals scored by home team
- `fulltime_away` (int): Goals scored by away team
- `home_*` (numeric): Home team statistics (goals, assists, passes, etc.)
- `away_*` (numeric): Away team statistics
- `date_utc` or similar (datetime): Match date

**Example Structure**:
```
match_id,date_utc,home_team_name,away_team_name,fulltime_home,fulltime_away,home_goals,away_goals,home_assists,away_assists,...
```

#### 2. **matches_with_features_ucl_enriched.csv** (Prediction data)
UCL-specific matches for inference.

**Same structure as training data** (but outcome columns optional)

#### 3. **team_features.csv** (Team statistics)
Pre-aggregated team performance metrics.

**Required Columns**:
- `team_id` or `team`: Team identifier
- Statistical columns for aggregation

---

## Output Files

After running the complete pipeline, you'll find:

```
models/
├── logreg_pipeline.pkl       # Logistic Regression model + preprocessing
├── xgb_pipeline.pkl          # XGBoost model + preprocessing

data/
├── model_features.csv        # Feature list used in training
├── predictions_ucl_logreg.csv # Model predictions on UCL matches

reports/
├── model_report.md           # Training performance report
├── model_metrics.csv         # Metrics summary (accuracy, F1)
├── confusion_matrix.png      # Prediction confusion matrix heatmap
├── qa_report.md              # Data quality analysis
├── qa_summary.csv            # QA metrics summary
└── figs/
    ├── matches_with_features_nulls.png
    ├── team_features_nulls.png
    └── ... (other visualizations)
```

### Key Output Descriptions

| File | Purpose | Format |
|------|---------|--------|
| `logreg_pipeline.pkl` | Trained Logistic Regression model with preprocessing | Binary (joblib) |
| `xgb_pipeline.pkl` | Trained XGBoost model with preprocessing | Binary (joblib) |
| `predictions_ucl_logreg.csv` | Match predictions with probabilities | CSV |
| `model_report.md` | Training metrics and summary | Markdown |
| `model_metrics.csv` | Accuracy and F1 scores per model | CSV |
| `confusion_matrix.png` | Prediction distribution heatmap | PNG |

---

## Pipeline Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PREPARATION PHASE                       │
│                                                                  │
│  Load Matches CSV  →  Parse Dates  →  Handle Missing Values    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   FEATURE ENGINEERING PHASE                      │
│                                                                  │
│  Extract home_* / away_* columns  →  Create delta features     │
│                                                                  │
│  Delta: home_goals - away_goals = advantage/disadvantage        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   PREPROCESSING PHASE                            │
│                                                                  │
│  Imputation (median)  →  Scaling (StandardScaler)              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                     ┌────────┴────────┐
                     ↓                  ↓
        ┌─────────────────────┐  ┌──────────────────┐
        │ TRAIN-TEST SPLIT    │  │ TIME-AWARE SPLIT │
        │ (chronological)      │  │ (no data leakage)│
        └─────────────────────┘  └──────────────────┘
                     ↓                  ↓
        ┌──────────────────────────────────┐
        │   MODEL TRAINING PHASE           │
        │                                   │
        │  • Logistic Regression (LBFGS)   │
        │  • XGBoost (multiclass softmax)  │
        └──────────────────────────────────┘
                     ↓
        ┌──────────────────────────────────┐
        │  INFERENCE PHASE                 │
        │                                   │
        │  Apply features → Scale → Predict│
        │  Output: p_away, p_draw, p_home  │
        └──────────────────────────────────┘
                     ↓
        ┌──────────────────────────────────┐
        │  EVALUATION PHASE                │
        │                                   │
        │  • Compute Accuracy & F1 Score   │
        │  • Generate Confusion Matrix     │
        │  • Run QA Checks                 │
        └──────────────────────────────────┘
```

### Data Flow

```
Input Data                Processing              Output
─────────────────────────────────────────────────────────────
matches_with_features.csv
    ↓
    ├─→ [train_baseline.py] ─→ logreg_pipeline.pkl
    │                      ├─→ xgb_pipeline.pkl
    │                      ├─→ model_features.csv
    │                      └─→ model_report.md
    │
    └─→ [predict.py] ─→ predictions_ucl_logreg.csv
         ↓
    [model_metrics.py] ─→ model_metrics.csv
         ↓
    [save_confusion_matrix.py] ─→ confusion_matrix.png
         ↓
    [qa_csvs.py] ─→ qa_report.md, qa_summary.csv
```

---

## Dependencies

### Core Libraries

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥1.3.0 | Data loading and manipulation |
| `numpy` | ≥1.21.0 | Numerical operations |
| `scikit-learn` | ≥1.0.0 | Machine learning models, preprocessing |
| `xgboost` | ≥1.5.0 | Gradient boosting models |
| `lightgbm` | ≥3.3.0 | Alternative gradient boosting |
| `matplotlib` | ≥3.4.0 | Visualization |
| `seaborn` | ≥0.11.0 | Statistical plotting |
| `joblib` | ≥1.0.0 | Model serialization |
| `kagglehub` | ≥0.1.0 | Data access utility |

### Installation
```bash
pip install -r requirements.txt
```

---

## Usage Examples

### Example 1: Run Complete Pipeline
```bash
# One-command execution of entire pipeline
python run_pipeline.py
```

### Example 2: Train Only
```bash
# Train models and save pipelines
python src/models/train_baseline.py --input data/matches_with_features.csv
```

### Example 3: Predict Only (with existing models)
```bash
# Generate predictions using pre-trained model
python src/models/predict.py \
  --model logreg \
  --input data/matches_with_features_ucl_enriched.csv \
  --features data/model_features.csv \
  --out data/predictions_ucl_logreg.csv
```

### Example 4: Evaluate Specific Model
```bash
# Compute metrics for trained models
python scripts/model_metrics.py
```

---

## Troubleshooting

### Issue: "Module not found" error
**Solution**: Ensure virtual environment is activated and requirements installed:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Issue: "No such file or directory" for data files
**Solution**: Verify data files exist in `data/` directory:
```bash
ls -la data/*.csv
```

### Issue: Pipeline runs but produces no output
**Solution**: Check that output directories exist:
```bash
mkdir -p models reports/figs
python run_pipeline.py
```

### Issue: XGBoost training fails
**Solution**: XGBoost may require compilation. Try installing pre-built wheel:
```bash
pip install --upgrade xgboost
```

---

## Performance Characteristics

### Typical Execution Times

| Step | Duration | Notes |
|------|----------|-------|
| Train Baseline | 2-5 min | Depends on dataset size |
| Predictions | 1-2 min | Batch inference on UCL matches |
| Metrics | 1 min | Evaluation computation |
| Confusion Matrix | 1 min | Visualization generation |
| QA Checks | 2 min | Optional, can skip with `--quick` |
| **Total** | **10-15 min** | Full pipeline |

### Memory Usage
- Training: ~300-500 MB peak
- Inference: ~100-200 MB
- Visualization: ~50 MB

---

## Contributing & Customization

### Add Custom Features
Edit `src/models/train_baseline.py` to add custom feature engineering.

### Try Different Models
Add new model classes in `src/models/train_baseline.py` and update `train_and_evaluate()`.

### Modify Hyperparameters
Edit model initialization in `train_and_evaluate()` function.

---

## License & Contact

Project for UEFA Champions League match prediction. For questions, refer to inline code comments and docstrings.

---

## Summary Checklist

Before submission, ensure:

- ✅ Python 3.10+ and virtual environment set up
- ✅ All requirements installed: `pip install -r requirements.txt`
- ✅ Data files present in `data/` directory
- ✅ Output directories created: `models/`, `reports/`
- ✅ Run complete pipeline: `python run_pipeline.py`
- ✅ Verify outputs in `reports/` and `models/`
- ✅ Check timing and success messages in console

**To Grade**:
1. Set up environment: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
2. Run pipeline: `python run_pipeline.py`
3. Review outputs in `reports/` and `models/` directories
