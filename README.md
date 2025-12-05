# ucl-predictor

Lightweight pipeline for training a baseline model to predict UEFA Champions League match outcomes, running inference, and inspecting results.

## Quickstart
- Requires Python 3.10+ and a Unix-like shell.
- Run everything from the repo root so relative data paths resolve.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data inputs
- Training defaults: `data/matches_with_features.csv`
- UCL scoring defaults: `data/matches_with_features_ucl_enriched.csv`
- Model feature header saved after training: `data/model_features.csv`
- If you need to build a UCL-only matches file from the raw data bundle, run:
	```bash
	python scripts/merge_ucl_matches.py
	```

## Train a baseline model
Trains delta-based features, fits Logistic Regression and XGBoost, writes pipelines and a short report.

```bash
python src/models/train_baseline.py --input data/matches_with_features.csv
# outputs: models/logreg_pipeline.pkl, models/xgb_pipeline.pkl, data/model_features.csv, reports/model_report.md
```

## Run predictions on UCL matches
Generates class probabilities for each match using a saved pipeline.

```bash
python src/models/predict.py \
	--model logreg \
	--input data/matches_with_features_ucl_enriched.csv \
	--features data/model_features.csv \
	--out data/predictions_ucl_logreg.csv
```

## Evaluate and visualize
- Metrics across all saved pipelines:
	```bash
	python scripts/model_metrics.py
	# writes reports/model_metrics.csv
	```
- Confusion matrix PNG (expects `predictions_ucl_logreg.csv`):
	```bash
	python scripts/save_confusion_matrix.py
	# writes reports/confusion_matrix.png
	```

## Explain predictions
Show per-feature contributions for Logistic Regression predictions.

```bash
python scripts/explain_predictions.py --match-id 523968
# or: python scripts/explain_predictions.py --top 5  # top-N highest p_home rows
```

## QA the data drops (optional)
Produces a lightweight QA summary and sample plots for key CSVs.

```bash
python scripts/qa_csvs.py
# writes reports/qa_report.md and reports/qa_summary.csv
```
