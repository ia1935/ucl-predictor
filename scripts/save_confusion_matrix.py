#!/usr/bin/env python3
"""
================================================================================
CONFUSION MATRIX VISUALIZATION
================================================================================
Purpose: Generate a heatmap visualization of prediction confusion matrix.

Input:
  - data/predictions_ucl_logreg.csv (model predictions with probabilities)
  - data/matches_with_features_ucl_enriched.csv (ground truth labels)

Output:
  - reports/confusion_matrix.png (3x3 confusion matrix heatmap)

Estimated Runtime: 1-2 minutes
================================================================================
"""
import os
import sys
import time
from datetime import datetime, timedelta
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

PRED_CSV = 'data/predictions_ucl_logreg.csv'
MATCH_CSV = 'data/matches_with_features_ucl_enriched.csv'
OUT_PNG = 'reports/confusion_matrix.png'


def print_progress(msg: str, level: str = "INFO"):
    """Print timestamped progress message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level:7s}] {msg}")
    sys.stdout.flush()


def load_data(pred_csv, match_csv):
    """Load prediction and match CSV files."""
    print_progress(f"Loading predictions from {pred_csv}")
    if not os.path.exists(pred_csv):
        raise FileNotFoundError(f"Predictions file not found: {pred_csv}")
    if not os.path.exists(match_csv):
        raise FileNotFoundError(f"Match file not found: {match_csv}")
    pred = pd.read_csv(pred_csv)
    match = pd.read_csv(match_csv)
    print_progress(f"Loaded {len(pred)} predictions and {len(match)} matches")
    return pred, match


def build_labels(pred, match):
    """Build true and predicted labels from predictions and match data."""
    print_progress("Merging predictions with ground truth...")
    # merge on match_id when possible
    if 'match_id' in pred.columns and 'match_id' in match.columns:
        merged = pred.merge(match[['match_id', 'fulltime_home', 'fulltime_away']], on='match_id', how='left')
    else:
        # try best-effort merge on date + teams
        keys = []
        for c in ('date_utc', 'home_team_name', 'away_team_name'):
            if c in pred.columns and c in match.columns:
                keys.append(c)
        if keys:
            merged = pred.merge(match[['date_utc','home_team_name','away_team_name','fulltime_home','fulltime_away']], on=keys, how='left')
        else:
            raise RuntimeError('No merge keys available (need match_id or date_utc+team names)')

    def outcome(r):
        """Convert goals to outcome: 0=Away, 1=Draw, 2=Home."""
        if pd.isna(r['fulltime_home']) or pd.isna(r['fulltime_away']):
            return None
        if r['fulltime_home'] > r['fulltime_away']:
            return 2
        if r['fulltime_home'] < r['fulltime_away']:
            return 0
        return 1

    merged['y_true'] = merged.apply(outcome, axis=1)

    # predicted labels from probabilities
    if {'p_away','p_draw','p_home'}.issubset(set(merged.columns)):
        probs = merged[['p_away','p_draw','p_home']].values
        y_pred = probs.argmax(axis=1)
    elif 'pred_class' in merged.columns:
        y_pred = merged['pred_class'].astype(int).values
    else:
        raise RuntimeError('No probability columns (p_away/p_draw/p_home) or pred_class found in predictions')

    y_true = merged['y_true'].values
    mask = pd.notnull(y_true)
    y_true = y_true[mask].astype(int)
    y_pred = y_pred[mask].astype(int)
    return y_true, y_pred


def plot_cm(y_true, y_pred, out_png):
    """Generate and save confusion matrix heatmap."""
    print_progress("Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    
    fig, ax = plt.subplots(figsize=(5,4))
    cax = ax.matshow(cm, cmap='Blues')
    
    # Add text annotations
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha='center', va='center', color='black', fontsize=12)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks([0,1,2])
    ax.set_yticks([0,1,2])
    ax.set_xticklabels(['Away','Draw','Home'])
    ax.set_yticklabels(['Away','Draw','Home'])
    fig.colorbar(cax, ax=ax)
    
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print_progress(f"✓ Confusion matrix saved to {out_png}")


def main():
    """Main confusion matrix generation."""
    print_progress("="*80)
    print_progress("CONFUSION MATRIX VISUALIZATION")
    print_progress("="*80)
    start_time = time.time()
    
    try:
        pred, match = load_data(PRED_CSV, MATCH_CSV)
        y_true, y_pred = build_labels(pred, match)
        
        if len(y_true) == 0:
            print_progress("✗ No ground-truth rows found after merge; check your CSVs", level="ERROR")
            return 1
        
        print_progress(f"Generating matrix from {len(y_true)} samples")
        plot_cm(y_true, y_pred, OUT_PNG)
        
        elapsed_time = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed_time)))
        print_progress("="*80)
        print_progress(f"✓ Visualization Completed in {elapsed_str}")
        print_progress("="*80)
        return 0
        
    except Exception as e:
        print_progress(f"✗ Visualization Failed: {str(e)}", level="ERROR")
        print_progress("="*80)
        return 1


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PRED_CSV = 'data/predictions_ucl_logreg.csv'
MATCH_CSV = 'data/matches_with_features_ucl_enriched.csv'   # or data/matches_with_features.csv
OUT_PNG = 'reports/confusion_matrix.png'

if not os.path.exists(PRED_CSV):
    raise SystemExit(f'Predictions file missing: {PRED_CSV}')
if not os.path.exists(MATCH_CSV):
    raise SystemExit(f'Match file missing: {MATCH_CSV}')

pred = pd.read_csv(PRED_CSV)
matches = pd.read_csv(MATCH_CSV)

# Merge on match_id (fallback to error if not present)
if 'match_id' not in pred.columns or 'match_id' not in matches.columns:
    raise SystemExit('match_id missing from predictions or matches file; ensure match_id present.')

df = pred.merge(matches[['match_id','fulltime_home','fulltime_away']], on='match_id', how='left')

# Build true label: 0=Away, 1=Draw, 2=Home
def outcome(r):
    if r['fulltime_home'] > r['fulltime_away']: return 2
    if r['fulltime_home'] < r['fulltime_away']: return 0
    return 1

df['y_true'] = df.apply(outcome, axis=1)

# Build predicted label from probabilities (if available)
if {'p_away','p_draw','p_home'}.issubset(df.columns):
    probs = df[['p_away','p_draw','p_home']].values
    y_pred = probs.argmax(axis=1)
else:
    # fallback: if pred contains pred_class column
    if 'pred_class' in df.columns:
        y_pred = df['pred_class'].astype(int).values
    else:
        raise SystemExit('No probability columns (p_away/p_draw/p_home) or pred_class found in predictions file.')

y_true = df['y_true'].astype(int).values

cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
fig, ax = plt.subplots(figsize=(5,4))
cax = ax.matshow(cm, cmap='Blues')
for (i, j), val in np.ndenumerate(cm):
    ax.text(j, i, int(val), ha='center', va='center', color='black', fontsize=12)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
ax.set_xticklabels(['Away','Draw','Home']); ax.set_yticklabels(['Away','Draw','Home'])
fig.colorbar(cax, ax=ax)
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
fig.savefig(OUT_PNG, bbox_inches='tight', dpi=150)
print('Wrote', OUT_PNG)
