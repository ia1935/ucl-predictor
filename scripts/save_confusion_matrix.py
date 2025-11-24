#!/usr/bin/env python3
"""Save a confusion matrix PNG from prediction CSV and match truth.

Usage:
  python3 scripts/save_confusion_matrix.py

Outputs: `reports/confusion_matrix.png`
"""
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

PRED_CSV = 'data/predictions_ucl_logreg.csv'
MATCH_CSV = 'data/matches_with_features_ucl_enriched.csv'
OUT_PNG = 'reports/confusion_matrix.png'


def load_data(pred_csv, match_csv):
    if not os.path.exists(pred_csv):
        raise FileNotFoundError(pred_csv)
    if not os.path.exists(match_csv):
        raise FileNotFoundError(match_csv)
    pred = pd.read_csv(pred_csv)
    match = pd.read_csv(match_csv)
    return pred, match


def build_labels(pred, match):
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
        if pd.isna(r['fulltime_home']) or pd.isna(r['fulltime_away']):
            return None
        if r['fulltime_home'] > r['fulltime_away']:
            return 2
        if r['fulltime_home'] < r['fulltime_away']:
            return 0
        return 1

    merged['y_true'] = merged.apply(outcome, axis=1)

    # predicted labels
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
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    fig, ax = plt.subplots(figsize=(5,4))
    cax = ax.matshow(cm, cmap='Blues')
    for (i, j), val in __import__('numpy').ndenumerate(cm):
        ax.text(j, i, int(val), ha='center', va='center', color='black', fontsize=12)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
    ax.set_xticklabels(['Away','Draw','Home']); ax.set_yticklabels(['Away','Draw','Home'])
    fig.colorbar(cax, ax=ax)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print('Wrote', out_png)


def main():
    try:
        pred, match = load_data(PRED_CSV, MATCH_CSV)
        y_true, y_pred = build_labels(pred, match)
        if len(y_true) == 0:
            print('No ground-truth rows found after merge; check your CSVs')
            return
        plot_cm(y_true, y_pred, OUT_PNG)
    except Exception as e:
        print('Error:', e)
        raise


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
