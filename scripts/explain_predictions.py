#!/usr/bin/env python3
"""Explain logistic-regression predictions by listing top per-feature contributions.

Usage:
  python3 scripts/explain_predictions.py --match-id 523968
  python3 scripts/explain_predictions.py --top 5

Outputs a simple text breakdown to stdout for selected matches.
"""
import argparse
import os
import joblib
import pandas as pd
import numpy as np

PIPE = 'models/logreg_pipeline.pkl'
FEATURES_CSV = 'data/model_features.csv'
PRED_CSV = 'data/predictions_ucl_logreg.csv'
MATCHES = 'data/matches_with_features_ucl_enriched.csv'


def load_feature_list(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    # header contains training-sample rows; take column names
    return [c for c in df.columns if not c.lower().startswith('date')]


def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()


def explain(pipeline_path, features_csv, preds_csv, matches_csv, match_id=None, top=3):
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(pipeline_path)
    pipe = joblib.load(pipeline_path)
    model = pipe.get('model')
    imputer = pipe.get('imputer')
    scaler = pipe.get('scaler')

    feat_list = load_feature_list(features_csv)
    preds = pd.read_csv(preds_csv)
    matches = pd.read_csv(matches_csv)

    if match_id is not None:
        rows = preds[preds['match_id'].astype(int) == int(match_id)]
    else:
        rows = preds.sort_values('p_home', ascending=False).head(top)

    if rows.empty:
        print('No prediction rows found to explain')
        return

    for _, prow in rows.iterrows():
        mid = int(prow['match_id']) if 'match_id' in prow else None
        mrow = None
        if mid is not None and 'match_id' in matches.columns:
            mm = matches[matches['match_id'].astype(int) == mid]
            if not mm.empty:
                mrow = mm.iloc[0]
        if mrow is None:
            print('No feature row found for match', mid)
            continue

        # build raw feature vector in training order
        x_raw = []
        for f in feat_list:
            val = mrow.get(f, np.nan) if f in mrow.index else np.nan
            try:
                x_raw.append(float(val))
            except Exception:
                x_raw.append(np.nan)
        X_raw = np.array(x_raw).reshape(1, -1)

        X_imp = imputer.transform(X_raw)
        X_scaled = scaler.transform(X_imp)

        coefs = model.coef_  # (K, M)
        intercept = model.intercept_
        scores = (coefs @ X_scaled.T).flatten() + intercept
        probs = softmax(scores)
        pred_class = int(np.argmax(probs))
        cls_map = {0: 'Away', 1: 'Draw', 2: 'Home'}

        contribs = coefs[pred_class] * X_scaled.flatten()
        idx_order = np.argsort(np.abs(contribs))[::-1]

        print('\n=== Explanation for match_id:', mid, prow.get('home_team_name'), 'vs', prow.get('away_team_name'), '===')
        print(' Predicted:', cls_map[pred_class])
        print(' Probabilities: Away={:.3f} Draw={:.3f} Home={:.3f}'.format(*probs))
        print(' Top feature contributions (feature, raw, scaled, contribution):')
        for idx in idx_order[:8]:
            f = feat_list[idx]
            raw_v = X_raw[0, idx]
            scaled_v = X_scaled[0, idx]
            c = contribs[idx]
            sign = '+' if c > 0 else '-'
            print(f"  {f:35s} raw={raw_v:8.3g} scaled={scaled_v:8.3g} contrib={sign}{abs(c):8.4f}")


def main():
    parser = argparse = __import__('argparse').ArgumentParser()
    parser.add_argument('--match-id', type=int, help='match_id to explain')
    parser.add_argument('--top', type=int, default=3, help='top N predicted home matches to explain')
    args = parser.parse_args()

    explain(PIPE, FEATURES_CSV, PRED_CSV, MATCHES, match_id=args.match_id, top=args.top)


if __name__ == '__main__':
    main()
