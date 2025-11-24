"""Inference script for saved baseline models.
Loads a saved pipeline (joblib containing {'model','imputer','scaler'}) and scores matches.
Usage: .venv/bin/python3 src/models/predict.py --model logreg --input data/matches_with_features_ucl_enriched.csv
"""
import argparse
import pandas as pd
import os
import joblib
import numpy as np


def load_pipeline(path):
    obj = joblib.load(path)
    # expect dict with keys 'model','imputer','scaler'
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['logreg','xgb'], default='logreg')
    parser.add_argument('--input', default='data/matches_with_features_ucl_enriched.csv')
    parser.add_argument('--features', default='data/model_features.csv')
    parser.add_argument('--out', default='data/predictions_ucl.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    # Build delta features (same logic as training)
    home_cols = [c for c in df.columns if c.startswith('home_')]
    away_cols = [c for c in df.columns if c.startswith('away_')]
    home_suffixes = {c[len('home_'):]: c for c in home_cols}
    away_suffixes = {c[len('away_'):]: c for c in away_cols}
    common = set(home_suffixes.keys()) & set(away_suffixes.keys())
    for s in common:
        h = home_suffixes[s]
        a = away_suffixes[s]
        try:
            if np.issubdtype(df[h].dtype, np.number) and np.issubdtype(df[a].dtype, np.number):
                df[f'delta_{s}'] = df[h] - df[a]
        except Exception:
            continue

    # load expected feature list from training (columns of data/model_features.csv)
    if os.path.exists(args.features):
        feat_df = pd.read_csv(args.features)
        feature_cols = [c for c in feat_df.columns]
    else:
        # fallback: use delta_ or last/elo features present in df
        feature_cols = [c for c in df.columns if c.startswith('delta_') or c.startswith('home_last') or c.endswith('_per90') or c.endswith('_elo_before')]

    # keep only features that exist in df (preserve order from feature_cols)
    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        raise RuntimeError('No feature columns found for prediction')

    model_file = os.path.join('models', f"{args.model}_pipeline.pkl")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model pipeline not found: {model_file}")

    pipeline = load_pipeline(model_file)
    imputer = pipeline.get('imputer')
    scaler = pipeline.get('scaler')
    model = pipeline.get('model')

    X = df[feature_cols].copy()
    # drop any non-numeric columns that snuck in
    X = X.select_dtypes(include=[np.number])
    # simple impute/scale
    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    probs = model.predict_proba(X_scaled)

    # build output
    cols = ['match_id','date_utc','home_team_name','away_team_name'] if 'match_id' in df.columns else ['date_utc','home_team_name','away_team_name']
    out = df[cols].copy()
    # probs columns correspond to classes [Away,Draw,Home]
    out['p_away'] = probs[:,0]
    out['p_draw'] = probs[:,1]
    out['p_home'] = probs[:,2]
    out.to_csv(args.out, index=False)
    print('Wrote predictions to', args.out)


if __name__ == '__main__':
    main()
