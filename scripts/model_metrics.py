#!/usr/bin/env python3
"""Compute model metrics (accuracy and F1) for saved model pipelines.

Generates `reports/model_metrics.csv` with columns: model, accuracy, f1_macro, n_samples

Behavior:
- Loads ground-truth from `data/matches_with_features_ucl_enriched.csv` (falls back to `data/matches_with_features.csv`).
- For each pipeline file `models/*_pipeline.pkl` it will either use an existing predictions file
  `data/predictions_{model}.csv` if present, or generate predictions by applying the pipeline to the
  enriched matches file (it will auto-build delta_ features similar to training code).
"""
import os
import glob
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


MATCH_ENRICHED = 'data/matches_with_features_ucl_enriched.csv'
MATCH_FALLBACK = 'data/matches_with_features.csv'
MODEL_GLOB = 'models/*_pipeline.pkl'
OUT_CSV = 'reports/model_metrics.csv'


def load_matches():
    path = MATCH_ENRICHED if os.path.exists(MATCH_ENRICHED) else MATCH_FALLBACK
    if not os.path.exists(path):
        raise FileNotFoundError(f'No matches file found (tried {MATCH_ENRICHED} and {MATCH_FALLBACK})')
    df = pd.read_csv(path)
    return df


def build_true_labels(df):
    if 'fulltime_home' in df.columns and 'fulltime_away' in df.columns:
        def outcome(row):
            if row['fulltime_home'] > row['fulltime_away']: return 2
            if row['fulltime_home'] < row['fulltime_away']: return 0
            return 1
        return df.apply(outcome, axis=1).astype(int)
    else:
        raise RuntimeError('Matches file missing fulltime_home/fulltime_away columns')


def build_delta_features(df):
    # create delta_ columns from common home_/away_ prefixed numeric columns
    out = df.copy()
    home_cols = [c for c in out.columns if c.startswith('home_')]
    away_cols = [c for c in out.columns if c.startswith('away_')]
    home_suffixes = {c[len('home_'):]: c for c in home_cols}
    away_suffixes = {c[len('away_'):]: c for c in away_cols}
    common = set(home_suffixes.keys()) & set(away_suffixes.keys())
    for s in common:
        h = home_suffixes[s]
        a = away_suffixes[s]
        try:
            if np.issubdtype(out[h].dtype, np.number) and np.issubdtype(out[a].dtype, np.number):
                out[f'delta_{s}'] = out[h] - out[a]
        except Exception:
            continue
    return out


def predict_with_pipeline(pipeline_path, matches_df, feature_cols=None):
    pipe = joblib.load(pipeline_path)
    imputer = pipe.get('imputer')
    scaler = pipe.get('scaler')
    model = pipe.get('model')

    df = build_delta_features(matches_df)

    # determine feature columns
    if feature_cols is None:
        # try to load data/model_features.csv header
        mf = 'data/model_features.csv'
        if os.path.exists(mf):
            try:
                feat_df = pd.read_csv(mf, nrows=0)
                feature_cols = [c for c in feat_df.columns if not c.lower().startswith('date')]
            except Exception:
                feature_cols = [c for c in df.columns if c.startswith('delta_') or c.endswith('_elo_before') or c.endswith('_per90')]
        else:
            feature_cols = [c for c in df.columns if c.startswith('delta_') or c.endswith('_elo_before') or c.endswith('_per90')]

    feature_cols = [c for c in feature_cols if c in df.columns]
    if not feature_cols:
        raise RuntimeError('No feature columns available for prediction')

    X = df[feature_cols].select_dtypes(include=[np.number]).copy()
    # impute and scale
    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    probs = model.predict_proba(X_scaled)
    preds = probs.argmax(axis=1)
    out = pd.DataFrame({'match_id': df['match_id'] if 'match_id' in df.columns else range(len(df)),
                        'y_pred': preds})
    return out


def main():
    matches = load_matches()
    y_true = build_true_labels(matches).values

    results = []
    for p in glob.glob(MODEL_GLOB):
        base = os.path.basename(p)
        name = base.replace('_pipeline.pkl','').replace('.pkl','')
        pred_csv = f'data/predictions_{name}.csv'
        y_pred = None

        if os.path.exists(pred_csv):
            try:
                pred_df = pd.read_csv(pred_csv)
                if {'p_away','p_draw','p_home'}.issubset(set(pred_df.columns)):
                    y_pred = pred_df[['p_away','p_draw','p_home']].values.argmax(axis=1)
                elif 'pred_class' in pred_df.columns:
                    y_pred = pred_df['pred_class'].astype(int).values
                elif 'y_pred' in pred_df.columns:
                    y_pred = pred_df['y_pred'].astype(int).values
                else:
                    # last resort: try running pipeline
                    y_pred = None
            except Exception:
                y_pred = None

        if y_pred is None:
            # generate predictions using pipeline
            try:
                pred_df = predict_with_pipeline(p, matches)
                y_pred = pred_df['y_pred'].astype(int).values
            except Exception as e:
                print(f'Could not generate predictions for {name}:', e)
                continue

        # ensure lengths match
        if len(y_pred) != len(y_true):
            # try to align by dropping NaNs in y_true
            if len(y_pred) == len(matches):
                # ok
                pass
            else:
                print(f'Length mismatch for {name}: y_pred {len(y_pred)} vs y_true {len(y_true)}; skipping')
                continue

        acc = float(accuracy_score(y_true, y_pred))
        f1m = float(f1_score(y_true, y_pred, average='macro'))
        results.append({'model': name, 'accuracy': acc, 'f1_macro': f1m, 'n_samples': int(len(y_true))})

    if not results:
        print('No results produced; check that model pipelines exist in models/ and matches file is available')
        return

    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print('Wrote', OUT_CSV)


if __name__ == '__main__':
    main()
