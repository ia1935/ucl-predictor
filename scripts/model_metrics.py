#!/usr/bin/env python3
"""
================================================================================
MODEL METRICS COMPUTATION
================================================================================
Purpose: Compute classification metrics (accuracy, F1) for all trained models.

Input:
  - models/*_pipeline.pkl (trained model pipelines)
  - data/matches_with_features_ucl_enriched.csv (ground truth labels)
  - data/predictions_{model}.csv (pre-computed predictions if available)

Output:
  - reports/model_metrics.csv (accuracy, F1-macro per model)

Estimated Runtime: 1-2 minutes
================================================================================
"""
import os
import sys
import glob
import time
from datetime import datetime, timedelta
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


MATCH_ENRICHED = 'data/matches_with_features_ucl_enriched.csv'
MATCH_FALLBACK = 'data/matches_with_features.csv'
MODEL_GLOB = 'models/*_pipeline.pkl'
OUT_CSV = 'reports/model_metrics.csv'


def print_progress(msg: str, level: str = "INFO"):
    """Print timestamped progress message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level:7s}] {msg}")
    sys.stdout.flush()


def load_matches():
    """Load matches file with ground-truth labels."""
    path = MATCH_ENRICHED if os.path.exists(MATCH_ENRICHED) else MATCH_FALLBACK
    if not os.path.exists(path):
        raise FileNotFoundError(f'No matches file found (tried {MATCH_ENRICHED} and {MATCH_FALLBACK})')
    print_progress(f"Loading matches from {path}")
    df = pd.read_csv(path)
    print_progress(f"Loaded {len(df)} matches")
    return df


def build_true_labels(df):
    """Build true labels: 0=Away, 1=Draw, 2=Home."""
    if 'fulltime_home' in df.columns and 'fulltime_away' in df.columns:
        def outcome(row):
            if row['fulltime_home'] > row['fulltime_away']: return 2
            if row['fulltime_home'] < row['fulltime_away']: return 0
            return 1
        return df.apply(outcome, axis=1).astype(int)
    else:
        raise RuntimeError('Matches file missing fulltime_home/fulltime_away columns')


def build_delta_features(df):
    """Build delta features from home_*/away_* prefixed columns."""
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
    """Main metrics computation pipeline."""
    print_progress("="*80)
    print_progress("MODEL METRICS COMPUTATION")
    print_progress("="*80)
    start_time = time.time()
    
    try:
        # ===== LOAD DATA =====
        matches = load_matches()
        print_progress("Building true labels...")
        y_true = build_true_labels(matches).values
        print_progress(f"True labels: Away={np.sum(y_true==0)}, Draw={np.sum(y_true==1)}, Home={np.sum(y_true==2)}")

        # ===== EVALUATE MODELS =====
        results = []
        model_files = glob.glob(MODEL_GLOB)
        print_progress(f"Found {len(model_files)} model pipelines")
        
        for p in model_files:
            base = os.path.basename(p)
            name = base.replace('_pipeline.pkl','').replace('.pkl','')
            print_progress(f"Evaluating {name}...")
            
            pred_csv = f'data/predictions_{name}.csv'
            y_pred = None

            # Try loading pre-computed predictions first
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
                        y_pred = None
                except Exception as e:
                    print_progress(f"  Could not load predictions from {pred_csv}: {str(e)}", level="WARN")
                    y_pred = None

            if y_pred is None:
                # generate predictions using pipeline
                try:
                    print_progress(f"  Generating predictions from pipeline...")
                    pred_df = predict_with_pipeline(p, matches)
                    y_pred = pred_df['y_pred'].astype(int).values
                except Exception as e:
                    print_progress(f"  Could not generate predictions for {name}: {e}", level="ERROR")
                    continue

            # ensure lengths match
            if len(y_pred) != len(y_true):
                # try to align by dropping NaNs in y_true
                if len(y_pred) == len(matches):
                    # ok
                    pass
                else:
                    print_progress(f"  Length mismatch: y_pred {len(y_pred)} vs y_true {len(y_true)}; skipping", level="WARN")
                    continue

            # Compute metrics
            acc = float(accuracy_score(y_true, y_pred))
            f1m = float(f1_score(y_true, y_pred, average='macro'))
            results.append({'model': name, 'accuracy': acc, 'f1_macro': f1m, 'n_samples': int(len(y_true))})
            print_progress(f"  ✓ {name}: accuracy={acc:.4f}, f1_macro={f1m:.4f}")

        if not results:
            print_progress("✗ No results produced; check that model pipelines exist", level="ERROR")
            return 1

        # ===== SAVE RESULTS =====
        print_progress(f"Saving {len(results)} results to {OUT_CSV}")
        out_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
        out_df.to_csv(OUT_CSV, index=False)
        print_progress(f"✓ Metrics saved to {OUT_CSV}")
        
        # ===== COMPLETION =====
        elapsed_time = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed_time)))
        print_progress("="*80)
        print_progress(f"✓ Metrics Computation Completed in {elapsed_str}")
        print_progress("="*80)
        return 0
        
    except Exception as e:
        print_progress(f"✗ Metrics Computation Failed: {str(e)}", level="ERROR")
        print_progress("="*80)
        return 1


if __name__ == '__main__':
    main()
