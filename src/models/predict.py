"""
================================================================================
INFERENCE / PREDICTION SCRIPT
================================================================================
Purpose: Generate class probability predictions for UEFA Champions League matches
         using a trained baseline model pipeline.

Features:
  - Loads pre-trained model pipeline (Logistic Regression or XGBoost)
  - Builds delta features from input match CSV
  - Generates class probabilities for Away/Draw/Home outcomes
  - Saves predictions to CSV

Input:
  - models/{model}_pipeline.pkl (trained model)
  - data/matches_with_features_ucl_enriched.csv (default match features)
  - data/model_features.csv (feature list from training)

Output:
  - data/predictions_ucl_{model}.csv

Estimated Runtime: 1-2 minutes
================================================================================
"""
import argparse
import sys
import time
from datetime import datetime, timedelta
import pandas as pd
import os
import joblib
import numpy as np


def print_progress(msg: str, level: str = "INFO"):
    """Print timestamped progress message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level:7s}] {msg}")
    sys.stdout.flush()


def load_pipeline(path):
    """
    Load saved model pipeline (joblib pickle).
    
    Args:
        path (str): Path to pipeline pickle file
        
    Returns:
        dict: Contains 'model', 'imputer', 'scaler'
    """
    print_progress(f"Loading pipeline from {path}")
    obj = joblib.load(path)
    # expect dict with keys 'model','imputer','scaler'
    return obj


def main():
    """Main prediction pipeline."""
    print_progress("="*80)
    print_progress("UCL Match Outcome Predictor - Inference")
    print_progress("="*80)
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Generate predictions on UCL matches")
    parser.add_argument('--model', choices=['logreg','xgb'], default='logreg',
                       help='Model type')
    parser.add_argument('--input', default='data/matches_with_features_ucl_enriched.csv',
                       help='Input match features CSV')
    parser.add_argument('--features', default='data/model_features.csv',
                       help='Feature list from training')
    parser.add_argument('--out', default='data/predictions_ucl.csv',
                       help='Output predictions CSV')
    args = parser.parse_args()

    try:
        # ===== LOAD DATA =====
        print_progress(f"Loading match data from {args.input}")
        df = pd.read_csv(args.input)
        print_progress(f"Loaded {len(df)} matches")
        
        # ===== BUILD DELTA FEATURES =====
        print_progress("Building delta features...")
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

        # ===== LOAD EXPECTED FEATURES =====
        print_progress(f"Loading feature list from {args.features}")
        if os.path.exists(args.features):
            feat_df = pd.read_csv(args.features)
            # Read feature names from 'feature' column
            if 'feature' in feat_df.columns:
                feature_cols = feat_df['feature'].tolist()
            else:
                # Fallback: treat all columns as features
                feature_cols = [c for c in feat_df.columns if c != 'Unnamed: 0']
        else:
            # fallback: use delta_ or last/elo features present in df
            feature_cols = [c for c in df.columns if c.startswith('delta_') or c.startswith('home_last') or c.endswith('_per90') or c.endswith('_elo_before')]
            print_progress(f"Feature file not found; using {len(feature_cols)} inferred features", level="WARN")

        # keep only features that exist in df (preserve order from feature_cols)
        feature_cols = [c for c in feature_cols if c in df.columns]
        if not feature_cols:
            raise RuntimeError('No feature columns found for prediction')
        print_progress(f"Using {len(feature_cols)} features for prediction")

        # ===== LOAD MODEL =====
        model_file = os.path.join('models', f"{args.model}_pipeline.pkl")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model pipeline not found: {model_file}")

        pipeline = load_pipeline(model_file)
        imputer = pipeline.get('imputer')
        scaler = pipeline.get('scaler')
        model = pipeline.get('model')

        # ===== PREDICT =====
        print_progress("Generating predictions...")
        X = df[feature_cols].copy()
        
        # Ensure only numeric columns
        numeric_feature_cols = []
        for col in feature_cols:
            if col in X.columns and np.issubdtype(X[col].dtype, np.number):
                numeric_feature_cols.append(col)
        
        X = X[numeric_feature_cols]
        if X.empty or len(numeric_feature_cols) == 0:
            raise RuntimeError('No numeric feature columns available after filtering')
        
        # simple impute/scale
        X_imp = imputer.transform(X)
        X_scaled = scaler.transform(X_imp)
        probs = model.predict_proba(X_scaled)

        # ===== SAVE RESULTS =====
        print_progress("Saving predictions...")
        cols = ['match_id','date_utc','home_team_name','away_team_name'] if 'match_id' in df.columns else ['date_utc','home_team_name','away_team_name']
        out = df[cols].copy()
        # probs columns correspond to classes [Away,Draw,Home]
        out['p_away'] = probs[:,0]
        out['p_draw'] = probs[:,1]
        out['p_home'] = probs[:,2]
        out.to_csv(args.out, index=False)
        print_progress(f"✓ Predictions saved to {args.out}")
        
        # ===== COMPLETION =====
        elapsed_time = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed_time)))
        print_progress("="*80)
        print_progress(f"✓ Inference Completed Successfully in {elapsed_str}")
        print_progress("="*80)
        
    except Exception as e:
        print_progress(f"✗ Inference Failed: {str(e)}", level="ERROR")
        print_progress("="*80)
        sys.exit(1)


if __name__ == '__main__':
    main()
