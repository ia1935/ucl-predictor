"""
================================================================================
BASELINE MODEL TRAINER
================================================================================
Purpose: Train Logistic Regression and XGBoost models for UEFA Champions League
         match outcome prediction using delta-based features.

Features:
  - Builds delta features (home_stat - away_stat) from input CSV
  - Trains two models: Logistic Regression and XGBoost
  - Time-aware train/test split (chronological ordering)
  - Saves trained pipelines and feature list for inference
  - Generates model performance report

Input:  data/matches_with_features.csv (default)
Output: 
  - models/logreg_pipeline.pkl
  - models/xgb_pipeline.pkl
  - data/model_features.csv
  - reports/model_report.md

Estimated Runtime: 2-5 minutes
================================================================================
"""
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def print_progress(msg: str, level: str = "INFO"):
    """Print timestamped progress message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level:7s}] {msg}")
    sys.stdout.flush()


def load_data(path='data/matches_with_features.csv'):
    """
    Load matches CSV and parse date column if present.
    
    Args:
        path (str): Path to matches CSV file
        
    Returns:
        tuple: (DataFrame, date_column_name or None)
    """
    print_progress(f"Loading data from {path}")
    df = pd.read_csv(path)
    print_progress(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # parse date if present (search for common date column names)
    date_col = None
    for c in df.columns:
        if c.lower().startswith('date') or 'date_' in c.lower() or 'utc' in c.lower():
            date_col = c
            break
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        print_progress(f"Parsed date column: {date_col}")
    else:
        print_progress("No date column found", level="WARN")
    return df, date_col


def build_features(df):
    """
    Build delta features from home_* and away_* prefixed columns.
    Delta features represent advantage/disadvantage: home_stat - away_stat
    
    Args:
        df (DataFrame): Matches dataframe
        
    Returns:
        tuple: (DataFrame with delta features, list of feature column names)
    """
    print_progress("Building delta features...")
    
    # identify numeric home/away prefixed columns
    home_cols = [c for c in df.columns if c.startswith('home_')]
    away_cols = [c for c in df.columns if c.startswith('away_')]
    print_progress(f"Found {len(home_cols)} home_* and {len(away_cols)} away_* columns")
    
    # match suffixes (e.g., home_goals / away_goals -> delta_goals)
    home_suffixes = {c[len('home_'):]: c for c in home_cols}
    away_suffixes = {c[len('away_'):]: c for c in away_cols}
    common = set(home_suffixes.keys()) & set(away_suffixes.keys())
    
    feature_cols = []
    for s in common:
        h = home_suffixes[s]
        a = away_suffixes[s]
        # only numeric columns
        try:
            if np.issubdtype(df[h].dtype, np.number) and np.issubdtype(df[a].dtype, np.number):
                df[f'delta_{s}'] = df[h] - df[a]
                feature_cols.append(f'delta_{s}')
        except Exception:
            continue

    # also include some raw features like home_team_id/away_team_id if present
    for c in ['home_team_id', 'away_team_id', 'match_id']:
        if c in df.columns and np.issubdtype(df[c].dtype, np.number):
            # don't include ids as features normally, skip match_id
            if c != 'match_id':
                df[c] = df[c].astype(float)
                feature_cols.append(c)

    # Filter to only numeric features with non-NaN values
    numeric_cols = []
    for col in feature_cols:
        if np.issubdtype(df[col].dtype, np.number):
            # Also ensure column has at least some non-NaN values
            if df[col].notna().sum() > 0:
                numeric_cols.append(col)
    
    print_progress(f"Created {len(numeric_cols)} delta/raw features (filtered to numeric with values)")
    return df, numeric_cols


def build_target(df):
    """
    Build multiclass target: 0=Away Win, 1=Draw, 2=Home Win
    Tries fulltime_home/fulltime_away columns first, then match_outcome.
    
    Args:
        df (DataFrame): Matches dataframe
        
    Returns:
        Series: Target labels
    """
    # create multiclass target: 0=away,1=draw,2=home
    if 'fulltime_home' in df.columns and 'fulltime_away' in df.columns:
        y = df[['fulltime_home', 'fulltime_away']].apply(
            lambda x: 2 if x['fulltime_home']>x['fulltime_away'] 
            else (1 if x['fulltime_home']==x['fulltime_away'] else 0), 
            axis=1
        )
        return y
    # fallback to match_outcome strings
    if 'match_outcome' in df.columns:
        mapping = {'Away Win':0, 'Draw':1, 'Home Win':2}
        return df['match_outcome'].map(mapping).fillna(1).astype(int)
    raise RuntimeError('No suitable target columns found (need fulltime_home/fulltime_away or match_outcome)')


def time_aware_split(df, date_col, test_size=0.2):
    """
    Time-aware train/test split: later matches in test set.
    Ensures no data leakage by maintaining temporal ordering.
    
    Args:
        df (DataFrame): Dataframe with date column
        date_col (str): Name of date column
        test_size (float): Fraction of data for test set
        
    Returns:
        tuple: (train_df, test_df)
    """
    if date_col is None:
        print_progress("No date column; using random split", level="WARN")
        return train_test_split(df, test_size=test_size, random_state=42)
    
    print_progress(f"Performing time-aware split on {date_col}")
    df_sorted = df.sort_values(date_col)
    n = len(df_sorted)
    split = int(n * (1 - test_size))
    train = df_sorted.iloc[:split]
    test = df_sorted.iloc[split:]
    print_progress(f"Train: {len(train)} rows, Test: {len(test)} rows")
    return train, test


def train_and_evaluate(X_train, y_train, X_test, y_test, out_dir='models'):
    """
    Train Logistic Regression and XGBoost models, evaluate on test set.
    Saves pipelines (model + imputer + scaler) to disk.
    
    Args:
        X_train (DataFrame): Training features
        y_train (Series): Training labels
        X_test (DataFrame): Test features
        y_test (Series): Test labels
        out_dir (str): Output directory for pipelines
        
    Returns:
        dict: Results with metrics for each model
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # impute + scale pipeline (manual)
    print_progress("Preprocessing: imputation and scaling")
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_train_imputed = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_imputed = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imputed)
    print_progress(f"Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")

    results = {}

    # ===== Logistic Regression (multinomial) =====
    print_progress("Training Logistic Regression...")
    logreg = LogisticRegression(
        multi_class='multinomial', 
        solver='lbfgs', 
        max_iter=1000, 
        random_state=42,
        verbose=0
    )
    logreg.fit(X_train_scaled, y_train)
    probs_lr = logreg.predict_proba(X_test_scaled)
    preds_lr = logreg.predict(X_test_scaled)
    
    lr_logloss = float(log_loss(y_test, probs_lr))
    lr_accuracy = float(accuracy_score(y_test, preds_lr))
    
    results['logreg'] = {
        'model': logreg,
        'logloss': lr_logloss,
        'accuracy': lr_accuracy,
        'classification_report': classification_report(y_test, preds_lr, output_dict=True)
    }
    
    lr_path = os.path.join(out_dir, 'logreg_pipeline.pkl')
    joblib.dump({'model': logreg, 'imputer': imputer, 'scaler': scaler}, lr_path)
    print_progress(f"Logistic Regression - Accuracy: {lr_accuracy:.4f}, LogLoss: {lr_logloss:.4f}")
    print_progress(f"Saved pipeline to {lr_path}")

    # ===== XGBoost =====
    try:
        import xgboost as xgb
        print_progress("Training XGBoost...")
        xgb_clf = xgb.XGBClassifier(
            objective='multi:softprob', 
            use_label_encoder=False, 
            eval_metric='mlogloss', 
            random_state=42,
            verbosity=0,
            n_jobs=-1
        )
        xgb_clf.fit(X_train_scaled, y_train)
        probs_xgb = xgb_clf.predict_proba(X_test_scaled)
        preds_xgb = xgb_clf.predict(X_test_scaled)
        
        xgb_logloss = float(log_loss(y_test, probs_xgb))
        xgb_accuracy = float(accuracy_score(y_test, preds_xgb))
        
        results['xgb'] = {
            'model': xgb_clf,
            'logloss': xgb_logloss,
            'accuracy': xgb_accuracy,
            'classification_report': classification_report(y_test, preds_xgb, output_dict=True)
        }
        
        xgb_path = os.path.join(out_dir, 'xgb_pipeline.pkl')
        joblib.dump({'model': xgb_clf, 'imputer': imputer, 'scaler': scaler}, xgb_path)
        print_progress(f"XGBoost - Accuracy: {xgb_accuracy:.4f}, LogLoss: {xgb_logloss:.4f}")
        print_progress(f"Saved pipeline to {xgb_path}")
    except Exception as e:
        results['xgb'] = {'error': str(e)}
        print_progress(f"XGBoost training failed: {str(e)}", level="ERROR")

    return results


def save_report(results, out_md='reports/model_report.md'):
    """
    Save model performance report to markdown.
    
    Args:
        results (dict): Model results from train_and_evaluate
        out_md (str): Output markdown path
    """
    print_progress(f"Writing report to {out_md}")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, 'w') as f:
        f.write('# Baseline Model Report\n\n')
        f.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        for k, v in results.items():
            f.write(f'## {k.upper()}\n')
            if 'error' in v:
                f.write('Error: ' + v['error'] + '\n')
                continue
            f.write(f"- logloss: {v['logloss']:.4f}\n")
            f.write(f"- accuracy: {v['accuracy']:.4f}\n")
            f.write('\n')
    print_progress(f"Report saved to {out_md}")


def main():
    """Main training pipeline."""
    import argparse
    
    print_progress("="*80)
    print_progress("UCL Match Outcome Predictor - Baseline Model Training")
    print_progress("="*80)
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Train baseline UEFA Champions League prediction models")
    parser.add_argument('--input', type=str, default='data/matches_with_features.csv', 
                       help='Path to merged matches CSV')
    args = parser.parse_args()
    
    try:
        # ===== LOAD =====
        df, date_col = load_data(args.input)
        
        # ===== FEATURE ENGINEERING =====
        df, feature_cols = build_features(df)
        if not feature_cols:
            raise RuntimeError('No delta features found to train on')
        
        # ===== TARGET =====
        print_progress("Building target variable...")
        y = build_target(df)
        print_progress(f"Class distribution - Away: {(y==0).sum()}, Draw: {(y==1).sum()}, Home: {(y==2).sum()}")
        
        # ===== PREPARE DATA =====
        print_progress("Preparing training data...")
        df_model = df[feature_cols + ([date_col] if date_col else [])].copy()
        df_model = df_model.reset_index(drop=True)
        # drop rows where all features are NaN
        df_model = df_model[df_model[feature_cols].notnull().any(axis=1)]
        y = y.loc[df_model.index]
        print_progress(f"After dropping NaN rows: {len(df_model)} samples")

        # ===== TRAIN/TEST SPLIT =====
        train_df, test_df = time_aware_split(
            pd.concat([df_model, y.rename('target')], axis=1), 
            date_col
        )
        X_train = train_df[feature_cols]
        y_train = train_df['target']
        X_test = test_df[feature_cols]
        y_test = test_df['target']

        # ===== TRAINING =====
        print_progress("="*80)
        results = train_and_evaluate(X_train, y_train, X_test, y_test)
        print_progress("="*80)
        
        # ===== SAVE RESULTS =====
        save_report(results)
        # save model features (ONLY feature names, not data)
        os.makedirs('data', exist_ok=True)
        pd.DataFrame({'feature': feature_cols}).to_csv('data/model_features.csv', index=False)
        print_progress(f"Saved {len(feature_cols)} feature names to data/model_features.csv")
        
        # ===== COMPLETION =====
        elapsed_time = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed_time)))
        print_progress("="*80)
        print_progress(f"✓ Training Completed Successfully in {elapsed_str}")
        print_progress("="*80)
        
    except Exception as e:
        print_progress(f"✗ Training Failed: {str(e)}", level="ERROR")
        print_progress("="*80)
        sys.exit(1)


if __name__ == '__main__':
    main()
