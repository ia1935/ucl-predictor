"""Baseline trainer: build delta features, train logistic regression and XGBoost.
Saves models to `models/` and writes a short report to `reports/model_report.md`.
"""
import os
from pathlib import Path
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


def load_data(path='data/matches_with_features.csv'):
    df = pd.read_csv(path)
    # parse date if present
    date_col = None
    for c in df.columns:
        if c.lower().startswith('date') or 'date_' in c.lower() or 'utc' in c.lower():
            date_col = c
            break
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    return df, date_col


def build_features(df):
    # identify numeric home/away prefixed columns
    home_cols = [c for c in df.columns if c.startswith('home_')]
    away_cols = [c for c in df.columns if c.startswith('away_')]
    # match suffixes
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

    return df, feature_cols


def build_target(df):
    # create multiclass target: 0=away,1=draw,2=home
    if 'fulltime_home' in df.columns and 'fulltime_away' in df.columns:
        y = df[['fulltime_home', 'fulltime_away']].apply(lambda x: 2 if x['fulltime_home']>x['fulltime_away'] else (1 if x['fulltime_home']==x['fulltime_away'] else 0), axis=1)
        return y
    # fallback to match_outcome strings
    if 'match_outcome' in df.columns:
        mapping = {'Away Win':0, 'Draw':1, 'Home Win':2}
        return df['match_outcome'].map(mapping).fillna(1).astype(int)
    raise RuntimeError('No suitable target columns found')


def time_aware_split(df, date_col, test_size=0.2):
    if date_col is None:
        return train_test_split(df, test_size=test_size, random_state=42)
    df_sorted = df.sort_values(date_col)
    n = len(df_sorted)
    split = int(n * (1 - test_size))
    train = df_sorted.iloc[:split]
    test = df_sorted.iloc[split:]
    return train, test


def train_and_evaluate(X_train, y_train, X_test, y_test, out_dir='models'):
    os.makedirs(out_dir, exist_ok=True)
    # impute + scale pipeline (manual)
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_train_imputed = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_imputed = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imputed)

    results = {}

    # Logistic Regression (multinomial)
    logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
    logreg.fit(X_train_scaled, y_train)
    probs_lr = logreg.predict_proba(X_test_scaled)
    preds_lr = logreg.predict(X_test_scaled)
    results['logreg'] = {
        'model': logreg,
        'logloss': float(log_loss(y_test, probs_lr)),
        'accuracy': float(accuracy_score(y_test, preds_lr)),
        'classification_report': classification_report(y_test, preds_lr, output_dict=True)
    }
    joblib.dump({'model': logreg, 'imputer': imputer, 'scaler': scaler}, os.path.join(out_dir, 'logreg_pipeline.pkl'))

    # XGBoost
    try:
        import xgboost as xgb
        xgb_clf = xgb.XGBClassifier(objective='multi:softprob', use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        xgb_clf.fit(X_train_scaled, y_train)
        probs_xgb = xgb_clf.predict_proba(X_test_scaled)
        preds_xgb = xgb_clf.predict(X_test_scaled)
        results['xgb'] = {
            'model': xgb_clf,
            'logloss': float(log_loss(y_test, probs_xgb)),
            'accuracy': float(accuracy_score(y_test, preds_xgb)),
            'classification_report': classification_report(y_test, preds_xgb, output_dict=True)
        }
        joblib.dump({'model': xgb_clf, 'imputer': imputer, 'scaler': scaler}, os.path.join(out_dir, 'xgb_pipeline.pkl'))
    except Exception as e:
        results['xgb'] = {'error': str(e)}

    return results


def save_report(results, out_md='reports/model_report.md'):
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, 'w') as f:
        f.write('# Baseline Model Report\n\n')
        for k, v in results.items():
            f.write(f'## {k}\n')
            if 'error' in v:
                f.write('Error: ' + v['error'] + '\n')
                continue
            f.write(f"- logloss: {v['logloss']:.4f}\n")
            f.write(f"- accuracy: {v['accuracy']:.4f}\n")
            f.write('\n')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/matches_with_features.csv', help='Path to merged matches CSV')
    args = parser.parse_args()
    df, date_col = load_data(args.input)
    df, feature_cols = build_features(df)
    if not feature_cols:
        raise RuntimeError('No delta features found to train on')
    y = build_target(df)
    df_model = df[feature_cols + [date_col] if date_col else feature_cols].copy()
    df_model = df_model.reset_index(drop=True)
    # drop rows where all features are NaN
    df_model = df_model[df_model[feature_cols].notnull().any(axis=1)]
    y = y.loc[df_model.index]

    # time-aware split
    train_df, test_df = time_aware_split(pd.concat([df_model, y.rename('target')], axis=1), date_col)
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']

    results = train_and_evaluate(X_train, y_train, X_test, y_test)
    save_report(results)
    # save model features
    os.makedirs('data', exist_ok=True)
    df_model.to_csv('data/model_features.csv', index=False)
    print('Training complete. Report written to reports/model_report.md')


if __name__ == '__main__':
    main()
