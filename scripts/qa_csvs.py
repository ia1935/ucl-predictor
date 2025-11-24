#!/usr/bin/env python3
"""QA CSVs and produce a markdown report + csv summary + small figures.

Usage: .venv/bin/python3 scripts/qa_csvs.py
"""
import os
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FILES = [
    'data/matches_with_features.csv',
    'data/team_features.csv',
    'data/football_matches_2024_2025.csv',
    'data/archive(1)/DAY_4/players_data.csv',
    'data/archive(1)/teams_data.csv'
]


def summarize_df(path):
    df = pd.read_csv(path)
    summary = {}
    summary['file'] = path
    summary['rows'] = df.shape[0]
    summary['cols'] = df.shape[1]
    # simple sample
    summary['columns'] = ','.join(list(df.columns))
    # top null columns
    nulls = (df.isnull().sum() / max(1, len(df)) * 100).sort_values(ascending=False)
    top_nulls = ';'.join([f"{c}:{round(pct,2)}%" for c, pct in nulls[nulls>0].head(10).items()])
    summary['top_nulls_pct'] = top_nulls
    # dtypes
    summary['dtypes'] = ','.join([f"{c}:{str(t)}" for c, t in df.dtypes.items()])
    # basic detection for id and team columns
    for c in ['match_id','home_team','away_team','home_team_id','away_team_id','team','id_team']:
        if c in df.columns:
            try:
                summary[f'unique_{c}'] = df[c].nunique()
                sample_vals = df[c].dropna().unique()[:5].tolist()
                summary[f'sample_{c}'] = '|'.join([str(x) for x in sample_vals])
            except Exception:
                pass
    return df, summary, nulls


def write_markdown(report_path, summaries, samples):
    with open(report_path, 'w') as f:
        f.write('# Data QA Report\n')
        f.write('\n')
        for s in summaries:
            f.write(f"## {s['file']}\n\n")
            f.write(f"- rows: {s['rows']}\n")
            f.write(f"- cols: {s['cols']}\n")
            f.write(f"- top nulls: {s.get('top_nulls_pct','')}\n")
            f.write(f"- sample ids: {s.get('sample_match_id','')}\n")
            f.write('\n')
            f.write('Sample rows:\n\n')
            df = samples[s['file']]
            f.write('---\n')
            f.write(df.to_string(index=False, max_rows=10))
            f.write('\n---\n\n')


def make_null_plot(nulls, out_png):
    plt.figure(figsize=(6,3))
    top = nulls[nulls>0].head(30)
    top.plot(kind='barh')
    plt.tight_layout()
    plt.xlabel('Percent null')
    plt.savefig(out_png)
    plt.close()


def main():
    os.makedirs('reports/figs', exist_ok=True)
    summaries = []
    samples = {}
    rows = []
    for p in FILES:
        if not Path(p).exists():
            print('missing:', p)
            continue
        df, summary, nulls = summarize_df(p)
        summaries.append(summary)
        samples[p] = df.head(5)
        rows.append(summary)
        # make plot
        png = f"reports/figs/{Path(p).stem}_nulls.png"
        try:
            make_null_plot(nulls, png)
        except Exception:
            pass

    # write CSV summary
    if rows:
        pd.DataFrame(rows).to_csv('reports/qa_summary.csv', index=False)
    # write markdown report
    write_markdown('reports/qa_report.md', summaries, samples)
    print('QA report written to reports/qa_report.md and reports/qa_summary.csv')


if __name__ == '__main__':
    main()
