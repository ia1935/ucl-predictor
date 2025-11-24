"""Compute additional team features: per-90 rates, recent-form rolling stats, and a simple Elo rating.
Produces `data/matches_with_features_ucl_enriched.csv`.
"""
import pandas as pd
import numpy as np
import os


def compute_per90(df):
    # expects home_/away_ prefixed numeric columns: team_goals, team_assists, team_minutes, passes_attempted, passes_completed
    out = df.copy()
    for side in ['home', 'away']:
        mins_col = f'{side}_team_minutes'
        if mins_col in out.columns:
            denom = out[mins_col] / 90.0
            denom = denom.replace(0, np.nan)
            for base in ['team_goals', 'team_assists', 'team_passes_attempted', 'team_passes_completed', 'team_distance_covered']:
                col = f'{side}_{base}'
                if col in out.columns:
                    out[f'{col}_per90'] = out[col] / denom
    return out


def build_long_matches(matches):
    # convert matches to long format per team to compute rolling stats
    rows = []
    for _, r in matches.iterrows():
        date = pd.to_datetime(r.get('date_utc', r.get('date', None)), errors='coerce')
        if pd.isna(date):
            continue
        # home row
        rows.append({
            'match_id': r['match_id'], 'team_id': int(r['home_team_id']), 'team_name': r.get('home_team', None),
            'is_home': 1, 'date': date, 'goals_for': r.get('fulltime_home', np.nan), 'goals_against': r.get('fulltime_away', np.nan),
            'points': r.get('home_points', np.nan)
        })
        # away row
        rows.append({
            'match_id': r['match_id'], 'team_id': int(r['away_team_id']), 'team_name': r.get('away_team', None),
            'is_home': 0, 'date': date, 'goals_for': r.get('fulltime_away', np.nan), 'goals_against': r.get('fulltime_home', np.nan),
            'points': r.get('away_points', np.nan)
        })
    long = pd.DataFrame(rows)
    long = long.sort_values(['team_id', 'date'])
    return long


def compute_recent_form(matches_df, full_matches_csv, windows=(3,5)):
    full = pd.read_csv(full_matches_csv)
    full['date_utc'] = pd.to_datetime(full['date_utc'], errors='coerce')
    long = build_long_matches(full)
    # compute rolling sums of points and goal_diff excluding current match (shift)
    long['goal_diff'] = long['goals_for'] - long['goals_against']
    out_rows = []
    grp = long.groupby('team_id')
    rolled = []
    for team, g in grp:
        g = g.sort_values('date').copy()
        for w in windows:
            g[f'last{w}_points'] = g['points'].shift(1).rolling(window=w, min_periods=1).sum()
            g[f'last{w}_gd'] = g['goal_diff'].shift(1).rolling(window=w, min_periods=1).sum()
        rolled.append(g)
    long2 = pd.concat(rolled)
    # pivot back to matches: for each match_id, extract home/away team's recent form
    recent = long2[['match_id','team_id'] + [c for c in long2.columns if c.startswith('last')]]
    recent_home = recent.merge(matches_df[['match_id','home_team_id']], left_on=['match_id','team_id'], right_on=['match_id','home_team_id'], how='inner')
    recent_away = recent.merge(matches_df[['match_id','away_team_id']], left_on=['match_id','team_id'], right_on=['match_id','away_team_id'], how='inner')
    recent_home = recent_home.set_index('match_id')[[c for c in recent.columns if c.startswith('last')]].add_prefix('home_')
    recent_away = recent_away.set_index('match_id')[[c for c in recent.columns if c.startswith('last')]].add_prefix('away_')
    merged = matches_df.set_index('match_id')
    merged = merged.join(recent_home, how='left')
    merged = merged.join(recent_away, how='left')
    merged = merged.reset_index()
    return merged


def compute_elo(full_matches_csv, k=20, home_adv=100):
    full = pd.read_csv(full_matches_csv)
    full['date_utc'] = pd.to_datetime(full['date_utc'], errors='coerce')
    full = full.sort_values('date_utc').copy()
    teams = pd.unique(pd.concat([full['home_team_id'], full['away_team_id']]).dropna()).tolist()
    elo = {int(t): 1500.0 for t in teams}
    records = []
    for _, r in full.iterrows():
        hid = int(r['home_team_id'])
        aid = int(r['away_team_id'])
        eh = elo.get(hid, 1500.0)
        ea = elo.get(aid, 1500.0)
        # expected
        exp_h = 1 / (1 + 10 ** (((ea) - (eh + home_adv)) / 400))
        exp_a = 1 - exp_h
        # result
        if r['fulltime_home'] > r['fulltime_away']:
            sh, sa = 1.0, 0.0
        elif r['fulltime_home'] == r['fulltime_away']:
            sh, sa = 0.5, 0.5
        else:
            sh, sa = 0.0, 1.0
        # record elo before match
        records.append({'match_id': r['match_id'], 'home_team_id': hid, 'away_team_id': aid, 'home_elo_before': eh, 'away_elo_before': ea, 'date_utc': r['date_utc']})
        # update
        elo[hid] = eh + k * (sh - exp_h)
        elo[aid] = ea + k * (sa - exp_a)
    return pd.DataFrame(records)


def main():
    # load UCL merged matches
    ucl = pd.read_csv('data/matches_with_features_ucl.csv')
    # per-90 on existing prefixed numeric columns
    ucl = compute_per90(ucl)
    # recent form from full matches
    ucl = compute_recent_form(ucl, 'data/football_matches_2024_2025.csv')
    # elo
    elo_df = compute_elo('data/football_matches_2024_2025.csv')
    elo_df = elo_df.set_index('match_id')
    # join elo before values to UCL matches by match_id
    ucl = ucl.set_index('match_id')
    ucl = ucl.join(elo_df[['home_elo_before','away_elo_before']], how='left')
    ucl = ucl.reset_index()
    out = 'data/matches_with_features_ucl_enriched.csv'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ucl.to_csv(out, index=False)
    print('Wrote enriched UCL matches to', out)


if __name__ == '__main__':
    main()
