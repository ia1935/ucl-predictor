#!/usr/bin/env python3
"""Create a Champions League-specific merged matches file by mapping team_features -> team names.

Produces: `data/matches_with_features_ucl.csv` containing only UCL matches where both teams have features.
"""
import pandas as pd
import os


def main():
    matches = pd.read_csv('data/football_matches_2024_2025.csv')
    team_features = pd.read_csv('data/team_features.csv')
    teams = pd.read_csv('data/archive(1)/teams_data.csv')

    # build id -> name map from teams_data.csv
    id_to_name = dict(zip(teams['team_id'].astype(str), teams['team']))
    team_features['team_name'] = team_features['team'].astype(str).map(id_to_name)

    # focus on UEFA Champions League matches
    ucl = matches[matches['competition_name']=='UEFA Champions League'].copy()

    # perform name-based join: team_features.team_name -> matches.home_team / away_team
    tf_home = team_features.copy()
    tf_away = team_features.copy()
    tf_home = tf_home.add_prefix('home_')
    tf_away = tf_away.add_prefix('away_')

    merged = ucl.merge(tf_home, left_on='home_team', right_on='home_team_name', how='left')
    merged = merged.merge(tf_away, left_on='away_team', right_on='away_team_name', how='left')

    # keep only rows where both sides have features
    numeric_home = [c for c in merged.columns if c.startswith('home_') and merged[c].dtype in [float, int]]
    # fallback: check whether mapped name columns are non-null
    mask = merged['home_team_name'].notnull() & merged['away_team_name'].notnull()
    filtered = merged[mask].copy()

    out = 'data/matches_with_features_ucl.csv'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    filtered.to_csv(out, index=False)
    print('Saved UCL merged matches to', out, 'rows:', len(filtered))


if __name__ == '__main__':
    main()
