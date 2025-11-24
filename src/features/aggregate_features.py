import os
import glob
import argparse
import pandas as pd
import numpy as np


def download_kaggle_dataset(dataset_id="pabloramoswilkins/ucl-2025-players-data"):
    try:
        import kagglehub
    except Exception:
        raise ImportError("kagglehub is required to download the dataset. Install it with `pip install kagglehub`.")
    path = kagglehub.dataset_download(dataset_id)
    print("Downloaded dataset to:", path)
    return path


def find_csv(base_path, candidates=None):
    if candidates is None:
        candidates = ["players.csv", "players_data.csv", "players_stats.csv", "ucl_players.csv"]
    # if base_path is a folder, search inside
    if os.path.isdir(base_path):
        for c in candidates:
            p = os.path.join(base_path, c)
            if os.path.exists(p):
                return p
        # fallback: any csv in folder
        files = glob.glob(os.path.join(base_path, "*.csv"))
        return files[0] if files else None
    # if base_path is a file
    if os.path.isfile(base_path) and base_path.lower().endswith(".csv"):
        return base_path
    return None


def safe_col(df, possible):
    for p in possible:
        if p in df.columns:
            return p
    return None


def aggregate_players(players_df):
    # normalize column names
    players = players_df.copy()
    players.columns = [c.strip() for c in players.columns]

    # prefer explicit team id/name columns commonly found in the UCL dataset
    club_col = safe_col(players, ["club", "team_name", "team", "teamname", "id_team"]) or "club"

    # helper to map common stat names to dataset columns
    col_map = {
        "goals": safe_col(players, ["goals", "goals_scored", "total_goals"]),
        "assists": safe_col(players, ["assists", "assist"]),
        "minutes_played": safe_col(players, ["minutes_played", "minutes", "mins_played"]),
        "passes_attempted": safe_col(players, ["passes_attempted", "passes_attempted_total", "passes"]),
        "passes_completed": safe_col(players, ["passes_completed", "passes_success", "passes_completed_total"]),
        "top_speed": safe_col(players, ["top_speed(km/h)", "top_speed_km_h", "top_speed"]),
        "distance_covered": safe_col(players, ["distance_covered(km)", "distance_covered_km", "distance_covered"]),
        "passing_accuracy": safe_col(players, ["passing_accuracy(%)", "passing_accuracy", "pass_accuracy"]),
        "age": safe_col(players, ["age"]),
    }

    # Basic per-team aggregates (season-level)
    agg_dict = {}
    if col_map["goals"]:
        agg_dict[col_map["goals"]] = "sum"
    if col_map["assists"]:
        agg_dict[col_map["assists"]] = "sum"
    if col_map["minutes_played"]:
        agg_dict[col_map["minutes_played"]] = "sum"
    if col_map["passes_attempted"]:
        agg_dict[col_map["passes_attempted"]] = "sum"
    if col_map["passes_completed"]:
        agg_dict[col_map["passes_completed"]] = "sum"
    if col_map["top_speed"]:
        agg_dict[col_map["top_speed"]] = "mean"
    if col_map["distance_covered"]:
        agg_dict[col_map["distance_covered"]] = "sum"

    team_agg = players.groupby(club_col).agg(agg_dict)
    # rename aggregated columns to consistent names
    rename_map = {}
    if col_map["goals"]:
        rename_map[col_map["goals"]] = "team_goals"
    if col_map["assists"]:
        rename_map[col_map["assists"]] = "team_assists"
    if col_map["minutes_played"]:
        rename_map[col_map["minutes_played"]] = "team_minutes"
    if col_map["top_speed"]:
        rename_map[col_map["top_speed"]] = "team_avg_top_speed"
    if col_map["distance_covered"]:
        rename_map[col_map["distance_covered"]] = "team_distance_covered"
    if col_map["passes_attempted"]:
        rename_map[col_map["passes_attempted"]] = "team_passes_attempted"
    if col_map["passes_completed"]:
        rename_map[col_map["passes_completed"]] = "team_passes_completed"

    team_agg = team_agg.rename(columns=rename_map)

    # pass acc average
    if col_map["passing_accuracy"]:
        team_agg["team_pass_acc"] = players.groupby(club_col)[col_map["passing_accuracy"]].mean()

    # Estimate top-11 by minutes
    def top_n_by_minutes(df, n=11):
        mcol = col_map.get("minutes_played")
        if mcol and mcol in df.columns:
            top = df.sort_values(mcol, ascending=False).head(n)
        else:
            top = df.head(n)
        out = {}
        if col_map.get("goals") in df.columns:
            out["top11_goals"] = top[col_map["goals"]].sum()
        else:
            out["top11_goals"] = np.nan
        if col_map.get("assists") in df.columns:
            out["top11_assists"] = top[col_map["assists"]].sum()
        else:
            out["top11_assists"] = np.nan
        if col_map.get("passing_accuracy") in df.columns:
            out["top11_avg_pass_acc"] = top[col_map["passing_accuracy"]].mean()
        else:
            out["top11_avg_pass_acc"] = np.nan
        if col_map.get("age") in df.columns:
            out["top11_avg_age"] = top[col_map["age"]].mean()
        else:
            out["top11_avg_age"] = np.nan
        return pd.Series(out)

    top11 = players.groupby(club_col).apply(top_n_by_minutes)

    team_features = team_agg.join(top11, how="left")
    team_features = team_features.reset_index().rename(columns={club_col: "team"})
    return team_features


def merge_team_features_into_matches(matches_df, team_features):
    import glob
    matches = matches_df.copy()
    # prefer numeric id-based joins when available (home_team_id / away_team_id)
    home_id_col = safe_col(matches, ["home_team_id", "home_id", "home_club_id"])
    away_id_col = safe_col(matches, ["away_team_id", "away_id", "away_club_id"])

    # try to enrich team_features with textual team names if a teams CSV exists
    team_features = team_features.copy()
    teams_files = glob.glob(os.path.join("data", "**", "teams*.csv"), recursive=True)
    if teams_files:
        try:
            teams_map = pd.read_csv(teams_files[0])
            if "team_id" in teams_map.columns and "team" in teams_map.columns:
                id_to_name = teams_map.set_index("team_id")["team"].to_dict()
                team_features["team_name"] = team_features["team"].map(id_to_name)
        except Exception:
            pass

    if home_id_col and away_id_col and "team" in team_features.columns:
        # coerce to string to avoid int/object mismatches
        matches[home_id_col] = matches[home_id_col].astype(str)
        matches[away_id_col] = matches[away_id_col].astype(str)

        tf_home = team_features.copy()
        tf_away = team_features.copy()
        tf_home["team"] = tf_home["team"].astype(str)
        tf_away["team"] = tf_away["team"].astype(str)

        matches = matches.merge(tf_home.add_prefix("home_"), left_on=home_id_col, right_on="home_team", how="left")
        matches = matches.merge(tf_away.add_prefix("away_"), left_on=away_id_col, right_on="away_team", how="left")
        return matches

    # fallback: name-based join
    home_col = safe_col(matches, ["home_team", "home", "home_club"]) or "home_team"
    away_col = safe_col(matches, ["away_team", "away", "away_club"]) or "away_team"

    if home_col in matches.columns:
        matches[home_col] = matches[home_col].astype(str)
    if away_col in matches.columns:
        matches[away_col] = matches[away_col].astype(str)

    tf_home = team_features.copy()
    tf_away = team_features.copy()

    # if we have a mapped textual name, prefer that for joining
    if "team_name" in tf_home.columns and tf_home["team_name"].notnull().any():
        tf_home["join_name"] = tf_home["team_name"].astype(str)
        tf_away["join_name"] = tf_away["team_name"].astype(str)
        matches = matches.merge(tf_home.add_prefix("home_"), left_on=home_col, right_on="home_join_name", how="left")
        matches = matches.merge(tf_away.add_prefix("away_"), left_on=away_col, right_on="away_join_name", how="left")
        return matches

    # last resort: stringified numeric team id
    tf_home["team"] = tf_home["team"].astype(str)
    tf_away["team"] = tf_away["team"].astype(str)
    matches = matches.merge(tf_home.add_prefix("home_"), left_on=home_col, right_on="home_team", how="left")
    matches = matches.merge(tf_away.add_prefix("away_"), left_on=away_col, right_on="away_team", how="left")
    return matches


def main(args=None):
    parser = argparse.ArgumentParser(description="Aggregate player-level UCL data into team features and merge with matches.")
    parser.add_argument("--download", action="store_true", help="Download dataset from Kaggle using kagglehub")
    parser.add_argument("--players", type=str, default="data/players.csv", help="Path to players csv or folder")
    parser.add_argument("--matches", type=str, default="data/matches.csv", help="Path to matches csv")
    parser.add_argument("--out_team", type=str, default="data/team_features.csv", help="Output team features csv")
    parser.add_argument("--out_matches", type=str, default="data/matches_with_features.csv", help="Output merged matches csv")
    parsed = parser.parse_args(args=args)

    players_path = parsed.players
    if parsed.download:
        downloaded = download_kaggle_dataset()
        csv = find_csv(downloaded)
        if csv:
            players_path = csv
        else:
            print("Downloaded dataset but could not find a CSV inside:", downloaded)

    players_csv = find_csv(players_path) or players_path
    if not players_csv or not os.path.exists(players_csv):
        raise FileNotFoundError(f"Players CSV not found at '{players_csv}'")

    print("Loading players from:", players_csv)
    # load base players file and try to merge any per-player stat files that live in the same folder
    players = pd.read_csv(players_csv)
    base_dir = os.path.dirname(players_csv)
    # merge additional CSVs by `id_player` if present (e.g. goals_data.csv, attempts_data.csv, attacking_data.csv)
    for f in glob.glob(os.path.join(base_dir, "*.csv")):
        if os.path.abspath(f) == os.path.abspath(players_csv):
            continue
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if "id_player" in df.columns:
            # avoid duplicating player name/id columns
            drop_cols = [c for c in ["player_name", "player_image", "nationality"] if c in df.columns]
            try:
                players = players.merge(df.drop(columns=drop_cols, errors="ignore"), on="id_player", how="left")
                print("Merged stats from:", f)
            except Exception:
                # fallback: skip problematic merges
                print("Skipping merge for:", f)

    team_features = aggregate_players(players)
    os.makedirs(os.path.dirname(parsed.out_team), exist_ok=True)
    team_features.to_csv(parsed.out_team, index=False)
    print("Saved team features to:", parsed.out_team)

    # merge with matches if provided
    if parsed.matches and os.path.exists(parsed.matches):
        print("Loading matches from:", parsed.matches)
        # safely detect a date column without forcing parse_dates to a missing name
        preview = pd.read_csv(parsed.matches, nrows=1)
        date_col = safe_col(preview, ["date", "match_date", "kickoff", "kickoff_time", "datetime"]) 
        if date_col:
            matches = pd.read_csv(parsed.matches, parse_dates=[date_col])
        else:
            matches = pd.read_csv(parsed.matches)
        merged = merge_team_features_into_matches(matches, team_features)
        os.makedirs(os.path.dirname(parsed.out_matches), exist_ok=True)
        merged.to_csv(parsed.out_matches, index=False)
        print("Saved merged matches to:", parsed.out_matches)
        # produce a filtered version keeping only rows where both home and away team ids exist in team_features
        try:
            team_ids = set(team_features['team'].astype(int).tolist())
            # prefer numeric id columns in matches
            hid = safe_col(matches, ['home_team_id', 'home_id', 'home_club_id'])
            aid = safe_col(matches, ['away_team_id', 'away_id', 'away_club_id'])
            if hid and aid:
                mask = matches[hid].isin(team_ids) & matches[aid].isin(team_ids)
                filtered = merged[mask].copy()
                out_filtered = parsed.out_matches.replace('.csv', '_filtered.csv')
                filtered.to_csv(out_filtered, index=False)
                print(f"Saved filtered merged matches to: {out_filtered} (rows kept: {len(filtered)} of {len(merged)})")
            else:
                print('Match id columns not found; skipping filtered output.')
        except Exception:
            print('Could not create filtered matches file due to id coercion error; skipping.')
    else:
        print("Matches file not found or not provided; skipping merge.")


if __name__ == "__main__":
    main()