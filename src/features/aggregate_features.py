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

    club_col = safe_col(players, ["club", "team_name", "team", "teamname"]) or "club"

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
    matches = matches_df.copy()
    # try common match column names
    home_col = safe_col(matches, ["home_team", "home", "home_club"]) or "home_team"
    away_col = safe_col(matches, ["away_team", "away", "away_club"]) or "away_team"

    matches = matches.merge(team_features.add_prefix("home_"), left_on=home_col, right_on="home_team", how="left")
    matches = matches.merge(team_features.add_prefix("away_"), left_on=away_col, right_on="away_team", how="left")
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
    players = pd.read_csv(players_csv)

    team_features = aggregate_players(players)
    os.makedirs(os.path.dirname(parsed.out_team), exist_ok=True)
    team_features.to_csv(parsed.out_team, index=False)
    print("Saved team features to:", parsed.out_team)

    # merge with matches if provided
    if parsed.matches and os.path.exists(parsed.matches):
        print("Loading matches from:", parsed.matches)
        matches = pd.read_csv(parsed.matches, parse_dates=[safe_col(pd.read_csv(parsed.matches, nrows=1), ["date"]) or "date"], infer_datetime_format=True)
        merged = merge_team_features_into_matches(matches, team_features)
        os.makedirs(os.path.dirname(parsed.out_matches), exist_ok=True)
        merged.to_csv(parsed.out_matches, index=False)
        print("Saved merged matches to:", parsed.out_matches)
    else:
        print("Matches file not found or not provided; skipping merge.")


if __name__ == "__main__":
    main()