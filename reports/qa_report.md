# Data QA Report

## data/matches_with_features.csv

- rows: 1941
- cols: 49
- top nulls: away_team_passes_completed:100.0%;home_top11_avg_pass_acc:100.0%;home_top11_avg_age:100.0%;home_team_name:100.0%;away_team_y:100.0%;away_team_goals:100.0%;away_team_assists:100.0%;home_top11_assists:100.0%;home_top11_goals:100.0%;home_team_assists:100.0%
- sample ids: 497410|497411|497412|497413|497414

Sample rows:

---
competition_code competition_name    season  match_id  matchday          stage   status                  date_utc        referee  home_team_id          home_team_x  away_team_id                away_team_x  fulltime_home  fulltime_away  halftime_home  halftime_away  goal_difference  total_goals match_outcome  home_points  away_points  referee_id  home_team_y  home_team_goals  home_team_assists  home_team_minutes  home_team_passes_attempted  home_team_passes_completed  home_team_avg_top_speed  home_team_pass_acc  home_top11_goals  home_top11_assists  home_top11_avg_pass_acc  home_top11_avg_age  home_team_name  away_team_y  away_team_goals  away_team_assists  away_team_minutes  away_team_passes_attempted  away_team_passes_completed  away_team_avg_top_speed  away_team_pass_acc  away_top11_goals  away_top11_assists  away_top11_avg_pass_acc  away_top11_avg_age  away_team_name
              PL   Premier League 2024/2025    497410         1 REGULAR_SEASON FINISHED 2024-08-16 19:00:00+00:00   Robert Jones            66 Manchester United FC            63                  Fulham FC              1              0              0              0                1            1      Home Win            3            0           1          NaN              NaN                NaN                NaN                         NaN                         NaN                      NaN                 NaN               NaN                 NaN                      NaN                 NaN             NaN          NaN              NaN                NaN                NaN                         NaN                         NaN                      NaN                 NaN               NaN                 NaN                      NaN                 NaN             NaN
              PL   Premier League 2024/2025    497411         1 REGULAR_SEASON FINISHED 2024-08-17 11:30:00+00:00   Tim Robinson           349      Ipswich Town FC            64               Liverpool FC              0              2              0              0               -2            2      Away Win            0            3           2          NaN              NaN                NaN                NaN                         NaN                         NaN                      NaN                 NaN               NaN                 NaN                      NaN                 NaN             NaN          NaN              NaN                NaN                NaN                         NaN                         NaN                      NaN                 NaN               NaN                 NaN                      NaN                 NaN             NaN
              PL   Premier League 2024/2025    497412         1 REGULAR_SEASON FINISHED 2024-08-17 14:00:00+00:00 Jarred Gillett            57           Arsenal FC            76 Wolverhampton Wanderers FC              2              0              1              0                2            2      Home Win            3            0           3          NaN              NaN                NaN                NaN                         NaN                         NaN                      NaN                 NaN               NaN                 NaN                      NaN                 NaN             NaN          NaN              NaN                NaN                NaN                         NaN                         NaN                      NaN                 NaN               NaN                 NaN                      NaN                 NaN             NaN
              PL   Premier League 2024/2025    497413         1 REGULAR_SEASON FINISHED 2024-08-17 14:00:00+00:00   Simon Hooper            62           Everton FC           397  Brighton & Hove Albion FC              0              3              0              1               -3            3      Away Win            0            3           4          NaN              NaN                NaN                NaN                         NaN                         NaN                      NaN                 NaN               NaN                 NaN                      NaN                 NaN             NaN          NaN              NaN                NaN                NaN                         NaN                         NaN                      NaN                 NaN               NaN                 NaN                      NaN                 NaN             NaN
              PL   Premier League 2024/2025    497414         1 REGULAR_SEASON FINISHED 2024-08-17 14:00:00+00:00   Craig Pawson            67  Newcastle United FC           340             Southampton FC              1              0              1              0                1            1      Home Win            3            0           5          NaN              NaN                NaN                NaN                         NaN                         NaN                      NaN                 NaN               NaN                 NaN                      NaN                 NaN             NaN          NaN              NaN                NaN                NaN                         NaN                         NaN                      NaN                 NaN               NaN                 NaN                      NaN                 NaN             NaN
---

## data/team_features.csv

- rows: 35
- cols: 12
- top nulls: 
- sample ids: 

Sample rows:

---
 team  team_goals  team_assists  team_minutes  team_passes_attempted  team_passes_completed  team_avg_top_speed  team_pass_acc  top11_goals  top11_assists  top11_avg_pass_acc  top11_avg_age
 7889        10.0           9.0        3963.0                 2005.0                 1735.0           31.505238      82.949048          9.0            8.0           85.818182      27.181818
50023        10.0           7.0        3963.0                 1957.0                 1671.0           31.611429      81.730952          8.0            7.0           82.380000      24.818182
50030         3.0           3.0        3940.0                 1946.0                 1675.0           30.824348      80.906522          2.0            3.0           82.364545      22.272727
50031         1.0           1.0        3963.0                 1476.0                 1244.0           31.228182      79.285455          1.0            0.0           81.084545      24.090909
50033         5.0           4.0        3960.0                 1076.0                  776.0           30.895455      67.944545          5.0            3.0           70.175455      25.545455
---

## data/football_matches_2024_2025.csv

- rows: 1941
- cols: 23
- top nulls: 
- sample ids: 497410|497411|497412|497413|497414

Sample rows:

---
competition_code competition_name    season  match_id  matchday          stage   status                  date_utc        referee  home_team_id            home_team  away_team_id                  away_team  fulltime_home  fulltime_away  halftime_home  halftime_away  goal_difference  total_goals match_outcome  home_points  away_points  referee_id
              PL   Premier League 2024/2025    497410         1 REGULAR_SEASON FINISHED 2024-08-16 19:00:00+00:00   Robert Jones            66 Manchester United FC            63                  Fulham FC              1              0              0              0                1            1      Home Win            3            0           1
              PL   Premier League 2024/2025    497411         1 REGULAR_SEASON FINISHED 2024-08-17 11:30:00+00:00   Tim Robinson           349      Ipswich Town FC            64               Liverpool FC              0              2              0              0               -2            2      Away Win            0            3           2
              PL   Premier League 2024/2025    497412         1 REGULAR_SEASON FINISHED 2024-08-17 14:00:00+00:00 Jarred Gillett            57           Arsenal FC            76 Wolverhampton Wanderers FC              2              0              1              0                2            2      Home Win            3            0           3
              PL   Premier League 2024/2025    497413         1 REGULAR_SEASON FINISHED 2024-08-17 14:00:00+00:00   Simon Hooper            62           Everton FC           397  Brighton & Hove Albion FC              0              3              0              1               -3            3      Away Win            0            3           4
              PL   Premier League 2024/2025    497414         1 REGULAR_SEASON FINISHED 2024-08-17 14:00:00+00:00   Craig Pawson            67  Newcastle United FC           340             Southampton FC              1              0              1              0                1            1      Home Win            3            0           5
---

## data/archive(1)/DAY_4/players_data.csv

- rows: 908
- cols: 10
- top nulls: weight(kg):83.04%;height(cm):80.29%;position:22.8%
- sample ids: 

Sample rows:

---
 id_player       player_name nationality field_position           position  weight(kg)  height(cm)  age  id_team                                                       player_image
 250016833        Harry Kane     England        Forward            STRIKER        65.0       188.0   31    50037 https://img.uefa.com/imgml/TP/players/1/2025/324x324/250016833.jpg
 250105927   Viktor Gyökeres      Sweden        Forward                NaN         NaN         NaN   26    50149 https://img.uefa.com/imgml/TP/players/1/2025/324x324/250105927.jpg
 250121533   Vinícius Júnior      Brazil        Forward            UNKNOWN         NaN         NaN   24    50051 https://img.uefa.com/imgml/TP/players/1/2025/324x324/250121533.jpg
 250121294 Tijjani Reijnders Netherlands     Midfielder CENTRAL_MIDFIELDER         NaN         NaN   26    50058 https://img.uefa.com/imgml/TP/players/1/2025/324x324/250121294.jpg
 250160436 Maghnes Akliouche      France     Midfielder            UNKNOWN         NaN         NaN   22    50023 https://img.uefa.com/imgml/TP/players/1/2025/324x324/250160436.jpg
---

## data/archive(1)/teams_data.csv

- rows: 36
- cols: 4
- top nulls: 
- sample ids: 

Sample rows:

---
 team_id  country                     team                                                      logo
   50138    Italy FC Internazionale Milano https://img.uefa.com/imgml/TP/teams/logos/70x70/50138.png
   50124    Spain       Atlético de Madrid https://img.uefa.com/imgml/TP/teams/logos/70x70/50124.png
   50111  Austria            SK Sturm Graz https://img.uefa.com/imgml/TP/teams/logos/70x70/50111.png
   52816    Italy              Atalanta BC https://img.uefa.com/imgml/TP/teams/logos/70x70/52816.png
   50050 Scotland                Celtic FC https://img.uefa.com/imgml/TP/teams/logos/70x70/50050.png
---

