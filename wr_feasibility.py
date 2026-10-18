# -*- coding: utf-8 -*-
"""Quick feasibility check for WR route-running direction."""
import pandas as pd

BASE = r"D:\kaggle\NFL2027\nfl-big-data-bowl-2027"

players = pd.read_csv(BASE + r"\players.csv", low_memory=False)
career = pd.read_csv(BASE + r"\player_career_successes.csv", low_memory=False)
pp = pd.read_csv(BASE + r"\player_play.csv", low_memory=False)
ct = pd.read_csv(BASE + r"\combine_tracking.csv", low_memory=False)

wr_ids = set(players.loc[players["nfl_position"] == "WR", "nfl_id"])
wr_career = career[career["nfl_id"].isin(wr_ids)]
print("WR count:", len(wr_ids))
print("WR career games_started stats:")
print(wr_career[["career_games_started", "career_games_active"]].describe().round(1).to_string())

# combine tracking: WR skill drills
ct_wr = ct[(ct["drill_type"] == "SKILL_DRILLS_WR") | (ct["drill_type"] == "FORTY_YARD_DASH")]
print("\ncombine_tracking WR-related rows:", len(ct_wr))
print("WR skill drill rows:", (ct["drill_type"] == "SKILL_DRILLS_WR").sum())
print("BALL frames in WR drills:", ((ct["drill_type"] == "SKILL_DRILLS_WR") & (ct["entity_type"] == "BALL")).sum())
print("unique WR-skill events:", ct.loc[ct["drill_type"] == "SKILL_DRILLS_WR", "event_id"].nunique())

# player_play for WRs
pp_wr = pp[pp["nfl_id"].isin(wr_ids)]
print("\nplayer_play rows for WR cohort:", len(pp_wr))
print("lined_up_position WR/SLOT:", pp_wr["lined_up_position"].value_counts(dropna=False).head(3).to_dict())
print("route_ran value_counts:")
print(pp_wr["route_ran"].value_counts(dropna=False).head(12).to_string())
print("\nseparation_at_pass_forward non-null:", pp_wr["separation_at_pass_forward"].notna().sum())
print("target==True rows:", pp_wr["target"].sum())
print("rec_yards non-null:", pp_wr["rec_yards"].notna().sum())
print("separation stats (WR cohort):")
print(pp_wr["separation_at_pass_forward"].describe().round(2).to_string())

# route drill names in combine (WR)
names = ct.loc[ct["drill_type"] == "SKILL_DRILLS_WR", "drill_name"].value_counts()
print("\nWR skill drill names (top 20):")
print(names.head(20).to_string())

# how many WR have separation data per draft year
pp_wr2 = pp_wr[pp_wr["separation_at_pass_forward"].notna()].merge(players[["nfl_id", "draft_year"]], on="nfl_id")
print("\nWRs with separation data by draft year:")
print(pp_wr2.groupby("draft_year")["nfl_id"].nunique().to_string())
