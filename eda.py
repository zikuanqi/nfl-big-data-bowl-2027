# -*- coding: utf-8 -*-
"""NFL Big Data Bowl 2027 - EDA script (memory-efficient)."""
import pandas as pd
import numpy as np
import os
import time

BASE = r"D:\kaggle\NFL2027\nfl-big-data-bowl-2027"
OUT = r"D:\kaggle\NFL2027\eda_report.txt"

buf = []


def log(*a):
    s = " ".join(str(x) for x in a)
    buf.append(s)
    print(s, flush=True)


t0 = time.time()


def stamp(msg):
    log(f"\n[{time.time() - t0:7.1f}s] ==== {msg} ====")


def count_lines_raw(path):
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024 * 1024), b""):
            n += chunk.count(b"\n")
    return n - 1  # minus header row


def summarize_df(df, name, topn=10):
    log(f"\n===== {name} | shape {df.shape} =====")
    log("columns: " + ", ".join(df.columns))
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        log(f"-- missing values ({len(missing)} cols):")
        log(missing.sort_values(ascending=False).to_string())
    else:
        log("-- missing values: none")
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    if num:
        log("-- numeric describe:")
        log(df[num].describe().round(2).to_string())
    # value counts for low-cardinality object columns
    for c in df.columns:
        if df[c].dtype == object:
            vc = df[c].value_counts(dropna=False)
            if 2 <= len(vc) <= 30:
                log(f"-- value_counts({c}):")
                log(vc.to_string())
    log(f"-- head(2):\n{df.head(2).to_string()}")


# ---------------- 1. small files ----------------
small = ["games.csv", "players.csv", "combine_results.csv", "player_career_successes.csv"]
dfs = {}
for f in small:
    stamp(f"load {f}")
    df = pd.read_csv(os.path.join(BASE, f), low_memory=False)
    dfs[f] = df
    summarize_df(df, f)

# ---------------- 2. cross-file analysis (players / combine / career) ----------------
stamp("cross-file analysis")
players = dfs["players.csv"]
combine = dfs["combine_results.csv"].drop(columns=["draft_year"])
career = dfs["player_career_successes.csv"]
m = players.merge(combine, on="nfl_id", how="left").merge(career, on="nfl_id", how="left")
log(f"merged players+combine+career shape: {m.shape}")
log("position distribution:")
log(m["nfl_position"].value_counts(dropna=False).to_string())
log("draft_year distribution:")
log(m["draft_year"].value_counts(dropna=False).sort_index().to_string())
log("career target columns distribution:")
for c in ["career_games_active", "career_games_started", "pro_bowl_original_ballot",
          "ap_all_pro_1st_team", "ap_all_pro_2nd_team"]:
    if c in m.columns:
        log(f"-- {c}:")
        log(m[c].value_counts(dropna=False).sort_index().head(30).to_string())
log("correlation: combine metrics vs career metrics:")
combine_num = [c for c in ["combine_height", "combine_weight", "hand_size", "arm_length",
                           "wing_span", "ten_yd_split", "forty", "vertical", "broad_jump",
                           "three_cone", "short_shuttle", "bench_reps", "ngs_athleticism_score",
                           "ngs_college_production_score", "ngs_final_score"] if c in m.columns]
career_num = [c for c in ["career_offensive_snaps", "career_defensive_snaps",
                          "career_special_teams_snaps", "career_games_active",
                          "career_games_started"] if c in m.columns]
corr = m[combine_num + career_num].corr(numeric_only=True).loc[career_num, combine_num]
log(corr.round(2).to_string())

# ---------------- 3. player_play.csv (full load) ----------------
stamp("load player_play.csv (full)")
pp = pd.read_csv(os.path.join(BASE, "player_play.csv"), low_memory=False)
log(f"shape: {pp.shape}")
log("unique games: %d, unique plays: %d, unique players: %d" %
    (pp["game_id"].nunique(), pp["play_id"].nunique(), pp["nfl_id"].nunique()))
log("lined_up_position value_counts:")
log(pp["lined_up_position"].value_counts(dropna=False).head(20).to_string())
log("down value_counts:")
log(pp["down"].value_counts(dropna=False).sort_index().to_string())
log("quarter value_counts:")
log(pp["quarter"].value_counts(dropna=False).sort_index().to_string())
log("missing per column (top 15):")
log(pp.isna().sum().sort_values(ascending=False).head(15).to_string())
log("season coverage (join games not needed, use play_id frequency):")
log("team_abbr value_counts (top 35):")
log(pp["team_abbr"].value_counts(dropna=False).head(35).to_string())

# ---------------- 4. combine_tracking.csv (full load) ----------------
stamp("load combine_tracking.csv (full)")
ct = pd.read_csv(os.path.join(BASE, "combine_tracking.csv"), low_memory=False)
log(f"shape: {ct.shape}")
log("drill_type value_counts:")
log(ct["drill_type"].value_counts(dropna=False).to_string())
log("drill_name value_counts:")
log(ct["drill_name"].value_counts(dropna=False).to_string())
log("unique players: %d, unique events: %d" % (ct["nfl_id"].nunique(), ct["event_id"].nunique()))
log("missing per column:")
log(ct.isna().sum().to_string())
log("entity_type value_counts:")
log(ct["entity_type"].value_counts(dropna=False).to_string())

# ---------------- 5. game_tracking_*.csv (raw count + sample + chunked uniques) ----------------
for year in ["2023", "2024", "2025"]:
    f = f"game_tracking_{year}.csv"
    path = os.path.join(BASE, f)
    stamp(f"raw line count {f}")
    rows = count_lines_raw(path)
    log(f"total rows (excl header): {rows:,}")

    stamp(f"sample 200k rows of {f}")
    sample = pd.read_csv(path, nrows=200_000, low_memory=False)
    log(f"sample shape: {sample.shape}")
    log("dtypes:")
    log(sample.dtypes.to_string())
    log("event value_counts (sample):")
    log(sample["event"].value_counts(dropna=True).head(15).to_string())
    log("time min/max (sample): %s .. %s" % (str(sample["time"].min()), str(sample["time"].max())))
    log("missing per column (sample):")
    log(sample.isna().sum().to_string())

    stamp(f"chunked unique aggregation {f}")
    games = set()
    plays = set()
    players = set()
    nrows = 0
    it = pd.read_csv(path, usecols=["game_id", "play_id", "nfl_id"],
                     chunksize=3_000_000, low_memory=False)
    for chunk in it:
        games.update(chunk["game_id"].unique().tolist())
        plays.update(chunk["play_id"].unique().tolist())
        players.update(chunk["nfl_id"].unique().tolist())
        nrows += len(chunk)
        log(f"  ... processed {nrows:,} rows")
    log(f"{year}: rows={nrows:,} unique_games={len(games)} unique_plays={len(plays)} unique_players={len(players)}")

# ---------------- save report ----------------
with open(OUT, "w", encoding="utf-8") as fout:
    fout.write("\n".join(buf))
log(f"\nREPORT SAVED TO {OUT}")
log(f"TOTAL TIME {time.time() - t0:.1f}s")
