# -*- coding: utf-8 -*-
"""
Career validation: does the Combine movement signature add predictive value
for career success beyond draft capital + traditional combine numbers?
LOOCV evaluation, n=106 WRs.
Outputs:
  features/career_results.json
  features/career_predictions.csv
"""
import numpy as np
import pandas as pd
import os, json, warnings
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings("ignore")
BASE = r"D:\kaggle\NFL2027\nfl-big-data-bowl-2027"
OUTD = r"D:\kaggle\NFL2027\features"

players = pd.read_csv(os.path.join(BASE, "players.csv"), low_memory=False)
combine = pd.read_csv(os.path.join(BASE, "combine_results.csv"), low_memory=False)
career = pd.read_csv(os.path.join(BASE, "player_career_successes.csv"), low_memory=False)
sep = pd.read_csv(os.path.join(OUTD, "player_expected_sep.csv"))
mqs = pd.read_csv(os.path.join(OUTD, "combine_player_mqs.csv"))
pr = pd.read_csv(os.path.join(OUTD, "combine_player_route.csv"))

wr = players[players["nfl_position"] == "WR"].copy()
df = wr.merge(career, on="nfl_id", how="left")
df = df.merge(combine[["nfl_id", "forty", "combine_height", "combine_weight",
                       "ngs_athleticism_score", "ngs_college_production_score"]], on="nfl_id", how="left")
df = df.merge(sep, on="nfl_id", how="left")
df = df.merge(mqs, on="nfl_id", how="left", suffixes=("", "_mqs"))
print("WR analysis set:", df.shape)

# outcomes
df["Y_games_started"] = df["career_games_started"].astype(float)
df["Y_off_snaps"] = np.log1p(df["career_offensive_snaps"])

# draft capital
df["udfa"] = df["draft_overall_pick"].isna().astype(int)
df["pick"] = df["draft_overall_pick"].fillna(270.0)
df["log_pick"] = np.log(df["pick"])

# movement composite features (small set)
mv_cols = ["pred_sep_mov", "sep_residual",
           "retention_mean", "cut_angle_max", "cut_angle_mean",
           "speed_cv", "decel_dist", "post_trough_accel", "speed_loss"]
for c in mv_cols:
    if c not in df.columns:
        df[c] = np.nan
df[mv_cols] = df[mv_cols].fillna(df[mv_cols].median())

base0 = ["log_pick", "udfa"]                                   # draft capital only
base1 = base0 + ["forty", "combine_height", "combine_weight"]  # + traditional combine
base2 = base1 + mv_cols                                        # + movement signature

# route-specific depth: number of route types available in combine tracking
route_counts = pr.groupby("nfl_id")["route_key"].nunique().rename("n_route_types")
df = df.merge(route_counts, on="nfl_id", how="left")
df["n_route_types"] = df["n_route_types"].fillna(0)


def loocv(X, y, alpha=5.0):
    n = len(y)
    pred = np.zeros(n)
    Xv = StandardScaler().fit_transform(X)
    yv = y.values.astype(float)
    for i in range(n):
        tr = np.ones(n, bool)
        tr[i] = False
        m = Ridge(alpha=alpha).fit(Xv[tr], yv[tr])
        pred[i] = m.predict(Xv[i:i + 1])[0]
    return pred


def eval_model(feats, yname):
    X = df[feats].fillna(df[feats].median())
    y = df[yname].astype(float)
    pred = loocv(X, y)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(ss_res / len(y))
    rho, p = spearmanr(y, pred)
    return dict(r2=r2, rmse=rmse, spearman=rho, spearman_p=p, pred=pred)


out = {}
for yname, ylabel in [("Y_games_started", "games_started"), ("Y_off_snaps", "log_offensive_snaps")]:
    print(f"\n===== outcome: {ylabel} (n={df[yname].notna().sum()}) =====")
    sub = df[df[yname].notna()].copy()
    for tag, feats in [("draft_only", base0), ("draft+traditional", base1),
                       ("draft+movement", base0 + mv_cols), ("movement_only", mv_cols)]:
        X = sub[feats].fillna(sub[feats].median())
        y = sub[yname]
        pred = loocv(X, y)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(ss_res / len(y))
        rho, p = spearmanr(y, pred)
        print(f"  {tag:20s} LOOCV R2={r2:+.3f}  RMSE={rmse:.2f}  Spearman={rho:+.3f}")
        if tag == "draft+movement":
            out[ylabel] = dict(r2=float(r2), rmse=float(rmse), spearman=float(rho),
                               spearman_p=float(p), pred=pred.tolist(),
                               y=y.values.tolist())
            df.loc[sub.index, f"pred_{ylabel}"] = pred

# scouting-style hit rates (games_started only)
sub = df.dropna(subset=["Y_games_started"]).copy()
thr = sub["career_games_started"].quantile(0.75)  # top-quartile starter threshold
print(f"\nstarter threshold (75th pct games_started): {thr:.0f}")
for col, label in [("pred_games_started", "draft+movement model"),
                   ("pred_sep_mov", "movement-expected separation"),
                   ("draft_overall_pick", "draft capital (early picks)")]:
    if col not in sub.columns:
        continue
    if col == "draft_overall_pick":
        r = sub[col].rank(ascending=True)
    else:
        r = sub[col].rank(ascending=False)
    top = sub.loc[r.nsmallest(15).index]
    hit = (top["career_games_started"] >= thr).mean()
    print(f"  top-15 by {label:32s} starter hit rate = {hit:.2f}")
    out[f"hit_top15_{col}"] = float(hit)

# hidden talent: late picks / UDFAs with elite movement signatures
late = sub[(sub["draft_overall_pick"] > 100) | (sub["draft_overall_pick"].isna())]
base_rate = (late["career_games_started"] >= thr).mean()
print(f"\nlate-pick pool (pick>100 or UDFA): n={len(late)}, baseline starter rate = {base_rate:.2f}")
out["late_pool"] = dict(n=int(len(late)), baseline_rate=float(base_rate))
if len(late) >= 10:
    top = late.nlargest(10, "pred_sep_mov")
    hit = (top["career_games_started"] >= thr).mean()
    print(f"  top-10 late picks by movement-expected separation: hit rate = {hit:.2f}")
    out["hit_late_top10_movement"] = float(hit)

# direct link: movement-expected separation vs career outcomes
print("\ndirect correlations with career (n=105):")
for yname in ["Y_games_started", "Y_off_snaps"]:
    m = sub.dropna(subset=[yname, "pred_sep_mov"])
    s, p = spearmanr(m["pred_sep_mov"], m[yname])
    print(f"  pred_sep_mov vs {yname}: spearman={s:+.3f} (p={p:.3f})")
    out[f"predsep_to_{yname}"] = dict(spearman=float(s), p=float(p), n=int(len(m)))
# partial: residualize draft capital
from scipy.stats import rankdata
for yname in ["Y_games_started", "Y_off_snaps"]:
    m = sub.dropna(subset=[yname, "pred_sep_mov"]).copy()
    m["y_r"] = m[yname] - np.polyval(np.polyfit(m["log_pick"], m[yname], 1), m["log_pick"])
    m["s_r"] = m["pred_sep_mov"] - np.polyval(np.polyfit(m["log_pick"], m["pred_sep_mov"], 1), m["log_pick"])
    s, p = spearmanr(m["s_r"], m["y_r"])
    print(f"  partial(draft) pred_sep_mov vs {yname}: spearman={s:+.3f} (p={p:.3f})")
    out[f"partial_predsep_{yname}"] = dict(spearman=float(s), p=float(p), n=int(len(m)))

# hidden talent: movement-expected separation vs draft capital
sub["sep_rank"] = sub["pred_sep_mov"].rank(ascending=False)
sub["draft_rank"] = sub["draft_overall_pick"].rank(ascending=True, na_option="bottom")
sub["talent_gap"] = sub["sep_rank"] - sub["draft_rank"]
out["talent_table"] = sub.sort_values("talent_gap", ascending=False).head(20)[
    ["nfl_id", "display_name", "draft_year", "draft_overall_pick", "forty",
     "career_games_started", "pred_sep_mov", "obs_sep"]].to_dict("records")
print("\ntop-15 movement-underrated-by-draft players (talent_gap):")
print(sub.nlargest(15, "talent_gap")[["display_name", "draft_overall_pick",
                                       "pred_sep_mov", "career_games_started"]].to_string(index=False))

df.to_csv(os.path.join(OUTD, "career_predictions.csv"), index=False)
with open(os.path.join(OUTD, "career_results.json"), "w") as f:
    json.dump(out, f, indent=2)
print("\nsaved career_predictions.csv and career_results.json")
print("DONE.")
