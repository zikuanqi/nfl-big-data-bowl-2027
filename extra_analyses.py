# -*- coding: utf-8 -*-
"""
Supplementary analyses:
 A. forty vs movement features for predicting route-level separation (beyond the stopwatch)
 B. movement->separation within late picks (actionable for Day-3 scouting)
 C. per-route coefficient profile (actionable table)
 D. case-study players (JSN and others)
Outputs: features/extra_results.json
"""
import numpy as np
import pandas as pd
import os, json, warnings
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
BASE = r"D:\kaggle\NFL2027\nfl-big-data-bowl-2027"
OUTD = r"D:\kaggle\NFL2027\features"

agg = pd.read_csv(os.path.join(OUTD, "analysis_player_route.csv"))
players = pd.read_csv(os.path.join(BASE, "players.csv"), low_memory=False)
combine = pd.read_csv(os.path.join(BASE, "combine_results.csv"), low_memory=False)
career = pd.read_csv(os.path.join(BASE, "player_career_successes.csv"), low_memory=False)

info = players[["nfl_id", "display_name", "draft_year", "draft_overall_pick"]].merge(
    combine[["nfl_id", "forty", "ten_yd_split", "vertical", "ngs_athleticism_score",
             "ngs_college_production_score", "ngs_final_score"]], on="nfl_id", how="left")
agg = agg.merge(info, on="nfl_id", how="left")

mov_feats = ["decel_into_cut_max", "accel_out_cut_mean", "retention_mean", "retention_min",
             "cut_angle_max", "cut_angle_mean", "speed_max", "speed_mean", "speed_cv",
             "turn_total", "cod_index", "dec_peak", "acc_peak", "dur_s", "dist_total",
             "cut_dur_mean", "n_cuts", "speed_trough", "speed_loss", "speed_loss_ratio",
             "decel_dist", "post_trough_accel"]
zcols = [c + "_z" for c in mov_feats]
rd = pd.get_dummies(agg["route_key"], prefix="rt", drop_first=False)
Xmov = pd.concat([agg[zcols].fillna(0.0), rd], axis=1).astype(float)
y = agg["obs_log_sep"].values
w = np.sqrt(agg["n_plays"].values)


def wr2(Xv, yv, wv, alpha=20.0):
    m = Ridge(alpha=alpha).fit(Xv, yv, sample_weight=wv)
    r2 = 1 - np.sum(wv * (yv - m.predict(Xv)) ** 2) / np.sum(wv * (yv - yv.mean()) ** 2)
    return m, r2


out = {}
# ---- A. forty vs movement ----
sub = agg.dropna(subset=["forty"]).copy()
Xf = pd.concat([sub[["forty"]].astype(float), rd.loc[sub.index]], axis=1)
_, r2_forty = wr2(Xf.values, sub["obs_log_sep"].values, np.sqrt(sub["n_plays"].values))
Xfm = pd.concat([sub[["forty"]].astype(float), Xmov.loc[sub.index]], axis=1)
_, r2_forty_mov = wr2(Xfm.values, sub["obs_log_sep"].values, np.sqrt(sub["n_plays"].values))
_, r2_mov_sub = wr2(Xmov.loc[sub.index].values, sub["obs_log_sep"].values, np.sqrt(sub["n_plays"].values))
print(f"A. forty-only R2={r2_forty:.3f} | movement-only R2={r2_mov_sub:.3f} | forty+movement R2={r2_forty_mov:.3f} (n={len(sub)})")
out["A"] = dict(n=int(len(sub)), r2_forty=float(r2_forty), r2_movement=float(r2_mov_sub),
                r2_forty_plus_movement=float(r2_forty_mov))

# ---- B. late picks subgroup ----
late = agg[(agg["draft_overall_pick"] > 100) | (agg["draft_overall_pick"].isna())].copy()
print(f"\nB. late-pick player x route rows: {len(late)}, players: {late['nfl_id'].nunique()}")
_, r2_late_mov = wr2(Xmov.loc[late.index].values, late["obs_log_sep"].values,
                     np.sqrt(late["n_plays"].values))
r2_late_base = 1 - np.sum(w[late.index] * (y[late.index] - y[late.index].mean()) ** 2) / \
    np.sum(w[late.index] * (y[late.index] - y[late.index].mean()) ** 2)
# route-dummy-only baseline for late
_, r2_late_rt = wr2(rd.loc[late.index].values, late["obs_log_sep"].values,
                    np.sqrt(late["n_plays"].values))
print(f"   late-pick movement-only R2={r2_late_mov:.3f} (route-dummy baseline={r2_late_rt:.3f})")
out["B"] = dict(n_rows=int(len(late)), n_players=int(late["nfl_id"].nunique()),
                r2_movement=float(r2_late_mov), r2_route_baseline=float(r2_late_rt))

# ---- C. per-route coefficient profile ----
print("\nC. per-route movement->separation (univariate spearman, weighted):")
route_rows = {}
for rk in ["SLANT", "HITCH", "OUT", "GO", "POST_CORNER"]:
    s = agg[agg["route_key"] == rk]
    if len(s) < 25:
        continue
    row = {"route": rk, "n": int(len(s))}
    for f in ["retention_mean_z", "cut_angle_max_z", "speed_cv_z", "decel_dist_z",
              "post_trough_accel_z", "speed_loss_z"]:
        m = s.dropna(subset=["obs_log_sep", f])
        if len(m) > 20:
            rho, p = spearmanr(m[f], m["obs_log_sep"])
            row[f.replace("_z", "")] = dict(rho=float(rho), p=float(p), n=int(len(m)))
    route_rows[rk] = row
    print(f"   {rk:12s} n={row['n']:3d}  " + "  ".join(
        f"{k}={v['rho']:+.2f}" for k, v in row.items() if isinstance(v, dict)))
out["C"] = route_rows

# ---- D. case studies ----
case_ids = {}
for name in ["Jaxon Smith-Njigba", "Quentin Johnston", "Brian Thomas", "Puka Nacua",
             "Will Anderson"]:
    m = players[players["display_name"].str.contains(name.split()[-1], na=False)]
    if len(m) and m["nfl_position"].iloc[0] == "WR":
        case_ids[name] = int(m["nfl_id"].iloc[0])
print("\nD. case studies found:", case_ids)
cases = {}
for name, pid in case_ids.items():
    pr_rows = agg[agg["nfl_id"] == pid]
    if len(pr_rows) == 0:
        continue
    r = {
        "name": name,
        "n_routes": int(len(pr_rows)),
        "avg_obs_sep": float(np.exp(pr_rows["obs_log_sep"].mean())),
        "avg_pred_sep_mov": float(np.exp(pr_rows["pred_log_sep_mov"].mean())),
        "retention_mean": float(pr_rows["retention_mean"].mean()),
        "cut_angle_max": float(pr_rows["cut_angle_max"].mean()),
        "speed_cv": float(pr_rows["speed_cv"].mean()),
        "decel_dist": float(pr_rows["decel_dist"].mean()),
        "forty": float(pr_rows["forty"].iloc[0]) if pd.notna(pr_rows["forty"].iloc[0]) else None,
        "draft_pick": float(pr_rows["draft_overall_pick"].iloc[0]) if pd.notna(pr_rows["draft_overall_pick"].iloc[0]) else None,
    }
    c = career[career["nfl_id"] == pid]
    if len(c):
        r["career_games_started"] = int(c["career_games_started"].iloc[0])
        r["career_offensive_snaps"] = int(c["career_offensive_snaps"].iloc[0])
    cases[name] = r
# cohort means for context
coh = {
    "avg_obs_sep": float(np.exp(agg["obs_log_sep"].mean())),
    "retention_mean": float(agg["retention_mean"].mean()),
    "cut_angle_max": float(agg["cut_angle_max"].mean()),
    "speed_cv": float(agg["speed_cv"].mean()),
    "decel_dist": float(agg["decel_dist"].mean()),
}
out["D"] = dict(cases=cases, cohort_means=coh)
print("   cohort means:", {k: round(v, 2) for k, v in coh.items()})
for name, r in cases.items():
    print(f"   {name:22s} sep={r['avg_obs_sep']:.2f} pred={r['avg_pred_sep_mov']:.2f} "
          f"retention={r['retention_mean']:.2f} cut_angle={r['cut_angle_max']:.0f} "
          f"speed_cv={r['speed_cv']:.2f} forty={r['forty']} GS={r.get('career_games_started')}")

with open(os.path.join(OUTD, "extra_results.json"), "w") as f:
    json.dump(out, f, indent=2)
print("\nsaved extra_results.json")
print("DONE.")
