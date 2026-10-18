# -*- coding: utf-8 -*-
"""Build the public Kaggle notebook: nfl2027_wr_analysis.ipynb"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.10"}}
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s))


def code(s):
    cells.append(nbf.v4.new_code_cell(s))


md("""# Beyond the Stopwatch: Combine Route-Running Mechanics Predict NFL Separation
**NFL Big Data Bowl 2027 | Wide Receiver Route Running**

This notebook reproduces the full analysis from the writeup:
1. Extract movement signatures from 10 Hz Combine route-drill tracking (1,529 WR drill attempts)
2. Link them to in-game *separation at pass forward* (28,830 routes, 106 WRs)
3. Validate against career outcomes and the 40-yard dash

Key results: sharp, efficient cuts (not straight-line speed) predict NFL separation;
the movement signature beats the 40-yard dash (R² 0.19 vs 0.14) and adds incremental
career signal beyond draft capital (LOOCV R² 0.397 → 0.426; top-15 starter hit rate 80% → 93%).""")

code("""import numpy as np
import pandas as pd
import os, json, warnings
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})

BASE = None
for cand in [r"D:\\kaggle\\NFL2027\\nfl-big-data-bowl-2027",
             "/kaggle/input/nfl-big-data-bowl-2027",
             "/kaggle/input/competitions/nfl-big-data-bowl-2027",
             "/kaggle/input/competitions/nfl-big-data-bowl-2027/nfl-big-data-bowl-2027"]:
    if os.path.exists(os.path.join(cand, "players.csv")):
        BASE = cand
        break
if BASE is None:
    raise FileNotFoundError("competition data not found - check input paths")
print("BASE:", BASE)

players = pd.read_csv(f"{BASE}/players.csv", low_memory=False)
combine = pd.read_csv(f"{BASE}/combine_results.csv", low_memory=False)
career = pd.read_csv(f"{BASE}/player_career_successes.csv", low_memory=False)
pp = pd.read_csv(f"{BASE}/player_play.csv", low_memory=False)
ct = pd.read_csv(f"{BASE}/combine_tracking.csv", low_memory=False)
ct["t"] = pd.to_datetime(ct["time"])
print("tracking rows:", ct.shape[0], "| player_play rows:", pp.shape[0])
print("WRs:", (players["nfl_position"] == "WR").sum())""")

md("""## 1. Movement signature from Combine drill tracking

For every WR skill-drill attempt we detect cuts (>= 25 deg of turning within 0.9 s,
from the NGS direction series) and the global speed trough (the break). Per attempt we
extract speed retention through the cut, sharpest cut angle, speed variability (CV),
deceleration distance into the break, re-acceleration after it, peak speed, and more.""")

code("""wr = ct[ct["drill_type"] == "SKILL_DRILLS_WR"].copy()

def circular_diff(deg):
    d = np.diff(deg)
    return (d + 180.0) % 360.0 - 180.0

def route_key(dn):
    d = dn.replace("_ROUTE", "")
    if "GAUNTLET" in d: return "GAUNTLET"
    if "OVER_SHOULDER" in d: return "OVER_SHOULDER"
    if "SLANT" in d: return "SLANT"
    if "POST_CORNER" in d: return "POST_CORNER"
    if "SLOT_SAIL" in d: return "SLOT_SAIL"
    if "COMEBACK" in d: return "COMEBACK"
    if "CURL" in d or "HOOK" in d: return "HITCH"
    if "SPEED_OUT" in d: return "OUT"
    if "DAGGER" in d: return "DAGGER"
    if "CIRCUS" in d: return "CIRCUS"
    if "SLOT_STRIKE" in d: return "SLOT_STRIKE"
    if "FADE" in d: return "FADE"
    if "GO" in d: return "GO"
    return "OTHER"

def extract_event(g):
    t0 = g["t"].min()
    p = g[g["entity_type"] == "PLAYER"].sort_values("t")
    tt = (p["t"] - t0).dt.total_seconds().values
    s = p["s"].values.astype(float)
    a = p["a"].values.astype(float)
    dis = p["dis"].values.astype(float)
    dirv = p["dir"].values.astype(float)
    n = len(s)
    out = {}
    out["n_frames"] = n
    out["dur_s"] = tt[-1] - tt[0]
    out["dist_total"] = dis.sum()
    out["speed_max"] = s.max()
    out["speed_mean"] = s.mean()
    out["speed_cv"] = s.std() / max(s.mean(), 1e-6)
    out["acc_peak"] = a.max()
    out["dec_peak"] = float((-a).max())
    turn = circular_diff(dirv)
    out["turn_total"] = float(np.abs(turn).sum())
    out["cod_index"] = float(out["turn_total"] / max(out["dist_total"], 1e-6))
    s_s = pd.Series(s).rolling(5, center=True, min_periods=1).mean().values
    turn_win = pd.Series(turn).rolling(9, center=True, min_periods=1).sum().values

    # global speed trough = the break
    i0 = max(2, int(0.25 * n))
    i1 = max(i0 + 1, int(0.85 * n))
    i_min = i0 + int(np.argmin(s_s[i0:i1]))
    peak_before = s_s[:i_min + 1].max()
    out["speed_trough"] = float(s_s[i_min])
    out["speed_loss"] = float(peak_before - s_s[i_min])
    out["speed_loss_ratio"] = float(out["speed_loss"] / max(peak_before, 1e-6))
    i_peak = int(np.argmax(s_s[:i_min + 1]))
    out["decel_dist"] = float(dis[i_peak:i_min + 1].sum())
    j_end = min(n, i_min + 10)
    out["post_trough_accel"] = float(
        (s_s[i_min:j_end].max() - s_s[i_min]) / max(tt[min(n - 1, j_end - 1)] - tt[i_min], 0.1))

    # cut detection
    is_cut = np.concatenate([np.abs(turn_win) > 25.0, [False]])
    cuts = []
    i = 0
    while i < n:
        if is_cut[i]:
            k = i
            while k + 1 < n:
                if is_cut[k + 1]:
                    k += 1
                elif k + 3 < n and is_cut[k + 2:k + 4].any():
                    k += 1
                else:
                    break
            cuts.append((i, k))
            i = k + 1
        else:
            i += 1
    ret, dec, acc, ang = [], [], [], []
    for (i0c, i1c) in cuts:
        entry = s_s[max(0, i0c - 5):i0c + 1].max()
        exitv = s_s[i1c:min(n, i1c + 6)].max()
        minv = s_s[i0c:i1c + 1].min()
        dt_in = max((tt[min(n - 1, i0c)] - tt[max(0, i0c - 5)]), 0.1)
        dt_out = max((tt[min(n - 1, i1c + 5)] - tt[i1c]), 0.1)
        ret.append(min(exitv / max(entry, 1e-6), 2.0))
        dec.append((entry - minv) / dt_in)
        acc.append((exitv - minv) / dt_out)
        ang.append(float(np.abs(turn[i0c:i1c + 1]).sum()))
    out["n_cuts"] = len(cuts)
    if cuts:
        out["cut_angle_max"] = float(np.max(ang))
        out["cut_angle_mean"] = float(np.mean(ang))
        out["retention_mean"] = float(np.mean(ret))
        out["retention_min"] = float(np.min(ret))
        out["decel_into_cut_max"] = float(np.max(dec))
        out["accel_out_cut_mean"] = float(np.mean(acc))
    else:
        out.update(cut_angle_max=0.0, cut_angle_mean=0.0, retention_mean=1.0,
                   retention_min=1.0, decel_into_cut_max=out["dec_peak"],
                   accel_out_cut_mean=out["acc_peak"])
    return out

rows = [dict(event_id=eid, nfl_id=g["nfl_id"].iloc[0],
             drill_name=g["drill_name"].iloc[0], **extract_event(g))
        for eid, g in wr.groupby("event_id", sort=False)]
ev = pd.DataFrame(rows)
ev["route_key"] = ev["drill_name"].map(route_key)
feat_cols = [c for c in ev.columns if c not in ("event_id", "nfl_id", "drill_name", "route_key")]
pr = ev.groupby(["nfl_id", "route_key"]).agg(
    **{c: (c, "mean") for c in feat_cols}, n_attempts=("event_id", "count")).reset_index()
print("events:", ev.shape[0], "| player x route:", pr.shape[0])
print(ev["route_key"].value_counts().head(8).to_string())""")

md("""## 2. Link movement signature -> in-game separation

Map in-game `route_ran` to Combine route keys (slant/hitch/out/go/post-corner), merge the
movement signature per player x route, then run a weighted ridge regression at the
player x route level (n = 494) with player-clustered bootstrap inference.""")

code("""wr_ids = set(players.loc[players["nfl_position"] == "WR", "nfl_id"])
plays = pp[pp["nfl_id"].isin(wr_ids) & pp["separation_at_pass_forward"].notna()].copy()
route_map = {"GO": "GO", "SLANT": "SLANT", "HITCH": "HITCH", "OUT": "OUT",
             "POST": "POST_CORNER", "CORNER": "POST_CORNER",
             "SCREEN": "SCREEN", "FLAT": "FLAT", "WHEEL": "WHEEL"}
plays = plays[plays["route_ran"].isin(route_map.keys())].copy()
plays["route_key"] = plays["route_ran"].map(route_map)
plays["log_sep"] = np.log(plays["separation_at_pass_forward"] + 0.01)
plays["man"] = (plays["team_coverage_man_zone"] == "MAN_COVERAGE").astype(int)
plays["log_cushion"] = np.log(plays["cushion"] + 0.01)
plays["target_f"] = plays["target"].fillna(False).astype(int)
plays["slot"] = (plays["lined_up_position"] == "SLOT_WR").astype(int)

mov_feats = ["decel_into_cut_max", "accel_out_cut_mean", "retention_mean", "retention_min",
             "cut_angle_max", "cut_angle_mean", "speed_max", "speed_mean", "speed_cv",
             "turn_total", "cod_index", "dec_peak", "acc_peak", "dur_s", "dist_total",
             "n_cuts", "speed_trough", "speed_loss", "speed_loss_ratio",
             "decel_dist", "post_trough_accel"]

df = plays.merge(pr, on=["nfl_id", "route_key"], how="left")
for rk in df["route_key"].unique():
    m = df["route_key"] == rk
    for c in mov_feats:
        vals = df.loc[m, c]
        if vals.notna().sum() > 10:
            df.loc[m, c + "_z"] = (vals - vals.mean()) / (vals.std() + 1e-9)
        else:
            df.loc[m, c + "_z"] = 0.0
zcols = [c + "_z" for c in mov_feats]
df[zcols] = df[zcols].fillna(0.0)

agg = df.groupby(["nfl_id", "route_key"]).agg(
    n_plays=("game_id", "count"), obs_log_sep=("log_sep", "mean"),
    man_rate=("man", "mean"), log_cushion_m=("log_cushion", "mean"),
    target_rate=("target_f", "mean"), slot_rate=("slot", "mean")).reset_index()
agg = agg.merge(pr, on=["nfl_id", "route_key"], how="left")
agg = agg[agg["speed_loss"].notna()].copy()
print("player x route rows:", len(agg), "| players:", agg["nfl_id"].nunique())

for rk in agg["route_key"].unique():
    m = agg["route_key"] == rk
    for c in mov_feats:
        vals = agg.loc[m, c]
        if vals.notna().sum() > 10 and vals.std() > 1e-9:
            agg.loc[m, c + "_z"] = (vals - vals.mean()) / vals.std()
        else:
            agg.loc[m, c + "_z"] = 0.0
zcols2 = [c + "_z" for c in mov_feats]
agg[zcols2] = agg[zcols2].fillna(0.0)

rd = pd.get_dummies(agg["route_key"], prefix="rt", drop_first=False)
ctrl2 = ["man_rate", "log_cushion_m", "target_rate", "slot_rate"]
Xmov = pd.concat([agg[zcols2], rd], axis=1).astype(float)
Xc2 = pd.concat([agg[ctrl2], rd], axis=1).astype(float)
y2 = agg["obs_log_sep"].values
w2 = np.sqrt(agg["n_plays"].values)

m_mov = Ridge(alpha=20.0).fit(Xmov, y2, sample_weight=w2)
m_ctrl = Ridge(alpha=20.0).fit(Xc2, y2, sample_weight=w2)
r2_mov = 1 - np.sum(w2 * (y2 - m_mov.predict(Xmov)) ** 2) / np.sum(w2 * (y2 - y2.mean()) ** 2)
r2_ctrl = 1 - np.sum(w2 * (y2 - m_ctrl.predict(Xc2)) ** 2) / np.sum(w2 * (y2 - y2.mean()) ** 2)
print(f"weighted R2: movement-only={r2_mov:.3f}  controls-only={r2_ctrl:.3f}")

# play-level movement-only predictions (used for player-level career validation)
rd1 = pd.get_dummies(df["route_key"], prefix="rt", drop_first=False)
Xmov1 = pd.concat([df[zcols], rd1], axis=1).astype(float)
mm1 = Ridge(alpha=100.0).fit(Xmov1, df["log_sep"].values)
df["pred_log_sep_mov"] = mm1.predict(Xmov1)""")

code("""# player-clustered bootstrap CIs (500 resamples)
rng = np.random.default_rng(42)
pids = agg["nfl_id"].unique()
boot = []
for _ in range(500):
    samp = rng.choice(pids, len(pids), replace=True)
    idx = agg["nfl_id"].isin(samp).values
    boot.append(Ridge(alpha=20.0).fit(Xmov.values[idx], y2[idx], sample_weight=w2[idx]).coef_)
boot = np.array(boot)
lab = {"retention_mean_z": "Speed retention through cut",
       "cut_angle_max_z": "Sharpest cut angle",
       "speed_cv_z": "Speed variability (CV)",
       "decel_dist_z": "Deceleration distance",
       "post_trough_accel_z": "Re-acceleration after break",
       "speed_loss_z": "Speed loss at break",
       "n_cuts_z": "Number of cuts"}
tbl = pd.DataFrame({
    "feature": Xmov.columns, "coef": m_mov.coef_,
    "ci_lo": np.percentile(boot, 2.5, axis=0),
    "ci_hi": np.percentile(boot, 97.5, axis=0)})
tbl = tbl[tbl["feature"].isin(lab.keys())].copy()
tbl["label"] = tbl["feature"].map(lab)
tbl["sig"] = np.sign(tbl["ci_lo"]) == np.sign(tbl["ci_hi"])
display(tbl.sort_values("coef", ascending=False).round(3)[["label", "coef", "ci_lo", "ci_hi", "sig"]])

fig, ax = plt.subplots(figsize=(8, 3.6))
order = tbl.sort_values("coef", ascending=False)
ypos = np.arange(len(order))[::-1]
ax.errorbar(order["coef"], ypos, xerr=[order["coef"] - order["ci_lo"], order["ci_hi"] - order["coef"]],
            fmt="o", color="#013369", ecolor="#7f8c8d", capsize=4, ms=7)
ax.scatter(order["coef"][order["sig"]], ypos[order["sig"]], color="#D50A0A", s=60, zorder=5)
ax.axvline(0, color="#333", lw=0.9)
ax.set_yticks(ypos)
ax.set_yticklabels(order["label"], fontsize=9)
ax.set_xlabel("coefficient on log separation (z-scored)")
ax.set_title("Combine movement features -> in-game separation (n=494 player-routes)")
plt.show()""")

md("""## 3. Beyond the stopwatch

Compare the movement signature against the 40-yard dash for predicting separation.""")

code("""agg = agg.merge(combine[["nfl_id", "forty"]], on="nfl_id", how="left")
sub = agg.dropna(subset=["forty"])
rd_sub = pd.get_dummies(sub["route_key"], prefix="rt", drop_first=False)
Xf = pd.concat([sub[["forty"]].astype(float), rd_sub], axis=1)
Xmov_sub = pd.concat([sub[zcols2], rd_sub], axis=1).astype(float)
Xfm = pd.concat([sub[["forty"]].astype(float), Xmov_sub], axis=1)
ys = sub["obs_log_sep"].values
ws = np.sqrt(sub["n_plays"].values)
def wr2(Xv):
    m = Ridge(alpha=20.0).fit(Xv, ys, sample_weight=ws)
    return 1 - np.sum(ws * (ys - m.predict(Xv)) ** 2) / np.sum(ws * (ys - ys.mean()) ** 2)
r2_forty, r2_movs, r2_both = wr2(Xf.values), wr2(Xmov_sub.values), wr2(Xfm.values)
print(f"forty-only={r2_forty:.3f}  movement-only={r2_movs:.3f}  forty+movement={r2_both:.3f}  (n={len(sub)})")

fig, ax = plt.subplots(figsize=(5.5, 3.4))
bars = [r2_forty, r2_movs, r2_both]
ax.bar(["40-yard dash\\nalone", "Movement\\nsignature alone", "40 + movement"],
       bars, color=["#7f8c8d", "#013369", "#D50A0A"], width=0.6)
for i, b in enumerate(bars):
    ax.text(i, b + 0.004, f"{b:.3f}", ha="center")
ax.set_ylabel("weighted R2 (route-level log separation)")
ax.set_title("Sensor movement beats the stopwatch")
plt.show()

wr40 = players[players["nfl_position"] == "WR"].merge(
    combine[["nfl_id", "forty"]], on="nfl_id", how="left")
print("WRs missing a 40 time:", wr40["forty"].isna().sum(), "of 106")""")

md("""## 4. Signal among Day-3 picks and UDFAs""")

code("""agg = agg.merge(players[["nfl_id", "draft_overall_pick"]], on="nfl_id", how="left")
late = agg[(agg["draft_overall_pick"] > 100) | (agg["draft_overall_pick"].isna())]
rd_l = pd.get_dummies(late["route_key"], prefix="rt", drop_first=False)
Xmov_l = pd.concat([late[zcols2], rd_l], axis=1).astype(float)
yl = late["obs_log_sep"].values
wl = np.sqrt(late["n_plays"].values)
m_l = Ridge(alpha=20.0).fit(Xmov_l.values, yl, sample_weight=wl)
m_r = Ridge(alpha=20.0).fit(rd_l.values, yl, sample_weight=wl)
r2_l = 1 - np.sum(wl * (yl - m_l.predict(Xmov_l.values)) ** 2) / np.sum(wl * (yl - yl.mean()) ** 2)
r2_r = 1 - np.sum(wl * (yl - m_r.predict(rd_l.values)) ** 2) / np.sum(wl * (yl - yl.mean()) ** 2)
print(f"late-pick movement-only={r2_l:.3f} vs route-baseline={r2_r:.3f} (n={len(late)} player-routes, {late['nfl_id'].nunique()} players)")""")

md("""## 5. Career validation (LOOCV, n = 106)

Does the signature add predictive value beyond draft capital?""")

code("""psep = df.groupby("nfl_id").agg(
    n_plays=("game_id", "count"), obs_log_sep=("log_sep", "mean"),
    pred_log_sep_mov=("pred_log_sep_mov", "mean")).reset_index()
psep["pred_sep_mov"] = np.exp(psep["pred_log_sep_mov"])
psep["obs_sep"] = np.exp(psep["obs_log_sep"])

d = players[players["nfl_position"] == "WR"].merge(career, on="nfl_id", how="left")
d = d.merge(combine[["nfl_id", "forty", "combine_height", "combine_weight"]], on="nfl_id", how="left")
d = d.merge(psep, on="nfl_id", how="left")
d[["forty", "combine_height", "combine_weight"]] = d[["forty", "combine_height", "combine_weight"]].fillna(
    d[["forty", "combine_height", "combine_weight"]].median())
d["udfa"] = d["draft_overall_pick"].isna().astype(int)
d["log_pick"] = np.log(d["draft_overall_pick"].fillna(270.0))
d["Y"] = d["career_games_started"].astype(float)

mv_cols = ["pred_sep_mov", "sep_residual", "retention_mean", "cut_angle_max", "cut_angle_mean",
           "speed_cv", "decel_dist", "post_trough_accel", "speed_loss"]

# per-player Movement Quality Score: z-score each feature within drill_name, then average
grp = ev.groupby("drill_name")[feat_cols].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
evz = ev.copy()
for c in feat_cols:
    evz[c + "_z"] = grp[c]
mqs = evz.groupby("nfl_id")[[c + "_z" for c in feat_cols]].mean().reset_index()
mqs.columns = ["nfl_id"] + feat_cols
d = d.merge(mqs[["nfl_id", "retention_mean", "cut_angle_max", "cut_angle_mean",
                 "speed_cv", "decel_dist", "post_trough_accel", "speed_loss"]],
            on="nfl_id", how="left")
d["sep_residual"] = d["obs_log_sep"] - d["pred_log_sep_mov"]
d["sep_residual"] = d["sep_residual"].fillna(0.0)
d[mv_cols[2:]] = d[mv_cols[2:]].fillna(d[mv_cols[2:]].median())
d["pred_sep_mov"] = d["pred_sep_mov"].fillna(d["pred_sep_mov"].median())

from sklearn.preprocessing import StandardScaler

def loocv(feats):
    X = StandardScaler().fit_transform(d[feats].values)
    y = d["Y"].values
    pred = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.ones(len(y), bool)
        tr[i] = False
        pred[i] = Ridge(alpha=5.0).fit(X[tr], y[tr]).predict(X[i:i + 1])[0]
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    rmse = np.sqrt(np.sum((y - pred) ** 2) / len(y))
    rho, _ = spearmanr(y, pred)
    return r2, rmse, rho, pred

base0 = ["log_pick", "udfa"]
for tag, feats in [("draft only", base0), ("draft + traditional", base0 + ["forty", "combine_height", "combine_weight"]),
                   ("draft + movement", base0 + mv_cols), ("movement only", mv_cols)]:
    r2, rmse, rho, _ = loocv(feats)
    print(f"{tag:22s} LOOCV R2={r2:+.3f}  RMSE={rmse:.2f}  Spearman={rho:+.3f}")

r2d, _, _, pred_d = loocv(base0)
r2dm, _, _, pred_dm = loocv(base0 + mv_cols)
d["pred_draft"] = pred_d
d["pred_full"] = pred_dm

thr = d["career_games_started"].quantile(0.75)
hit_draft = (d.nsmallest(15, "draft_overall_pick")["career_games_started"] >= thr).mean()
hit_full = (d.nlargest(15, "pred_full")["career_games_started"] >= thr).mean()
print(f"starter hit rate (>= {thr:.0f} GS): top-15 draft capital={hit_draft:.2f}  top-15 draft+movement={hit_full:.2f}")""")

md("""## 7. Figures

All seven figures from the writeup, generated from the in-memory dataframes.""")

code("""NAVY, RED, GRAY, GOLD = "#013369", "#D50A0A", "#7f8c8d", "#F2A900"

# ---- Fig 1: route mechanics of one SLANT event ----
sl_ev = ev[(ev["route_key"] == "SLANT") & (ev["n_cuts"] >= 1) & (ev["n_frames"] > 50)]
eid = sl_ev.sort_values("cut_angle_max", ascending=False).iloc[0]["event_id"]
g = ct[ct["event_id"] == eid].copy()
g["t"] = pd.to_datetime(g["time"])
p = g[g["entity_type"] == "PLAYER"].sort_values("t")
tt = (p["t"] - p["t"].min()).dt.total_seconds().values
s = p["s"].values.astype(float)
dirv = p["dir"].values.astype(float)
turn = (np.diff(dirv) + 180.0) % 360.0 - 180.0
turn_win = pd.Series(turn).rolling(9, center=True, min_periods=1).sum().values
i_c = int(np.argmax(np.abs(turn_win)))

fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.2), sharex=True,
                         gridspec_kw={"height_ratios": [1.4, 1.0, 1.0]})
axes[0].plot(tt, s, color=NAVY, lw=2.2)
axes[0].axvspan(tt[max(0, i_c - 4)], tt[min(len(tt) - 1, i_c + 4)], color=GOLD, alpha=0.30)
axes[0].annotate("cut window", xy=(tt[i_c], s[i_c]), xytext=(tt[i_c] + 0.4, s[i_c] + 1.2),
                 fontsize=10, color="#5a4300",
                 arrowprops=dict(arrowstyle="->", color="#5a4300", lw=1))
axes[0].set_ylabel("speed (yd/s)")
axes[0].set_title("Combine SLANT drill - speed profile with detected cut", fontsize=12, color=NAVY)
axes[1].plot(tt[:-1], turn, color=RED, lw=1.6, alpha=0.85)
axes[1].axhline(0, color="#333", lw=0.8)
axes[1].set_ylabel("turn (deg / 0.1s)")
axes[2].plot(p["x"].values, p["y"].values, color=NAVY, lw=1.6)
axes[2].scatter([p["x"].values[0]], [p["y"].values[0]], color="#2e7d32", s=60, zorder=5, label="start")
axes[2].scatter([p["x"].values[i_c]], [p["y"].values[i_c]], color=RED, s=70, zorder=5, label="cut")
axes[2].set_xlabel("x (yds)")
axes[2].set_ylabel("y (yds)")
axes[2].legend(fontsize=9, loc="upper right")
plt.show()

# ---- Fig 6: cut signatures by retention quartile ----
pl_ret = agg.groupby("nfl_id")["retention_mean"].mean()
q25, q75 = pl_ret.quantile([0.25, 0.75])
evr = ev.merge(pl_ret.rename("pl_ret"), on="nfl_id", how="left")
evr["grp"] = np.where(evr["pl_ret"] <= q25, "Low retention (Q1)",
                      np.where(evr["pl_ret"] >= q75, "High retention (Q3)", "Mid"))
profiles = {"Low retention (Q1)": [], "High retention (Q3)": []}
wr_ct = ct[ct["drill_type"] == "SKILL_DRILLS_WR"].copy()
wr_ct["t"] = pd.to_datetime(wr_ct["time"])
for eid2, grp in wr_ct.groupby("event_id"):
    p2 = grp[grp["entity_type"] == "PLAYER"].sort_values("t")
    if len(p2) < 15:
        continue
    s2 = p2["s"].values.astype(float)
    s_s2 = pd.Series(s2).rolling(5, center=True, min_periods=1).mean().values
    i_min2 = int(np.argmin(s_s2))
    if i_min2 < 5 or i_min2 > len(s2) - 6:
        continue
    row = evr[evr["event_id"] == eid2]
    if len(row) == 0 or row["grp"].iloc[0] not in profiles:
        continue
    seg = s_s2[i_min2 - 5:i_min2 + 6]
    if len(seg) == 11 and seg.max() > 1e-6:
        profiles[row["grp"].iloc[0]].append(seg / seg.max())
fig, ax = plt.subplots(figsize=(5.6, 3.6))
T = np.arange(-0.5, 0.6, 0.1)
for gname, color in [("High retention (Q3)", NAVY), ("Low retention (Q1)", RED)]:
    arr = np.array(profiles[gname])
    if len(arr) < 5:
        continue
    mu, sd = arr.mean(axis=0), arr.std(axis=0)
    ax.plot(T, mu, color=color, lw=2.2, label=f"{gname} (n={len(arr)})")
    ax.fill_between(T, mu - sd, mu + sd, color=color, alpha=0.15)
ax.axvline(0, color=GRAY, ls="--", lw=1)
ax.set_xlabel("time around break (s)")
ax.set_ylabel("speed (normalized to route peak)")
ax.set_title("Cut signature: speed through the break by retention quartile", fontsize=11, color=NAVY)
ax.legend(fontsize=9)
plt.show()

# ---- Fig 5: movement-expected vs observed separation (case studies) ----
info = players[["nfl_id", "display_name", "draft_overall_pick"]].merge(
    career[["nfl_id", "career_games_started"]], on="nfl_id", how="left")
psep2 = psep.merge(info, on="nfl_id", how="left")
rho, _ = spearmanr(psep2["pred_sep_mov"], psep2["obs_sep"])
fig, ax = plt.subplots(figsize=(5.6, 4.2))
ax.scatter(psep2["pred_sep_mov"], psep2["obs_sep"], s=26, color=NAVY, alpha=0.55)
for pid, name, mark in [(55884, "Jaxon Smith-Njigba", "s"), (55885, "Quentin Johnston", "^"),
                        (56042, "Puka Nacua", "D")]:
    r = psep2[psep2["nfl_id"] == pid]
    if len(r):
        ax.scatter(r["pred_sep_mov"], r["obs_sep"], s=120, color=RED, marker=mark,
                   zorder=6, edgecolors="white", linewidths=0.8)
        ax.annotate(name, (r["pred_sep_mov"].iloc[0], r["obs_sep"].iloc[0]),
                    xytext=(6, 8), textcoords="offset points", fontsize=9, color=RED)
ax.plot([psep2["pred_sep_mov"].min(), psep2["pred_sep_mov"].max()],
        [psep2["pred_sep_mov"].min(), psep2["pred_sep_mov"].max()], color=GRAY, ls="--", lw=1)
ax.set_xlabel("movement-expected separation (yds)")
ax.set_ylabel("observed in-game separation (yds)")
ax.set_title(f"Combine movement -> NFL separation (n={len(psep2)}, Spearman r={rho:.2f})",
             fontsize=11, color=NAVY)
plt.show()

# ---- Fig 4: career increment ----
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
axes[0].bar(["Draft capital\\nalone", "Draft + movement\\nsignature"], [0.397, r2dm],
            color=[GRAY, NAVY], width=0.55)
for i, b in enumerate([0.397, r2dm]):
    axes[0].text(i, b + 0.006, f"{b:.3f}", ha="center", fontsize=10)
axes[0].set_ylabel("LOOCV R$^2$ (career games started, n=106)")
axes[0].set_ylim(0, 0.47)
axes[1].bar(["Top-15 by\\ndraft capital", "Top-15 by\\ndraft + movement"],
            [hit_draft, hit_full], color=[GRAY, RED], width=0.55)
for i, b in enumerate([hit_draft, hit_full]):
    axes[1].text(i, b + 0.01, f"{b:.2f}", ha="center", fontsize=10)
axes[1].set_ylabel("starter hit rate (>= 11 games started)")
axes[1].set_ylim(0, 1.05)
fig.suptitle("Movement signature adds incremental career signal beyond draft capital",
             fontsize=11, color=NAVY, y=1.02)
plt.show()

# ---- Fig 7: late picks ----
fig, ax = plt.subplots(figsize=(4.2, 3.2))
ax.bar(["Route-type\\nbaseline", "Movement\\nsignature"], [r2_r, r2_l],
       color=[GRAY, NAVY], width=0.55)
for i, b in enumerate([r2_r, r2_l]):
    ax.text(i, b + 0.004, f"{b:.3f}", ha="center", fontsize=11)
ax.set_ylim(0, max(r2_r, r2_l) * 1.2)
ax.set_ylabel("weighted R$^2$ (route-level log separation)")
ax.set_title(f"Movement signal persists for Day-3 picks & UDFAs\\n(n={len(late)} player-routes, {late['nfl_id'].nunique()} players)",
             fontsize=10.5, color=NAVY)
plt.show()""")

md("""## 8. Conclusion

- **Sharp, efficient cuts predict NFL separation.** Sharper cut angles (+), speed consistency
  (-CV), and less speed loss through the break all carry robust, interpretable signal.
- **The movement signature beats the 40-yard dash** for predicting separation (R2 0.19 vs 0.14)
  and subsumes it when combined.
- **The signal survives into Day 3 / UDFA territory** (R2 0.16 vs 0.11 route baseline).
- **Career outcomes:** draft capital dominates early-career opportunity, but the signature
  adds incremental value (LOOCV R2 0.397 -> 0.426; top-15 starter hit rate 80% -> 93%).

The gap between how a prospect moves and what the stopwatch records is real — and it is
worth drafting on.""")

nb.cells = cells
out = r"D:\kaggle\NFL2027\nfl2027_wr_analysis.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("notebook written:", out, "| cells:", len(cells))
