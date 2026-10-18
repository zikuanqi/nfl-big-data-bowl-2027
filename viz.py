# -*- coding: utf-8 -*-
"""Figures for the writeup (English labels). Outputs D:/kaggle/NFL2027/figs/*.png"""
import numpy as np
import pandas as pd
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BASE = r"D:\kaggle\NFL2027\nfl-big-data-bowl-2027"
OUTD = r"D:\kaggle\NFL2027\features"
FIGD = r"D:\kaggle\NFL2027\figs"
os.makedirs(FIGD, exist_ok=True)

NAVY, RED, GRAY, GOLD = "#013369", "#D50A0A", "#7f8c8d", "#F2A900"
plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.6,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
})

players = pd.read_csv(os.path.join(BASE, "players.csv"), low_memory=False)
career = pd.read_csv(os.path.join(BASE, "player_career_successes.csv"), low_memory=False)
ev = pd.read_csv(os.path.join(OUTD, "combine_event_feat.csv"))
agg = pd.read_csv(os.path.join(OUTD, "analysis_player_route.csv"))
psep = pd.read_csv(os.path.join(OUTD, "player_expected_sep.csv"))
res = json.load(open(os.path.join(OUTD, "results_model.json")))
career_res = json.load(open(os.path.join(OUTD, "career_results.json")))
extra = json.load(open(os.path.join(OUTD, "extra_results.json")))

# ---------------- FIG 1: route mechanics of one drill event ----------------
ct = pd.read_csv(os.path.join(BASE, "combine_tracking.csv"), low_memory=False)
# pick a clean SLANT event with a detected cut
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

fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.8), sharex=True,
                         gridspec_kw={"height_ratios": [1.4, 1.0, 1.0]})
axes[0].plot(tt, s, color=NAVY, lw=2.2)
axes[0].axvspan(tt[max(0, i_c - 4)], tt[min(len(tt) - 1, i_c + 4)], color=GOLD, alpha=0.30)
axes[0].annotate("cut window", xy=(tt[i_c], s[i_c]), xytext=(tt[i_c] + 0.4, s[i_c] + 1.2),
                 fontsize=10, color="#5a4300",
                 arrowprops=dict(arrowstyle="->", color="#5a4300", lw=1))
axes[0].set_ylabel("speed (yd/s)")
axes[0].set_ylim(0, None)
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
fig.savefig(os.path.join(FIGD, "fig1_route_mechanics.png"))
plt.close(fig)
print("fig1 saved")

# ---------------- FIG 2: L2 coefficient forest plot ----------------
tbl = pd.DataFrame(res["L2"]["coef_table"])
mov_names = ["retention_mean_z", "cut_angle_max_z", "speed_cv_z", "decel_dist_z",
             "post_trough_accel_z", "speed_loss_z", "n_cuts_z"]
lab = {"retention_mean_z": "Speed retention through cut",
       "cut_angle_max_z": "Sharpest cut angle",
       "speed_cv_z": "Speed variability (CV)",
       "decel_dist_z": "Deceleration distance",
       "post_trough_accel_z": "Re-acceleration after break",
       "speed_loss_z": "Speed loss at break",
       "n_cuts_z": "Number of cuts"}
sub = tbl[tbl["feature"].isin(mov_names)].copy()
sub["label"] = sub["feature"].map(lab)
sub = sub.sort_values("coef", ascending=False)
ypos = np.arange(len(sub))[::-1]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.errorbar(sub["coef"], ypos, xerr=[sub["coef"] - sub["ci_lo"], sub["ci_hi"] - sub["coef"]],
            fmt="o", color=NAVY, ecolor=GRAY, capsize=4, ms=7)
sig = sub["sig"].values
ax.scatter(sub["coef"][sig], ypos[sig], color=RED, s=55, zorder=5)
ax.axvline(0, color="#333", lw=0.9)
ax.set_yticks(ypos)
ax.set_yticklabels(sub["label"], fontsize=10)
ax.set_xlabel("coefficient on log separation (z-scored movement feature)")
ax.set_title("Combine movement features -> in-game separation\n(player x route level, n=494; player-clustered bootstrap 95% CI)",
             fontsize=11, color=NAVY)
fig.savefig(os.path.join(FIGD, "fig2_coef_forest.png"))
plt.close(fig)
print("fig2 saved")

# ---------------- FIG 3: beyond the stopwatch ----------------
A = extra["A"]
fig, ax = plt.subplots(figsize=(4.6, 3.4))
bars = [A["r2_forty"], A["r2_movement"], A["r2_forty_plus_movement"]]
colors = [GRAY, NAVY, RED]
ax.bar(["40-yard dash\nalone", "Movement\nsignature alone", "40 + movement"],
       bars, color=colors, width=0.62)
for i, b in enumerate(bars):
    ax.text(i, b + 0.005, f"{b:.3f}", ha="center", fontsize=11)
ax.set_ylim(0, max(bars) * 1.15)
ax.set_ylabel("weighted R$^2$ (route-level log separation)")
ax.set_title("Sensor movement beats the stopwatch\nfor predicting separation", fontsize=11, color=NAVY)
fig.savefig(os.path.join(FIGD, "fig3_beyond_stopwatch.png"))
plt.close(fig)
print("fig3 saved")

# ---------------- FIG 4: career increment ----------------
# values from career_validation.log (stable after winsorization)
draft_r2, draft_mov_r2 = 0.397, 0.426
hit_draft, hit_mov = career_res["hit_top15_draft_overall_pick"], career_res["hit_top15_pred_games_started"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
axes[0].bar(["Draft capital\nalone", "Draft + movement\nsignature"], [draft_r2, draft_mov_r2],
            color=[GRAY, NAVY], width=0.55)
for i, b in enumerate([draft_r2, draft_mov_r2]):
    axes[0].text(i, b + 0.006, f"{b:.3f}", ha="center", fontsize=10)
axes[0].set_ylabel("LOOCV R$^2$ (career games started, n=106)")
axes[0].set_ylim(0, 0.47)
axes[1].bar(["Top-15 by\ndraft capital", "Top-15 by\ndraft + movement"],
            [hit_draft, hit_mov], color=[GRAY, RED], width=0.55)
for i, b in enumerate([hit_draft, hit_mov]):
    axes[1].text(i, b + 0.01, f"{b:.2f}", ha="center", fontsize=10)
axes[1].set_ylabel("starter hit rate (>= 11 games started)")
axes[1].set_ylim(0, 1.05)
fig.suptitle("Movement signature adds incremental career signal beyond draft capital",
             fontsize=11, color=NAVY, y=1.02)
fig.savefig(os.path.join(FIGD, "fig4_career_increment.png"))
plt.close(fig)
print("fig4 saved")

# ---------------- FIG 5: case-study scatter ----------------
info = players[["nfl_id", "display_name", "draft_overall_pick"]].merge(
    career[["nfl_id", "career_games_started"]], on="nfl_id", how="left")
psep2 = psep.merge(info, on="nfl_id", how="left")
rho, pv = spearmanr(psep2["pred_sep_mov"], psep2["obs_sep"])
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
ax.set_title(f"Combine movement -> NFL separation (n=105, Spearman r={rho:.2f})",
             fontsize=11, color=NAVY)
fig.savefig(os.path.join(FIGD, "fig5_case_scatter.png"))
plt.close(fig)
print("fig5 saved")

# ---------------- FIG 6: cut signatures (speed profiles aligned at trough) ----------------
# group events by player mean retention quartile; average normalized speed around trough
pl_ret = agg.groupby("nfl_id")["retention_mean"].mean()
q = pl_ret.quantile([0.25, 0.75])
ev_ret = ev.merge(pl_ret.rename("pl_retention"), on="nfl_id", how="left")
ev_ret["group"] = np.where(ev_ret["pl_retention"] <= q.iloc[0], "Low retention (Q1)",
                           np.where(ev_ret["pl_retention"] >= q.iloc[1], "High retention (Q3)", "Mid"))
profiles = {g: [] for g in ["Low retention (Q1)", "High retention (Q3)"]}
T = np.arange(-0.5, 0.6, 0.1)
wr_ct = ct[ct["drill_type"] == "SKILL_DRILLS_WR"].copy()
wr_ct["t"] = pd.to_datetime(wr_ct["time"])
for eid, grp in wr_ct.groupby("event_id"):
    p = grp[grp["entity_type"] == "PLAYER"].sort_values("t")
    if len(p) < 15:
        continue
    tt = (p["t"] - p["t"].min()).dt.total_seconds().values
    s = p["s"].values.astype(float)
    s_s = pd.Series(s).rolling(5, center=True, min_periods=1).mean().values
    i_min = int(np.argmin(s_s))
    if i_min < 5 or i_min > len(s) - 6:
        continue
    seg = s_s[i_min - 5:i_min + 6]
    row = ev_ret[ev_ret["event_id"] == eid]
    if len(row) == 0 or row["group"].iloc[0] not in profiles:
        continue
    gname = row["group"].iloc[0]
    if len(seg) == 11 and seg.max() > 1e-6:
        profiles[gname].append(seg / seg.max())
fig, ax = plt.subplots(figsize=(5.6, 3.6))
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
ax.set_title("Cut signature: speed through the break\nby combine speed-retention quartile", fontsize=11, color=NAVY)
ax.legend(fontsize=9)
fig.savefig(os.path.join(FIGD, "fig6_cut_signature.png"))
plt.close(fig)
print("fig6 saved")

# ---------------- FIG 7: late-pick movement signal ----------------
B = extra["B"]
fig, ax = plt.subplots(figsize=(4.2, 3.2))
ax.bar(["Route-type\nbaseline", "Movement\nsignature"], [B["r2_route_baseline"], B["r2_movement"]],
       color=[GRAY, NAVY], width=0.55)
for i, b in enumerate([B["r2_route_baseline"], B["r2_movement"]]):
    ax.text(i, b + 0.004, f"{b:.3f}", ha="center", fontsize=11)
ax.set_ylim(0, max(B["r2_route_baseline"], B["r2_movement"]) * 1.2)
ax.set_ylabel("weighted R$^2$ (route-level log separation)")
ax.set_title(f"Movement signal persists for Day-3 picks & UDFAs\n(n={B['n_rows']} player-routes, {B['n_players']} players)",
             fontsize=10.5, color=NAVY)
fig.savefig(os.path.join(FIGD, "fig7_late_picks.png"))
plt.close(fig)
print("fig7 saved")
print("ALL FIGURES DONE")
