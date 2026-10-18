# -*- coding: utf-8 -*-
"""
Link combine route-drill movement features -> in-game separation at pass forward.
Two levels:
  L1 play-level ridge (controls + route dummies + movement z-features), player-cluster bootstrap
  L2 player x route aggregate weighted regression (the core movement->separation claim)
Outputs:
  features/analysis_play_level.csv
  features/analysis_player_route.csv
  features/player_expected_sep.csv
  features/results_model.json
"""
import numpy as np
import pandas as pd
import os, json, warnings
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
BASE = r"D:\kaggle\NFL2027\nfl-big-data-bowl-2027"
OUTD = r"D:\kaggle\NFL2027\features"

pr = pd.read_csv(os.path.join(OUTD, "combine_player_route.csv"))
players = pd.read_csv(os.path.join(BASE, "players.csv"), low_memory=False)
pp = pd.read_csv(os.path.join(BASE, "player_play.csv"), low_memory=False)
print("player_play loaded:", pp.shape)

# ---------- in-game WR plays with separation ----------
wr_ids = set(players.loc[players["nfl_position"] == "WR", "nfl_id"])
plays = pp[pp["nfl_id"].isin(wr_ids)].copy()
plays = plays[plays["separation_at_pass_forward"].notna()]
route_map = {
    "GO": "GO", "SLANT": "SLANT", "HITCH": "HITCH", "OUT": "OUT",
    "POST": "POST_CORNER", "CORNER": "POST_CORNER",
    "SCREEN": "SCREEN", "FLAT": "FLAT", "WHEEL": "WHEEL",
}
plays = plays[plays["route_ran"].isin(route_map.keys())].copy()
plays["route_key"] = plays["route_ran"].map(route_map)

comb_cols = [c for c in pr.columns if c not in ("nfl_id", "route_key", "n_attempts")]
df = plays.merge(pr, on=["nfl_id", "route_key"], how="left")
print("merge rate:", round(df["speed_loss"].notna().mean(), 3))

# ---------- controls ----------
df["draft_year"] = df["nfl_id"].map(players.set_index("nfl_id")["draft_year"])
df["draft_pick"] = df["nfl_id"].map(players.set_index("nfl_id")["draft_overall_pick"])
df["udfa"] = df["draft_pick"].isna().astype(int)
df["draft_pick_f"] = df["draft_pick"].fillna(270.0)
df["man"] = (df["team_coverage_man_zone"] == "MAN_COVERAGE").astype(int)
df["log_sep"] = np.log(df["separation_at_pass_forward"] + 0.01)
df["log_cushion"] = np.log(df["cushion"] + 0.01)
df["target_f"] = df["target"].fillna(False).astype(int)
df["slot"] = (df["lined_up_position"] == "SLOT_WR").astype(int)

mov_feats = ["decel_into_cut_max", "accel_out_cut_mean", "retention_mean", "retention_min",
             "cut_angle_max", "cut_angle_mean", "speed_max", "speed_mean", "speed_cv",
             "turn_total", "cod_index", "dec_peak", "acc_peak", "dur_s", "dist_total",
             "cut_dur_mean", "n_cuts", "speed_trough", "speed_loss", "speed_loss_ratio",
             "decel_dist", "post_trough_accel"]

# z-score movement features within route_key
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

controls = ["man", "log_cushion", "target_f", "slot",
            "down", "yards_to_go", "quarter", "draft_year", "draft_pick_f", "udfa"]
df[controls] = df[controls].fillna(df[controls].median())
route_dummies = pd.get_dummies(df["route_key"], prefix="rt", drop_first=False)
X = pd.concat([df[zcols], df[controls], route_dummies], axis=1).astype(float)
y = df["log_sep"].values
print("X shape:", X.shape)

# ---------- L1: play-level ----------
best = None
for al in [1.0, 10.0, 100.0, 500.0]:
    m = Ridge(alpha=al).fit(X, y)
    r2 = m.score(X, y)
    print(f"alpha={al:6.1f}  R2={r2:.4f}")
    if best is None or r2 > best[1]:
        best = (m, r2)
model, r2_full = best

Xmov = pd.concat([df[zcols], route_dummies], axis=1).astype(float)
mm = Ridge(alpha=100.0).fit(Xmov, y)
Xctrl = pd.concat([df[controls], route_dummies], axis=1).astype(float)
mc = Ridge(alpha=100.0).fit(Xctrl, y)
r2_mov = 1 - np.sum((y - mm.predict(Xmov)) ** 2) / np.sum((y - y.mean()) ** 2)
r2_ctrl = 1 - np.sum((y - mc.predict(Xctrl)) ** 2) / np.sum((y - y.mean()) ** 2)
print(f"R2 full={r2_full:.4f}  movement-only={r2_mov:.4f}  controls-only={r2_ctrl:.4f}")

# player-cluster bootstrap
rng = np.random.default_rng(42)
pids = df["nfl_id"].unique()
Xv = X.values


def boot_fit():
    samp = rng.choice(pids, len(pids), replace=True)
    idx = df["nfl_id"].isin(samp).values
    return Ridge(alpha=best[0].alpha).fit(Xv[idx], y[idx]).coef_


coefs = np.vstack([boot_fit() for _ in range(400)])
ci_lo = np.percentile(coefs, 2.5, axis=0)
ci_hi = np.percentile(coefs, 97.5, axis=0)
coef_tbl = pd.DataFrame({
    "feature": X.columns, "coef": model.coef_,
    "ci_lo": ci_lo, "ci_hi": ci_hi, "sig": np.sign(ci_lo) == np.sign(ci_hi)})
print("\nMovement coefficients (z within route), play level:")
print(coef_tbl[coef_tbl["feature"].isin(zcols)].round(3).to_string())

df["pred_log_sep_mov"] = mm.predict(Xmov)
df.to_csv(os.path.join(OUTD, "analysis_play_level.csv"), index=False)

# ---------- L2: player x route aggregate ----------
df["mov_na"] = df["speed_loss"].isna().astype(int)
agg = df.groupby(["nfl_id", "route_key"], sort=False).agg(
    n_plays=("game_id", "count"),
    obs_log_sep=("log_sep", "mean"),
    man_rate=("man", "mean"),
    log_cushion_m=("log_cushion", "mean"),
    target_rate=("target_f", "mean"),
    slot_rate=("slot", "mean"),
).reset_index()
agg = agg.merge(pr, on=["nfl_id", "route_key"], how="left")
agg = agg[agg["speed_loss"].notna()].copy()
print("\nL2 player x route rows:", len(agg), "| players:", agg["nfl_id"].nunique())

# z within route for movement feats at L2
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

rd2 = pd.get_dummies(agg["route_key"], prefix="rt", drop_first=False)
ctrl2 = ["man_rate", "log_cushion_m", "target_rate", "slot_rate"]
Xm2 = pd.concat([agg[zcols2], rd2], axis=1).astype(float)
Xc2 = pd.concat([agg[ctrl2], rd2], axis=1).astype(float)
y2 = agg["obs_log_sep"].values
w2 = np.sqrt(agg["n_plays"].values)

mm2 = Ridge(alpha=20.0).fit(Xm2, y2, sample_weight=w2)
mc2 = Ridge(alpha=20.0).fit(Xc2, y2, sample_weight=w2)
r2_m2 = 1 - np.sum(w2 * (y2 - mm2.predict(Xm2)) ** 2) / np.sum(w2 * (y2 - y2.mean()) ** 2)
r2_c2 = 1 - np.sum(w2 * (y2 - mc2.predict(Xc2)) ** 2) / np.sum(w2 * (y2 - y2.mean()) ** 2)
print(f"L2 R2: movement-only={r2_m2:.4f}  controls-only={r2_c2:.4f}")

# L2 bootstrap CIs (player-cluster)
pids2 = agg["nfl_id"].unique()
n_boot = 500
coefs2 = []
for _ in range(n_boot):
    samp = rng.choice(pids2, len(pids2), replace=True)
    idx = agg["nfl_id"].isin(samp).values
    coefs2.append(Ridge(alpha=20.0).fit(Xm2.values[idx], y2[idx],
                                        sample_weight=w2[idx]).coef_)
coefs2 = np.array(coefs2)
tbl2 = pd.DataFrame({
    "feature": Xm2.columns, "coef": mm2.coef_,
    "ci_lo": np.percentile(coefs2, 2.5, axis=0),
    "ci_hi": np.percentile(coefs2, 97.5, axis=0),
    "sig": np.sign(np.percentile(coefs2, 2.5, axis=0)) == np.sign(np.percentile(coefs2, 97.5, axis=0))})
print("\nL2 movement coefficients (weighted, player x route):")
print(tbl2[tbl2["feature"].isin(zcols2)].round(3).to_string())

agg["pred_log_sep_mov"] = mm2.predict(Xm2)
agg["sep_residual"] = agg["obs_log_sep"] - agg["pred_log_sep_mov"]
agg.to_csv(os.path.join(OUTD, "analysis_player_route.csv"), index=False)

# per-player expected separation = weighted avg across routes
psep = df.groupby("nfl_id", sort=False).agg(
    n_plays=("game_id", "count"),
    obs_sep=("separation_at_pass_forward", "mean"),
    obs_log_sep=("log_sep", "mean"),
    pred_log_sep_mov=("pred_log_sep_mov", "mean"),
).reset_index()
psep["pred_sep_mov"] = np.exp(psep["pred_log_sep_mov"])
psep["sep_residual"] = psep["obs_log_sep"] - psep["pred_log_sep_mov"]
psep.to_csv(os.path.join(OUTD, "player_expected_sep.csv"), index=False)
print("\nplayers with expected separation:", len(psep))

results = {
    "L1": {"n_plays": int(len(df)), "r2_full": float(r2_full),
           "r2_movement_only": float(r2_mov), "r2_controls_only": float(r2_ctrl),
           "coef_table": coef_tbl.round(4).to_dict("records")},
    "L2": {"n_rows": int(len(agg)), "n_players": int(agg["nfl_id"].nunique()),
           "r2_movement_only": float(r2_m2), "r2_controls_only": float(r2_c2),
           "coef_table": tbl2.round(4).to_dict("records")},
}
with open(os.path.join(OUTD, "results_model.json"), "w") as f:
    json.dump(results, f, indent=2)
print("saved results_model.json")
print("DONE.")
