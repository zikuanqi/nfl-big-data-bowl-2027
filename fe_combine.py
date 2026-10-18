# -*- coding: utf-8 -*-
"""
FE step 1: extract kinematic features from Combine WR skill-drill tracking.
Output:
  features/combine_event_feat.csv   - one row per drill attempt (event)
  features/combine_player_route.csv - one row per player x route_type
  features/combine_player_mqs.csv   - per-player aggregate Movement Quality Score
"""
import numpy as np
import pandas as pd
import os

BASE = r"D:\kaggle\NFL2027\nfl-big-data-bowl-2027"
OUTD = r"D:\kaggle\NFL2027\features"
os.makedirs(OUTD, exist_ok=True)

ct = pd.read_csv(os.path.join(BASE, "combine_tracking.csv"), low_memory=False)
print("combine_tracking loaded:", ct.shape)

wr = ct[ct["drill_type"] == "SKILL_DRILLS_WR"].copy()
print("WR skill drill rows:", len(wr), "| events:", wr["event_id"].nunique())
print("BALL rows in WR drills:", (wr["entity_type"] == "BALL").sum())

wr["t"] = pd.to_datetime(wr["time"])


def circular_diff(deg):
    """Per-frame turning angle (deg) in (-180,180]."""
    d = np.diff(deg)
    return (d + 180.0) % 360.0 - 180.0


def extract_event(g):
    t0 = g["t"].min()
    p = g[g["entity_type"] == "PLAYER"].sort_values("t")
    b = g[g["entity_type"] == "BALL"].sort_values("t")
    out = {}
    out["event_id"] = g["event_id"].iloc[0]
    out["nfl_id"] = g["nfl_id"].iloc[0]
    out["drill_name"] = g["drill_name"].iloc[0]
    out["attempt"] = g["attempt"].iloc[0]
    out["n_frames"] = len(p)
    out["n_ball_frames"] = len(b)

    tt = (p["t"] - t0).dt.total_seconds().values
    x = p["x"].values.astype(float)
    y = p["y"].values.astype(float)
    s = p["s"].values.astype(float)
    a = p["a"].values.astype(float)
    dis = p["dis"].values.astype(float)

    out["dur_s"] = tt[-1] - tt[0]
    out["dist_total"] = dis.sum()
    out["speed_max"] = s.max()
    out["speed_mean"] = s.mean()
    out["speed_cv"] = s.std() / max(s.mean(), 1e-6)
    out["acc_peak"] = a.max()
    out["dec_peak"] = float((-a).max())

    # turning from NGS motion direction (smooth) -> per-frame turn degrees
    dirv = p["dir"].values.astype(float)
    turn = circular_diff(dirv)  # deg per frame, 0.1s
    turn_s = pd.Series(turn).rolling(3, center=True, min_periods=1).mean().values
    turn_win = pd.Series(turn).rolling(9, center=True, min_periods=1).sum().values  # ~0.9s
    out["turn_total"] = float(np.abs(turn).sum())
    out["cod_index"] = float(out["turn_total"] / max(out["dist_total"], 1e-6))

    # smooth speed
    s_s = pd.Series(s).rolling(5, center=True, min_periods=1).mean().values

    # ---- global speed trough (decel-into-break / re-accel out of break) ----
    n = len(s)
    i0 = max(2, int(0.25 * n))
    i1 = max(i0 + 1, int(0.85 * n))
    i_min = i0 + int(np.argmin(s_s[i0:i1]))
    peak_before = s_s[:i_min + 1].max()
    out["speed_trough"] = float(s_s[i_min])
    out["speed_loss"] = float(peak_before - s_s[i_min])
    out["speed_loss_ratio"] = float(out["speed_loss"] / max(peak_before, 1e-6))
    # distance traveled while decelerating from peak to trough
    i_peak = int(np.argmax(s_s[:i_min + 1]))
    out["decel_dist"] = float(dis[i_peak:i_min + 1].sum())
    # re-acceleration in the 1s after the trough
    j_end = min(n, i_min + 10)
    out["post_trough_accel"] = float((s_s[i_min:j_end].max() - s_s[i_min]) / max(tt[min(n - 1, j_end - 1)] - tt[i_min], 0.1))

    # detect cuts: |turning over ~0.9s window| > 25 deg
    is_cut = np.abs(turn_win) > 25.0
    is_cut = np.concatenate([is_cut, [False]])  # pad to length n (turn has n-1)
    cuts = []  # (i_start, i_end)
    i = 0
    while i < n:
        if is_cut[i]:
            # extend while next frame is cut, or short gap followed by cut
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

    retentions, decels, accels, angles, durs = [], [], [], [], []
    for (i0, i1) in cuts:
        entry = s_s[max(0, i0 - 5):i0 + 1].max()
        exitv = s_s[i1:min(n, i1 + 6)].max()
        minv = s_s[i0:i1 + 1].min()
        ang = float(np.abs(turn[i0:i1 + 1]).sum())
        dt_in = max((tt[min(n - 1, i0)] - tt[max(0, i0 - 5)]), 0.1)
        dt_out = max((tt[min(n - 1, i1 + 5)] - tt[i1]), 0.1)
        retentions.append(min(exitv / max(entry, 1e-6), 2.0))  # winsorized speed retention
        decels.append((entry - minv) / dt_in)
        accels.append((exitv - minv) / dt_out)
        angles.append(ang)
        durs.append(tt[i1] - tt[i0])

    out["n_cuts"] = len(cuts)
    if cuts:
        out["cut_angle_max"] = float(np.max(angles))
        out["cut_angle_mean"] = float(np.mean(angles))
        out["retention_mean"] = float(np.mean(retentions))
        out["retention_min"] = float(np.min(retentions))
        out["decel_into_cut_max"] = float(np.max(decels))
        out["accel_out_cut_mean"] = float(np.mean(accels))
        out["cut_dur_mean"] = float(np.mean(durs))
    else:
        # no detectable cut -> straight route: informative fallbacks
        out["cut_angle_max"] = 0.0
        out["cut_angle_mean"] = 0.0
        out["retention_mean"] = 1.0
        out["retention_min"] = 1.0
        out["decel_into_cut_max"] = float(out["dec_peak"])
        out["accel_out_cut_mean"] = float(out["acc_peak"])
        out["cut_dur_mean"] = 0.0

    # ball / route timing features
    if len(b) >= 2:
        btt = (b["t"] - t0).dt.total_seconds().values
        bx = b["x"].values.astype(float)
        by = b["y"].values.astype(float)
        t_arr = btt[0]  # first ball frame (release already in flight)
        px_a = np.interp(t_arr, tt, x)
        py_a = np.interp(t_arr, tt, y)
        out["ball_dist_at_arrival"] = float(np.hypot(px_a - bx[0], py_a - by[0]))
        out["ball_speed_at_arrival"] = float(b["s"].values.astype(float)[0])
        # min player-ball distance over joint trajectory (0.1s grid)
        tg = np.arange(max(tt[0], btt[0]), min(tt[-1], btt[-1]) + 0.05, 0.1)
        if len(tg) > 0:
            pxi = np.interp(tg, tt, x)
            pyi = np.interp(tg, tt, y)
            bxi = np.interp(tg, btt, bx)
            byi = np.interp(tg, btt, by)
            d = np.hypot(pxi - bxi, pyi - byi)
            out["catch_min_dist"] = float(d.min())
            out["catch_time"] = float(tg[np.argmin(d)])
            out["speed_at_catch"] = float(np.interp(tg[np.argmin(d)], tt, s))
        else:
            out["catch_min_dist"] = out["ball_dist_at_arrival"]
            out["catch_time"] = np.nan
            out["speed_at_catch"] = np.nan
    else:
        for k in ["ball_dist_at_arrival", "ball_speed_at_arrival",
                  "catch_min_dist", "catch_time", "speed_at_catch"]:
            out[k] = np.nan
    return out


def route_key(dn):
    d = dn.replace("_ROUTE", "")
    if "GAUNTLET" in d:
        return "GAUNTLET"
    if "OVER_SHOULDER" in d:
        return "OVER_SHOULDER"
    if "SLANT" in d:
        return "SLANT"
    if "POST_CORNER" in d:
        return "POST_CORNER"
    if "SLOT_SAIL" in d:
        return "SLOT_SAIL"
    if "COMEBACK" in d:
        return "COMEBACK"
    if "CURL" in d or "HOOK" in d:
        return "HITCH"
    if "SPEED_OUT" in d:
        return "OUT"
    if "DAGGER" in d:
        return "DAGGER"
    if "CIRCUS" in d:
        return "CIRCUS"
    if "SLOT_STRIKE" in d:
        return "SLOT_STRIKE"
    if "FADE" in d:
        return "FADE"
    if "GO" in d:
        return "GO"
    return "OTHER"


rows = []
for eid, g in wr.groupby("event_id", sort=False):
    try:
        rows.append(extract_event(g))
    except Exception as ex:
        print("skip event", eid, ex)

ev = pd.DataFrame(rows)
ev["route_key"] = ev["drill_name"].map(route_key)
ev.to_csv(os.path.join(OUTD, "combine_event_feat.csv"), index=False)
print("\nevent features saved:", ev.shape)
print("route_key counts:")
print(ev["route_key"].value_counts().to_string())
print("\ncut detection rate by route (events with >=1 cut):")
det = ev.groupby("route_key").agg(n_events=("event_id", "count"),
                                  n_with_cuts=("n_cuts", lambda s: (s > 0).sum()))
det["pct"] = (det["n_with_cuts"] / det["n_events"] * 100).round(0)
print(det.to_string())

# ---- aggregate per player x route ----
feat_cols = [c for c in ev.columns if c not in
             ("event_id", "nfl_id", "drill_name", "attempt", "route_key", "n_frames", "n_ball_frames")]
pr = ev.groupby(["nfl_id", "route_key"], sort=False).agg(
    **{c: (c, "mean") for c in feat_cols},
    n_attempts=("event_id", "count"),
).reset_index()
pr.to_csv(os.path.join(OUTD, "combine_player_route.csv"), index=False)
print("\nplayer x route features saved:", pr.shape)

# ---- per-player Movement Quality Score: z-score features within drill_name, average ----
evz = ev.copy()
grp = evz.groupby("drill_name", sort=False)[feat_cols].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
for c in feat_cols:
    evz[c + "_z"] = grp[c]
zcols = [c + "_z" for c in feat_cols]
mqs = evz.groupby("nfl_id", sort=False).agg(
    mqs_mean=("speed_mean_z", "mean"),  # placeholder; replaced below
)
agg = evz.groupby("nfl_id", sort=False)[zcols].mean()
# directional fixes: lower decel_peak better (more negative a = harder decel is good? keep sign-agnostic here)
agg.columns = [c.replace("_z", "") for c in agg.columns]
mqs = agg.reset_index()
mqs["n_events"] = ev.groupby("nfl_id", sort=False).size().reindex(mqs["nfl_id"]).values
mqs.to_csv(os.path.join(OUTD, "combine_player_mqs.csv"), index=False)
print("\nplayer MQS saved:", mqs.shape)
print("\nDONE.")
