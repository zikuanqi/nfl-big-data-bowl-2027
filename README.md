# Beyond the Stopwatch: Combine Route-Running Mechanics Predict NFL Separation

**NFL Big Data Bowl 2027 — Wide Receiver Route Running**

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-NFL%20Big%20Data%20Bowl%202027-20BEFF)](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027)
[![Public Notebook](https://img.shields.io/badge/Kaggle-Notebook-20BEFF)](https://www.kaggle.com/code/zikuanqi/combine-route-running-predicts-nfl-separation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 中文摘要（Chinese Summary）

本项目是 NFL Big Data Bowl 2027 的参赛作品，研究方向为**外接手（WR）路线跑动**。核心问题：**Combine 体测营里"怎么跑路线"的传感器运动特征，能否预测进入 NFL 后的摆脱（separation）能力？**

结论：能，而且比 40 码秒表成绩更准。从 1,529 次 Combine 训练追踪中提取「切步锐度、过弯速度保持、速度稳定性」等运动学特征后，发现它们能单独解释 **19.1%** 的比赛摆脱方差（40 码只有 13.6%）；在生涯预测上，选秀资本 + 运动特征的 LOOCV R² 从 0.397 提升到 **0.426**，前 15 名主力命中率从 80% 提升到 **93%**；对三轮后秀和落选秀（Day 3 / UDFA）同样有效（R² 0.163 vs 路线基线 0.108）。案例：第五轮秀 Puka Nacua 拥有 2023 届最锐利的变向（133° vs 均值 85°），生涯首发 43 场。

---

## TL;DR — Key Findings

| Finding | Evidence |
|---|---|
| **Combine route mechanics → NFL separation** | 1,529 Combine drill attempts (106 WRs) linked to 28,830 in-game routes; movement signature alone explains **19.1%** of route-level separation variance (n = 494 player-routes, weighted ridge, player-clustered bootstrap CIs) |
| **Sharp, efficient cuts carry the signal** | Sharpest cut angle **+0.055/SD**, speed variability **−0.035/SD**, speed loss at break **−0.047/SD** (95% bootstrap CIs exclude 0) |
| **Beats the stopwatch** | Movement-only R² **0.188** vs 40-yard-dash-only **0.136**; forty + movement = 0.190 (movement subsumes the forty) |
| **Survives into Day 3 / UDFA** | Late-pick subgroup (74 players, 347 player-routes): movement R² **0.163** vs route-type baseline 0.108 |
| **Incremental career value** | LOOCV (n = 106): draft capital R² 0.397 → draft + movement **0.426**; RMSE 8.00 → 7.81; top-15 starter hit rate (≥11 career starts) **80% → 93%** |
| **Honest limits** | No gain for career offensive snaps; movement alone is a weak career predictor (R² 0.08) — early-career opportunity is dominated by draft capital. The signature is a *tier tie-breaker*, not a standalone draft grade |

## Background

Every March, NFL evaluators reduce wide receiver movement to a handful of stopwatch numbers — the 40-yard dash, 3-cone, short shuttle. Those measure straight-line speed and standardized agility, not the skill that earns a receiver snaps on Sundays: **creating separation**. The 2027 Big Data Bowl provides 10 Hz optical tracking of every Combine drill alongside three seasons of in-game tracking (with *separation at pass forward*) for the same 510 prospects.

This project asks: **do the mechanics of how a prospect runs a route at the Combine predict how much separation he creates in the NFL?**

## Data

| File | Size | Description |
|---|---|---|
| `combine_tracking.csv` | 80 MB | 10 Hz sensor frames for 6,310 Combine drill attempts; 1,529 are WR skill-drill events (23,975 ball frames) |
| `player_play.csv` | 140 MB | Per-play in-game NGS metrics; `route_ran` + `separation_at_pass_forward` (37,525 non-null WR routes) |
| `game_tracking_2023/24/25.csv` | 320 / 685 / 975 MB | 10 Hz in-game tracking of cohort players (3.7M / 7.8M / 11.2M rows) |
| `players.csv` | 47 KB | 510 prospects (2023–2025 drafts), positions, draft capital |
| `combine_results.csv` | 41 KB | Official Combine measurements + NGS scores |
| `player_career_successes.csv` | 14 KB | Career snaps, games active/started, All-Pro, Pro Bowl |
| `games.csv` | 43 KB | Schedule metadata for 1,002 games |

> Raw data is **not** committed to this repo. Download via Kaggle CLI (see Reproduction) or use the public notebook with `competition_sources`.

## Methodology

### 1. Movement signature (feature engineering)

For each of the 1,529 WR skill-drill attempts (`fe_combine.py`):

- **Cut detection** — from the smoothed NGS direction series, flag every ≥25° turn within a 0.9 s window
- **Break characterization** — speed trough (global minimum in the middle 25–85% of the route): speed loss, deceleration distance, re-acceleration after the break
- **Per-cut features** — speed retention (post-cut ÷ pre-cut speed, winsorized at 2.0), sharpest/mean cut angle, deceleration into the cut, acceleration out of it
- **Global features** — peak/mean speed, speed CV, peak acceleration/deceleration, total turn, COD index, duration, distance

Features are averaged per **player × route type** and z-scored within route. Route mapping: `SLANT_ROUTE↔slant`, `CURL_ROUTE↔hitch`, `SPEED_OUT↔out`, `POST_CORNER↔post/corner`, `GO_ROUTE↔go`.

### 2. Linkage model (`link_separation.py`)

Weighted ridge regression at the **player × route level** (n = 494): mean log separation ~ movement signature + route dummies, with coverage (man/zone), cushion, target rate, and slot rate as controls. Inference via **player-clustered bootstrap** (500 resamples, 95% CIs).

### 3. External validation (`extra_analyses.py`, `career_validation.py`)

- **Beyond the stopwatch** — forty-only vs movement-only vs both (n = 447 with forty recorded)
- **Day-3 subgroup** — same linkage within picks > 100 and UDFAs
- **Career validation** — leave-one-out cross-validated ridge predicting career games started; model comparison: draft capital only → draft + traditional combine → draft + movement; scouting-style top-15 starter hit rates

### 4. Case studies

| Player | Draft pick | Signature (vs cohort) | In-game sep | Career starts |
|---|---|---|---|---|
| **Puka Nacua** | 177 | Sharpest cut **133°** (85°) | 2.07 yd | **43** |
| **Jaxon Smith-Njigba** | 20 | Retention **1.18** (1.16) | 2.41 yd | **36** |
| **Quentin Johnston** | 21 | Sharpest cut 95°, retention 1.33 | 2.12 yd | 31 |
| *Cohort mean (n = 106)* | — | — | 2.32 yd | 7.3 |

Note: all three stars skipped the 40-yard dash — the tracking signature covers what the stopwatch misses.

## Repository Structure

```
├── README.md                        this file
├── LICENSE                          MIT
├── fe_combine.py                    step 1: 22 kinematic features per Combine drill attempt
├── link_separation.py               step 2: L1 play-level + L2 player×route models, bootstrap CIs
├── extra_analyses.py                step 3: forty-vs-movement, Day-3 subgroup, per-route profiles, case studies
├── career_validation.py             step 4: LOOCV incremental career validation + hit rates
├── viz.py                           step 5: the 7 writeup figures
├── build_notebook.py                step 6: builds the Kaggle notebook programmatically
├── make_cover.py                    writeup cover card (560×280)
├── eda.py / wr_feasibility.py       exploratory analyses
├── nfl2027_wr_analysis.ipynb        executed public notebook (also on Kaggle)
├── features/                        intermediate outputs (event features, player×route tables, model JSON)
└── submission/                      exact submission package
    ├── writeup.md                   full writeup (1,444 words, 7 figures, 1 table)
    ├── project_description.md       body for the Kaggle writeup form
    ├── FORM_FIELDS.md               every form field filled in
    ├── cover_card.png               560×280 card & thumbnail
    ├── figs/                        the 7 figures (PNG)
    └── nfl2027_wr_analysis.ipynb
```

## Reproduction

### Prerequisites

- Python 3.11+, packages: `pandas`, `numpy`, `scikit-learn`, `scipy`, `matplotlib` (optionally `statsmodels`)
- Kaggle CLI with API credentials (for data download)

### Steps

```bash
# 1. clone
git clone git@github.com:zikuanqi/nfl-big-data-bowl-2027.git
cd nfl-big-data-bowl-2027

# 2. download competition data (2.2 GB, not committed)
kaggle competitions download -c nfl-big-data-bowl-2027 -p .
unzip nfl-big-data-bowl-2027.zip          # creates nfl-big-data-bowl-2027/ with 9 CSVs

# 3. run the pipeline
python fe_combine.py                      # -> features/combine_event_feat.csv, combine_player_route.csv, combine_player_mqs.csv
python link_separation.py                 # -> features/analysis_player_route.csv, results_model.json
python extra_analyses.py                  # -> features/extra_results.json (stopwatch, late picks, case studies)
python career_validation.py               # -> features/career_results.json (LOOCV + hit rates)
python viz.py                             # -> figs/*.png
python build_notebook.py                  # -> nfl2027_wr_analysis.ipynb
```

The notebook auto-detects the data path and also runs on Kaggle (mount: `/kaggle/input/competitions/nfl-big-data-bowl-2027/...`).

## Kaggle Submission

- **Competition**: [nfl-big-data-bowl-2027](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027) (deadline **2027-01-06 23:59 UTC**)
- **Public notebook**: [combine-route-running-predicts-nfl-separation](https://www.kaggle.com/code/zikuanqi/combine-route-running-predicts-nfl-separation) — executes end-to-end on Kaggle with identical numbers
- **Writeup**: `submission/writeup.md` (1,444 words, 7 figures, 1 table — within the 2,000-word / 10-figure limits)
- **Track**: Open Track

## Git Workflow (this repo)

Commits are **SSH-signed** and show `Verified` on GitHub:

- **Authentication key**: `~/.ssh/id_ed25519` (registered on the account)
- **Signing key**: `~/.ssh/id_github_zikuanqi` (registered as Signing key)
- Local git config: `gpg.format=ssh`, `user.signingkey=~/.ssh/id_github_zikuanqi.pub`, `commit.gpgsign=true`, `user.email=zikuanqi@outlook.com`

```bash
git add -A && git commit -m "message" && git push
```

## License

- Analysis code: MIT (see [LICENSE](LICENSE))
- Competition data: © NFL / Kaggle, [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

## Acknowledgments

NFL & AWS for the Big Data Bowl; the competition provides tracking data from Next Gen Stats. Puka Nacua, Jaxon Smith-Njigba, and Quentin Johnston appear as case studies; all players consented to tracking data collection as part of league operations.
