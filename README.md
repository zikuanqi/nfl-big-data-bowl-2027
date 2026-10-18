# Beyond the Stopwatch: Combine Route-Running Mechanics Predict NFL Separation

**NFL Big Data Bowl 2027 — Wide Receiver Route Running** · [Competition page](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027) · [Public notebook](https://www.kaggle.com/code/zikuanqi/combine-route-running-predicts-nfl-separation)

> **TL;DR** — Using 10 Hz Combine sensor tracking, we show that *how* a receiver runs a route (cut sharpness, speed retention through breaks, speed consistency) predicts in-game **separation** better than the 40-yard dash (weighted R² 0.19 vs 0.14), adds incremental career value beyond draft capital (LOOCV R² 0.397 → 0.426; top-15 starter hit rate 80% → 93%), and still works for Day-3 picks and UDFAs (R² 0.163 vs 0.108 route baseline).

## Key findings

| Finding | Evidence |
|---|---|
| Combine route mechanics → NFL separation | 1,529 Combine drill attempts (106 WRs) → 28,830 in-game routes; movement signature alone explains **19.1%** of route-level separation variance (n = 494 player-routes, player-clustered bootstrap CIs) |
| Sharp, efficient cuts carry the signal | Sharpest cut angle **+0.055/SD**, speed variability **−0.035/SD**, speed loss at break **−0.047/SD** (95% bootstrap CIs exclude 0) |
| Beats the stopwatch | Movement-only R² **0.188** vs 40-yard-dash-only **0.136**; combining adds almost nothing (0.190) |
| Survives into Day 3 | Late-pick/UDFA subgroup (n = 347): movement R² **0.163** vs route-type baseline 0.108 |
| Honest limits | Early-career opportunity is dominated by draft capital; movement signal is a tier tie-breaker, not a standalone grade |

## Repository structure

```
submission/                  ← exact submission package (writeup + figures + notebook)
  writeup.md                   full writeup (1,444 words, 7 figures, 1 table)
  project_description.md       body for the Kaggle writeup form
  FORM_FIELDS.md               every form field filled in (title/subtitle/track/cover)
  cover_card.png               560×280 card & thumbnail image
  figs/                        the 7 figures (PNG)
  nfl2027_wr_analysis.ipynb    executed public notebook
fe_combine.py               feature extraction: 22 kinematic features per Combine drill attempt
link_separation.py          L1 play-level + L2 player×route ridge models with bootstrap inference
career_validation.py        LOOCV incremental career validation + scouting hit rates
extra_analyses.py           forty-vs-movement, late-pick subgroup, per-route profiles, case studies
viz.py                      figure generation
build_notebook.py           builds the Kaggle notebook programmatically
eda.py / wr_feasibility.py  exploratory analysis scripts
features/                   intermediate outputs (event features, player×route tables, model JSON)
```

## Reproduce

```bash
# 1. download competition data (Kaggle CLI, requires kaggle.json)
kaggle competitions download -c nfl-big-data-bowl-2027 -p .

# 2. run the pipeline (Python 3.11, pandas/scikit-learn/scipy/matplotlib)
python fe_combine.py          # combine_tracking.csv -> features/combine_event_feat.csv
python link_separation.py     # movement -> separation models
python extra_analyses.py      # stopwatch comparison, late picks, case studies
python career_validation.py   # LOOCV career validation
python viz.py                 # figures/
python build_notebook.py      # nfl2027_wr_analysis.ipynb
```

## Notes

- Raw data (2.2 GB) is **not** committed — download via the Kaggle CLI or in a notebook with `competition_sources: ["nfl-big-data-bowl-2027"]`.
- All model claims use player-clustered bootstrap 95% CIs; all predictive claims use leave-one-out cross-validation.
- License: analysis code MIT; competition data © NFL / Kaggle (CC BY-NC 4.0).
