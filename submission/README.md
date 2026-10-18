# NFL Big Data Bowl 2027 — Submission Package

**Team / Account**: zikuanqi · **Competition**: [nfl-big-data-bowl-2027](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027)
**Title**: Beyond the Stopwatch: Combine Route-Running Mechanics Predict NFL Separation

## 文件夹内容

| 文件 | 用途 |
|---|---|
| `writeup.md` | 参赛报告全文（1444 词 + 7 图 + 1 表，符合官方限制：≤2000 词、<10 图表）。图片引用为相对路径 `figs/...`，在本文件夹中用 Markdown 预览器可直接查看完整排版 |
| `figs/fig1~fig7` | 报告中的 7 张图（PNG），在 Kaggle Writeup 编辑器里拖入对应位置即可 |
| `nfl2027_wr_analysis.ipynb` | 公开 Notebook（已执行，含全部输出）。云端版本：https://www.kaggle.com/code/zikuanqi/combine-route-running-predicts-nfl-separation |
| `SUBMIT_GUIDE.md` | 最后一步的网页提交步骤（约 5 分钟） |

## 提交要点

1. 打开 https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/writeups → Create Writeup
2. Title: **Beyond the Stopwatch: Combine Route-Running Mechanics Predict NFL Separation**
3. Track: **Open Track**（在校学生可选 University Track）
4. 粘贴 `writeup.md` 正文，7 个 `![](figs/...)` 占位处拖入对应 PNG
5. 关联公开 Notebook → Submit（截止 **2027-01-06 23:59 UTC**）

## 核心结论速览

- Combine 路线跑动的「刹车-出弯」运动签名可解释 **19.1%** 的比赛摆脱（separation）方差
- 传感器运动特征预测摆脱 **完胜 40 码成绩**（R² 0.188 vs 0.136）
- 生涯验证：选秀资本 + 运动签名，LOOCV R² **0.397 → 0.426**；前 15 名主力命中率 **80% → 93%**
- 信号在 Day-3 秀/UDFA 中依然有效（R² 0.163 vs 路线基线 0.108）
