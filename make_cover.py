# -*- coding: utf-8 -*-
"""Cover card v3: 560x280, cleaner hierarchy + mini football-field motif."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

NAVY = "#013369"
PANEL = "#01244d"
RED = "#D50A0A"
GOLD = "#F2A900"
WHITE = "#F5F8FC"
LBLUE = "#9db8d8"

fig = plt.figure(figsize=(5.6, 2.8), dpi=100)  # 560 x 280 px
ax = fig.add_axes([0, 0, 1, 1])
ax.set_facecolor(NAVY)

# ---------- right panel: mini field with route speed profile ----------
px0, py0, pw, ph = 0.545, 0.10, 0.425, 0.80
ax.add_patch(Rectangle((px0, py0), pw, ph, facecolor=PANEL,
                       edgecolor="#0d3d74", lw=1.4))
# yard lines
for k in range(1, 15):
    xx = px0 + k * (pw / 15)
    ax.plot([xx, xx], [py0 + 0.04, py0 + ph - 0.04], color=WHITE, alpha=0.08, lw=1.0)
# center hash
ax.plot([px0, px0 + pw], [py0 + ph / 2, py0 + ph / 2], color=WHITE, alpha=0.06, lw=1.0)

gx = np.linspace(px0 + 0.045, px0 + pw - 0.045, 260)
t = (gx - gx.min()) / (gx.max() - gx.min())
gy = 0.32 - 0.085 * np.exp(-((t - 0.42) ** 2) / 0.0045) + 0.012 * np.sin(t * 55)
gy = gy / gy.max() * 0.34 + py0 + 0.22
ax.plot(gx, gy, color=GOLD, lw=3.0, solid_capstyle="round", zorder=5)

i_b = int(np.argmin(gy))
ax.axvspan(gx[max(0, i_b - 10)], gx[min(len(gx) - 1, i_b + 10)],
           color=GOLD, alpha=0.15, zorder=4)
ax.add_patch(Circle((gx[i_b], gy[i_b]), 0.012, facecolor=GOLD,
                    edgecolor=WHITE, lw=1.2, zorder=6))
ax.annotate("break", xy=(gx[i_b], gy[i_b] - 0.015),
            xytext=(gx[i_b] + 0.055, py0 + 0.60), fontsize=8.2, color=GOLD,
            ha="center", arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.4), zorder=7)

ax.text(px0 + pw / 2, py0 + ph - 0.075, "route speed through the cut",
        fontsize=7.6, color=WHITE, alpha=0.88, ha="center")
ax.text(px0 + pw / 2, py0 + 0.045, "sharp cuts \u00b7 speed retention \u00b7 consistency",
        fontsize=7.0, color=LBLUE, ha="center")

# ---------- left: text block ----------
ax.text(0.045, 0.905, "NFL BIG DATA BOWL 2027 \u00b7 WIDE RECEIVERS",
        fontsize=8.0, color=RED, fontweight="bold", va="top")
ax.text(0.045, 0.795, "BEYOND THE", fontsize=19.5, fontweight="bold",
        color=WHITE, va="top", family="DejaVu Sans")
ax.text(0.045, 0.645, "STOPWATCH", fontsize=19.5, fontweight="bold",
        color=GOLD, va="top", family="DejaVu Sans")
ax.text(0.045, 0.495, "Combine Route-Running Mechanics", fontsize=10.2,
        color=LBLUE, va="top")
ax.text(0.045, 0.405, "Predict NFL Separation", fontsize=10.2,
        color=LBLUE, va="top")

ax.plot([0.045, 0.415], [0.315, 0.315], color=RED, lw=2.0, solid_capstyle="round")

ax.text(0.045, 0.215, "Sensor movement R\u00b2 0.19  vs  forty 0.14", fontsize=8.4,
        color=WHITE, alpha=0.96, va="top")
ax.text(0.045, 0.125, "Top-15 starter hit rate 80% \u2192 93%", fontsize=8.4,
        color=WHITE, alpha=0.96, va="top")
ax.text(0.045, 0.038, "10 Hz tracking \u00b7 1,529 Combine drills \u00b7 28,830 NFL routes",
        fontsize=7.2, color="#6f8cb0", va="bottom")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.savefig(r"D:\kaggle\NFL2027\submission\cover_card.png", dpi=100, facecolor=NAVY)
plt.close(fig)

from PIL import Image
im = Image.open(r"D:\kaggle\NFL2027\submission\cover_card.png")
print("cover size:", im.size)
