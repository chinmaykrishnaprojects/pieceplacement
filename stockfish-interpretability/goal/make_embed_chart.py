"""Chart: does the chess-LM's incidental internal state beat a purpose-built embedding?"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = {"blue": "#2a78d6", "aqua": "#1baf7a", "violet": "#4a3aa7",
     "orange": "#eb6834", "grey": "#9a9992"}
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": "#d8d7d2",
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": "#e8e7e2",
    "grid.linewidth": .8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "figure.dpi": 150})

lm = json.load(open("results/embed_compare.json"))
c2 = json.load(open("results/chess2vec_probes.json"))
b = lm["lm_best"]

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8))

# ---- LEFT: head to head on the CLEAN probes -------------------------------
names = ["who is ahead\n(side to move)", "material\n(R², rescaled)",
         "board\nreconstruction", "open vs\nclosed"]
c2v = [c2["who_ahead_acc"], max(c2["material_r2"], 0), c2["board_recon"],
       c2["open_closed_acc"]]
lmv = [b["who_ahead_acc"], max(b["material_r2"], 0), lm["lm_board_recon"]["acc"],
       b["open_closed_acc"]]
base = [0.523, 0.0, c2["board_recon_baseline"], 0.5]
x = np.arange(len(names))
w = 0.27
ax[0].bar(x - w, base, w, color=C["grey"], label="trivial baseline")
ax[0].bar(x, c2v, w, color=C["orange"], label="chess2vec (256-d, purpose-built)")
ax[0].bar(x + w, lmv, w, color=C["aqua"], label=f"chess-LM layer {b['layer']} (512-d)")
for xi, (a, c) in enumerate(zip(c2v, lmv)):
    ax[0].text(xi, a + .015, f"{a:.2f}", ha="center", fontsize=8, color=INK2)
    ax[0].text(xi + w, c + .015, f"{c:.2f}", ha="center", fontsize=8, color=INK2)
ax[0].set_xticks(x); ax[0].set_xticklabels(names, fontsize=9)
ax[0].set_ylim(0, 1.0); ax[0].set_ylabel("probe accuracy / R²")
ax[0].set_title("A language model's incidental state vs an\nembedding built to be one",
                fontsize=12)
ax[0].legend(frameon=False, fontsize=8.5, loc="upper left")
ax[0].annotate("chess2vec has NO signal here\nbeyond the move number",
               (0, 0.545), xytext=(0.34, 0.30), fontsize=8, color=C["orange"],
               arrowprops=dict(arrowstyle="->", color=C["orange"], lw=1.2))

# ---- RIGHT: layer sweep ---------------------------------------------------
L = [r["layer"] for r in lm["lm_layers"]]
for key, col, lab in [("open_closed_acc", C["blue"], "open vs closed"),
                      ("who_ahead_acc", C["aqua"], "who is ahead"),
                      ("material_r2", C["violet"], "material (R²)")]:
    ax[1].plot(L, [r[key] for r in lm["lm_layers"]], marker="o", ms=6,
               color=col, lw=2, markeredgecolor=SURF, label=lab)
ax[1].axhline(0, color="#c9c8c3", lw=1)
ax[1].axvspan(6, 11, color=C["aqua"], alpha=.07)
ax[1].annotate("chess structure peaks\nMID-STACK, then decays as\nthe model commits to the\nnext character",
               (11.4, .62), fontsize=8.5, color=INK2, linespacing=1.35)
ax[1].set_xlabel("transformer layer (of 16)")
ax[1].set_ylabel("probe score")
ax[1].set_title("Where chess lives inside the model", fontsize=12)
ax[1].legend(frameon=False, fontsize=9, loc="lower left")

fig.tight_layout()
fig.savefig("results/embed_compare.png")
print("wrote results/embed_compare.png")
