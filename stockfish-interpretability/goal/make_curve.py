"""Strength-per-LM-call curve: the frontier push, on the same axes as Stockfish.

Cost axis is LM forward passes per GAME (exact, metered by the arena).
Note for honesty: at budget=1 every candidate is capped at <=1 call per MOVE by
construction, so differences in per-game totals there reflect GAME LENGTH (a
better player survives longer), not a higher per-move price.
"""
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"blue": "#2a78d6", "aqua": "#1baf7a", "violet": "#4a3aa7",
     "red": "#e34948", "orange": "#eb6834"}
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.edgecolor": "#d8d7d2", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": "#e8e7e2", "grid.linewidth": .8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 11, "figure.dpi": 150})


def load():
    rows = []
    for line in open("wiki/results.jsonl"):
        r = json.loads(line)
        if "elo" not in r:
            continue
        r["name"] = r["candidate"].split("/")[-1].replace(".py", "")
        rows.append(r)
    return rows


def pick(rows, name, budget, games):
    for r in rows:
        if (r["name"] == name and r["budget"] == budget
                and r.get("games_per_rung") == games):
            return r
    return None


def main():
    rows = load()
    g = 20
    series = [
        ("baseline_policy", "baseline: policy argmax", "#9a9992", "o"),
        ("blunder_filter", "blunder filter (SEE + quiescence)", C["orange"], "D"),
        ("policy_negamax", "policy-pruned negamax", C["violet"], "s"),
        ("adaptive_budget", "adaptive budget (winner)", C["aqua"], "o"),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    # Stockfish reference: the opponent rung we measured against
    ax.axhline(1468, color=C["blue"], lw=1.4, ls="--", alpha=.8)
    ax.annotate("Stockfish @ 32 nodes = 1468 Elo (the opponent)", (31, 1478),
                fontsize=8.5, color=C["blue"])

    for name, label, col, mk in series:
        pts = []
        for b in (1, 4):
            r = pick(rows, name, b, g)
            if r:
                pts.append((r["lm_calls_total"] / g, r["elo"], b))
        if not pts:
            continue
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=col, lw=2, marker=mk, ms=8,
                markeredgecolor=SURF, markeredgewidth=1.2, label=label, zorder=4)
        for x, y, b in pts:
            ax.annotate(f"{y:.0f}", (x, y), xytext=(0, 9),
                        textcoords="offset points", ha="center",
                        fontsize=8.5, color=INK)

    # the headline: the vertical free gain at ~1 call/move
    base = pick(rows, "baseline_policy", 1, g)
    win = pick(rows, "adaptive_budget", 1, g)
    if base and win:
        x0, x1 = base["lm_calls_total"] / g, win["lm_calls_total"] / g
        ax.annotate("", xy=(x1 * 1.02, win["elo"] - 8),
                    xytext=(x0 * 1.02, base["elo"] + 8),
                    arrowprops=dict(arrowstyle="->", color=C["aqua"], lw=2.4,
                                    shrinkA=2, shrinkB=2))
        ax.annotate("+300 Elo at the same per-move\nmodel cost — bought with free\n"
                    "arithmetic (SEE + quiescence),\nnot with more compute",
                    (46, 1180), fontsize=9.5, color=C["aqua"], ha="left",
                    weight="bold", linespacing=1.35)

    ax.set_xscale("log")
    ax.set_xlabel("LM forward passes per game  (measured by the harness, log scale)")
    ax.set_ylabel("Elo  (from games vs Stockfish at fixed nodes)")
    ax.set_title("Pushing the interpretable chess frontier\n"
                 "a frozen 50M human-trained LM: strength per model call",
                 fontsize=13)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_ylim(1140, 1600)
    fig.tight_layout()
    fig.savefig("results/frontier_curve.png")
    print("wrote results/frontier_curve.png")

    # console summary
    print(f"\n{'candidate':20s} {'b=1':>18s} {'b=4':>18s}")
    for name, label, _, _ in series:
        r1, r4 = pick(rows, name, 1, g), pick(rows, name, 4, g)
        f = lambda r: (f"{r['elo']:.0f} Elo/{r['lm_calls_total']/g:.0f}c"
                       if r else "-")
        print(f"{name:20s} {f(r1):>18s} {f(r4):>18s}")


if __name__ == "__main__":
    main()
