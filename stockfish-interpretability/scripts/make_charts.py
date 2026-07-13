"""Render all report charts as PNGs (light surface, validated palette)."""
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit

# validated reference palette (dataviz skill), light mode
C = {"blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100", "green": "#008300",
     "violet": "#4a3aa7", "red": "#e34948", "magenta": "#e87ba4", "orange": "#eb6834"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": "#d8d7d2", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "figure.dpi": 140,
})

A = json.load(open("results/analysis.json"))
W = json.load(open("results/win_prob_fit.json"))


def chart_eval_score():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = [b for b in A["eval_to_score"]["empirical_bins"] if b["n"] >= 200]
    x = [b["cp"] / 100 for b in bins]
    y = [b["score"] for b in bins]
    k, b0 = A["eval_to_score"]["k_cp"], A["eval_to_score"]["bias_cp"]
    xs = np.linspace(-10, 10, 300)
    ax.plot(xs, expit((xs * 100 + b0) / k), color=C["blue"], lw=2,
            label=f"expected score fit (k={k:.0f}cp)")
    kw, bw = W["win_k"], W["win_bias"]
    ax.plot(xs, expit((xs * 100 + bw) / kw), color=C["orange"], lw=2, ls="--",
            label=f"P(White wins) fit")
    ax.scatter(x, y, s=42, color=C["blue"], zorder=3, edgecolor=SURFACE, lw=1)
    ax.axhline(0.5, color="#c9c8c3", lw=1)
    cp50 = W["cp_for_50pct_win"] / 100
    ax.axvline(cp50, color=C["orange"], lw=1, ls=":")
    ax.annotate(f"50% win chance at +{cp50:.2f},\nnot +1.00", (cp50, 0.18),
                xytext=(2.4, 0.12), color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlabel("Stockfish depth-18 eval (pawns, White POV)")
    ax.set_ylabel("actual result (White POV)")
    ax.set_title("What an eval is worth: lichess games vs their depth-18 evals")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(-8, 8); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig("results/chart_eval_score.png"); plt.close(fig)


def chart_piece_values():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pieces = ["p", "n", "b", "r", "q"]
    names = ["Pawn", "Knight", "Bishop", "Rook", "Queen"]
    classical = [1, 3, 3, 5, 9]
    sf = [A["piece_values"]["sf_eval_implied_pawn_units"][p] for p in pieces]
    outc = [A["piece_values"]["outcome_implied_pawn_units"][p] for p in pieces]
    xpos = np.arange(5)
    wdt = 0.27
    for off, vals, col, lab in [(-wdt, classical, "#b5b4ae", "classical (1/3/3/5/9)"),
                                (0, sf, C["blue"], "SF depth-18 eval–implied"),
                                (wdt, outc, C["aqua"], "outcome-implied (amateur conversion)")]:
        bars = ax.bar(xpos + off, vals, width=wdt - 0.03, color=col, label=lab, zorder=3)
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.12, f"{v:.1f}",
                    ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(xpos); ax.set_xticklabels(names)
    ax.set_ylabel("value (pawn = 1)")
    ax.set_title("Reverse-engineered piece values (115k quiet positions)")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig("results/chart_piece_values.png"); plt.close(fig)


def chart_conversion():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    ax = axes[0]
    rows = A["trade_when_ahead"]
    labels = ["≤20", "20–35", "35–50", "50–65", "65–80"]
    y = [r["score"] for r in rows]
    ax.plot(labels, y, color=C["blue"], lw=2, marker="o", ms=7,
            markeredgecolor=SURFACE)
    for i, r in enumerate(rows):
        ax.annotate(f"{r['score']:.0%}", (i, y[i] + 0.012), ha="center",
                    fontsize=9, color=INK2)
    ax.set_xlabel("total material remaining (both sides, pawn units)")
    ax.set_ylabel("expected score for the side up 2–4")
    ax.set_title("When ahead, trade: same edge,\nless material → better conversion")
    ax.set_ylim(0.55, 0.87)

    ax = axes[1]
    rows = A["leader_pawnshare_effect"]
    labels = ["Q1\n(piece-heavy)", "Q2", "Q3", "Q4\n(pawn-heavy)"]
    y = [r["score"] for r in rows]
    ax.bar(labels, y, color=C["aqua"], width=0.55, zorder=3)
    for i, v in enumerate(y):
        ax.text(i, v + 0.008, f"{v:.0%}", ha="center", fontsize=9, color=INK2)
    ax.set_ylabel("expected score for the side up 2–4")
    ax.set_title("Leader's pawn share:\npawn-heavy advantages convert best")
    ax.set_ylim(0.55, 0.87)
    fig.tight_layout(); fig.savefig("results/chart_conversion.png"); plt.close(fig)


def elo_from_score(s, n):
    s = min(max(s, 0.5 / n), 1 - 0.5 / n)
    return -400 * math.log10(1 / s - 1)


def ladder_elos():
    df = pd.read_csv("results/ladder_results.csv", dtype={"low": str, "high": str})
    levels = [32, 64, 256, 1024, 4096, 16384, 65536]
    elo = {65536: 0.0}
    for hi, lo in zip(reversed(levels[1:]), reversed(levels[:-1])):
        g = df[(df.low == str(lo)) & (df.high == str(hi))]
        s = g.score_low.mean()
        elo[lo] = elo[hi] + elo_from_score(s, len(g))  # negative diff
    # material engine placed vs 32 and 64
    ms = []
    for hi in (32, 64):
        g = df[(df.low == "MATERIAL") & (df.high == str(hi))]
        if len(g):
            ms.append(elo[hi] + elo_from_score(g.score_low.mean(), len(g)))
    mat_elo = float(np.mean(ms)) if ms else None
    return elo, mat_elo


def chart_ladder(elo, mat_elo, anchor):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = sorted(elo)
    ys = [elo[x] + anchor for x in xs]
    ax.plot(xs, ys, color=C["blue"], lw=2, marker="o", ms=7,
            markeredgecolor=SURFACE, label="SF16 NNUE, fixed nodes/move")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.0f}", (x, y + 55), ha="center", fontsize=8.5, color=INK2)
    if mat_elo is not None:
        ax.scatter([32], [mat_elo + anchor], color=C["red"], s=70, zorder=4,
                   label="material-only αβ (fully interpretable)")
        ax.annotate(f"{mat_elo + anchor:.0f}", (32, mat_elo + anchor - 130),
                    ha="center", fontsize=8.5, color=INK2)
    # interpretable no-/low-search engines at the 32-node abscissa
    try:
        import pandas as _pd, math as _m
        rr = _pd.read_csv("results/ruleset_results.csv", dtype={"low": str, "high": str})
        def _elo(s, n):
            s = min(max(s, 0.5 / n), 1 - 0.5 / n); return -400 * _m.log10(1 / s - 1)
        base = {"32": elo[32] + anchor, "64": elo[64] + anchor}
        for name, col, lab, dx in [
            ("RULESET", C["aqua"], "rule-set αβ (piece values + PST principles)", 1.25),
            ("EVALONLY", C["violet"], "SF NNUE as pure policy (depth 1, no search)", 0.8)]:
            est = [base[hi] + _elo(rr[(rr.low == name) & (rr.high == hi)].score_low.mean(),
                   len(rr[(rr.low == name) & (rr.high == hi)])) for hi in ("32", "64")]
            e = sum(est) / 2
            ax.scatter([32 * dx], [e], color=col, s=70, zorder=4, label=lab)
            ax.annotate(f"{e:.0f}", (32 * dx, e + 50), ha="center", fontsize=8, color=INK2)
    except FileNotFoundError:
        pass
    ax.set_xscale("log", base=2)
    ax.set_xlabel("nodes per move (log₂)")
    ax.set_ylabel(f"Elo (anchored: 65 536 nodes ≈ {anchor})")
    ax.set_title("Strength per node: Elo vs search budget (self-play ladder)")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig("results/chart_ladder.png"); plt.close(fig)
    return dict(zip([str(x) for x in xs], ys)), (mat_elo + anchor if mat_elo else None)


def chart_pareto(systems):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for s in systems:
        ax.scatter(s["cost"], s["elo"], s=110, color=s["color"], zorder=4,
                   edgecolor=SURFACE, lw=1.2)
        ax.annotate(f"{s['name']}\n(interp {s['interp']}/10)",
                    (s["cost"], s["elo"]), xytext=(s.get("dx", 0), s.get("dy", 12)),
                    textcoords="offset points", ha="center", fontsize=8.3,
                    color=INK)
    # frontier line: sort by cost, upper envelope
    pts = sorted(systems, key=lambda s: s["cost"])
    env, best = [], -1e9
    for p in pts:
        if p["elo"] > best:
            env.append(p); best = p["elo"]
    ax.plot([p["cost"] for p in env], [p["elo"] for p in env],
            color="#c9c8c3", lw=1.2, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("marginal cost per move (USD, log scale)")
    ax.set_ylabel("playing strength (Elo, anchored estimates)")
    ax.set_title("The chess strength–cost–interpretability frontier")
    fig.tight_layout(); fig.savefig("results/chart_pareto.png"); plt.close(fig)


def main():
    chart_eval_score()
    chart_piece_values()
    chart_conversion()
    try:
        elo, mat = ladder_elos()
        ANCHOR = 3100  # assumed Elo of SF16 @ 65k nodes/move (~depth 17); see report
        abs_elo, mat_abs = chart_ladder(elo, mat, ANCHOR)
        json.dump({"relative": {str(k): v for k, v in elo.items()},
                   "anchored": abs_elo, "material_anchored": mat_abs,
                   "anchor_assumption": ANCHOR},
                  open("results/ladder_elo.json", "w"), indent=2)
        print("ladder elos:", abs_elo, "material:", mat_abs)
    except FileNotFoundError:
        print("ladder results not ready; skipping ladder/pareto")
        return
    # cost/move: CPU-second ≈ $1.4e-5 (c7 on-demand /4 cores); SF ~1.4M nps
    cpu = 1.4e-5
    nps = 1.4e6
    def sf_cost(nodes): return max(nodes / nps * cpu, 1e-9)
    systems = [
        dict(name="material-only αβ", elo=mat_abs, cost=3e-7, interp=9, color=C["red"]),
    ]
    systems = [s for s in systems if s["elo"] is not None] + [
        dict(name="SF16 @ 32 nodes", elo=abs_elo["32"], cost=sf_cost(32), interp=5, color=C["blue"], dy=-22),
        dict(name="SF16 @ 1k nodes", elo=abs_elo["1024"], cost=sf_cost(1024), interp=4, color=C["blue"]),
        dict(name="SF16 @ 65k nodes", elo=abs_elo["65536"], cost=sf_cost(65536), interp=4, color=C["blue"]),
        dict(name="SF16 full (~1s)", elo=3500, cost=sf_cost(1.4e6), interp=4, color=C["blue"], dy=-24),
        dict(name="SF16 @ 65k, 3-line explain", elo=abs_elo["65536"] - 215, cost=sf_cost(65536), interp=6, color=C["green"], dx=-62, dy=-30),
        dict(name="Maia-1900 (policy)", elo=1900, cost=1e-6, interp=3, color=C["violet"], dx=-14),
        dict(name="Chess-GPT 50M (your app)", elo=1500, cost=5e-7, interp=4, color=C["aqua"], dx=18, dy=14),
        dict(name="Leela policy-only", elo=2500, cost=2e-5, interp=3, color=C["yellow"]),
        dict(name="Leela + MCTS", elo=3400, cost=2e-3, interp=5, color=C["yellow"], dy=-24),
        dict(name="frontier LLM (Gemini-class)", elo=1650, cost=2e-2, interp=9, color=C["orange"]),
    ]
    chart_pareto(systems)
    json.dump(systems, open("results/pareto_points.json", "w"), indent=2)
    print("charts done")


if __name__ == "__main__":
    main()


def chart_multipv():
    df = pd.read_csv("results/multipv_results.csv")
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ks, costs, errs = [], [], []
    for k, g in df.groupby("k"):
        s, n = g.score_mpv.mean(), len(g)
        se = math.sqrt(max(s * (1 - s), 1e-9) / n)
        d = elo_from_score(s, n)
        lo = elo_from_score(min(s + se, 1 - 1e-6), n)
        hi = elo_from_score(max(s - se, 1e-6), n)
        ks.append(k); costs.append(-d); errs.append([abs(-d + lo), abs(-hi + d)])
    x = np.arange(len(ks))
    ax.bar(x, costs, width=0.5, color=C["blue"], zorder=3)
    ax.errorbar(x, costs, yerr=np.array(errs).T, fmt="none",
                ecolor=INK2, elinewidth=1.2, capsize=4, zorder=4)
    for i, c in enumerate(costs):
        ax.text(i, c + 28, f"−{c:.0f} Elo", ha="center", fontsize=9.5, color=INK2)
    # nodes-to-recover annotation using measured 16k->65k slope (~140/doubling)
    ax.set_xticks(x)
    ax.set_xticklabels([f"MultiPV={k}\n({k} candidate lines)" for k in ks])
    ax.set_ylabel("Elo cost vs normal SF16, same 16 384 nodes")
    ax.set_title("The measured price of interpretability:\nforcing Stockfish to resolve k candidate lines")
    ax.set_ylim(0, max(costs) * 1.35)
    fig.tight_layout(); fig.savefig("results/chart_multipv.png"); plt.close(fig)


chart_multipv()


def sf_win_permille(v_norm, ply=64):
    v = v_norm * 328 / 100.0
    m = min(240, ply) / 64.0
    a_s = [0.38036525, -2.82015070, 23.17882135, 307.36768407]
    b_s = [-2.29434733, 13.27689788, -14.26828904, 63.45318330]
    a = ((a_s[0]*m+a_s[1])*m+a_s[2])*m+a_s[3]
    b = ((b_s[0]*m+b_s[1])*m+b_s[2])*m+b_s[3]
    x = np.clip(v, -4000, 4000)
    return 1000/(1+np.exp((a-x)/b))


def chart_engine_vs_human():
    """Stockfish's own engine WDL model vs the human-fit curve."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xs = np.linspace(-6, 6, 400)
    # human expected score + human win prob (from fits)
    k, b0 = A["eval_to_score"]["k_cp"], A["eval_to_score"]["bias_cp"]
    kw, bw = W["win_k"], W["win_bias"]
    ax.plot(xs, expit((xs*100+bw)/kw), color=C["orange"], lw=2,
            label="human P(win)  (lichess ~1200-2200)")
    # SF engine win prob + expected score
    sfw = np.array([sf_win_permille(x*100)/1000 for x in xs])
    sfl = np.array([sf_win_permille(-x*100)/1000 for x in xs])
    sfscore = sfw + 0.5*(1-sfw-sfl)
    ax.plot(xs, sfw, color=C["blue"], lw=2,
            label="Stockfish P(win)  (engine LTC self-play)")
    ax.plot(xs, sfscore, color=C["blue"], lw=2, ls="--",
            label="Stockfish expected score")
    ax.axhline(0.5, color="#c9c8c3", lw=1)
    ax.axvline(1.0, color="#c9c8c3", lw=1, ls=":")
    ax.annotate("at +1.00 (SF's calibration point):\nSF 50% win but 75% expected score;\nhuman only ~52% win (they draw less,\nbut small edges rarely decide)",
                (1.0, 0.5), xytext=(1.5, 0.24), fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlabel("normalized eval (pawns)")
    ax.set_ylabel("probability / expected score")
    ax.set_title("Engine vs human: the same eval means different things")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax.set_xlim(-6, 6); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig("results/chart_engine_vs_human.png"); plt.close(fig)


chart_engine_vs_human()


def chart_humanlike():
    import json as _j
    H = _j.load(open("results/humanlike.json"))
    levels = [32, 256, 4096, 65536]
    elo_map = {32: 1468, 256: 1561, 4096: 2439, 65536: 3100}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    ax = axes[0]
    x = [elo_map[l] for l in levels]
    top1 = [H[f"top1_match_{l}"] for l in levels]
    top3 = [H[f"in_top3_{l}"] for l in levels]
    ax.plot(x, top1, color=C["blue"], lw=2, marker="o", ms=7, markeredgecolor=SURFACE,
            label="SF top-1 == human move")
    ax.plot(x, top3, color=C["aqua"], lw=2, marker="s", ms=7, markeredgecolor=SURFACE,
            label="human move in SF top-3")
    for xi, y in zip(x, top1):
        ax.annotate(f"{y:.0%}", (xi, y + 0.02), ha="center", fontsize=8.5, color=INK2)
    ax.axhspan(0.46, 0.52, color=C["orange"], alpha=0.12)
    ax.annotate("Maia / chess-LLM\nreported ~46-52%", (2000, 0.49),
                fontsize=8, color=C["orange"], ha="center")
    ax.set_xlabel("Stockfish strength (Elo, by node budget)")
    ax.set_ylabel("agreement with the human move")
    ax.set_title("How human-like is Stockfish?\nMove-match peaks at ~1560, falls as it gets stronger")
    ax.legend(frameon=False, loc="center right", fontsize=8.5)
    ax.set_ylim(0, 0.8)

    ax = axes[1]
    bands = ["u1400", "1400-1800", "1800-2200", "2200+"]
    blab = ["<1400", "1400-1800", "1800-2200", "2200+"]
    for l, col, mk in [(256, C["blue"], "o"), (65536, C["red"], "s")]:
        ys = [H["by_band"][b][f"top1_{l}"] for b in bands]
        ax.plot(blab, ys, color=col, lw=2, marker=mk, ms=7, markeredgecolor=SURFACE,
                label=f"SF @ {l} nodes ({elo_map[l]} Elo)")
    ax.set_xlabel("human player's rating band")
    ax.set_ylabel("SF top-1 == human move")
    ax.set_title("Stronger humans play more\nStockfish-like moves")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax.set_ylim(0.15, 0.5)
    fig.tight_layout(); fig.savefig("results/chart_humanlike.png"); plt.close(fig)


chart_humanlike()
