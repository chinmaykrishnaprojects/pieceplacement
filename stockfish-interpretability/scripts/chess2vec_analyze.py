"""Analyze the trained chess2vec space: NN search, concept probes, emb->FEN, viz.

Answers the user's asks directly:
  1. most-similar positions across lichess (cosine NN in the embedding)
  2. is the space rich enough to PROBE for concepts? linear probes for
     phase (opening/middle/end), who's-ahead (sign of side-to-move material),
     open/closed, and a material regression
  3. embedding->FEN: decode embeddings back to boards, report accuracy + examples
  4. 2D PCA scatter colored by phase and who's-ahead
"""
import json

import chess
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score

import sys
sys.path.insert(0, "scripts")
from chess2vec import Encoder, Decoder, board_vec, vec_to_board

EMB = "results/chess2vec_emb.npz"
CKPT = "models/chess2vec.pt"


def load():
    d = np.load(EMB, allow_pickle=True)
    ck = torch.load(CKPT, map_location="cpu")
    enc = Encoder(ck["d"]); enc.load_state_dict(ck["enc"]); enc.eval()
    dec = Decoder(ck["d"]); dec.load_state_dict(ck["dec"]); dec.eval()
    return d, enc, dec


def nn_examples(d, enc, out):
    emb = d["emb"]; fens = d["fen"]
    queries = {
        "italian": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "kingside_attack": "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
        "rook_endgame": "8/5pk1/6p1/7p/7P/6P1/5PK1/R7 w - - 0 1",
    }
    res = {}
    for name, fen in queries.items():
        q = enc(torch.tensor(board_vec(chess.Board(fen)))[None]).detach().numpy()[0]
        sims = emb @ q
        top = np.argsort(-sims)[:5]
        res[name] = [{"fen": str(fens[i]), "cos": float(sims[i])} for i in top]
    out["nearest_neighbors"] = res


def probes(d, out):
    emb = d["emb"]
    ntr = int(0.8 * len(emb))
    Xtr, Xte = emb[:ntr], emb[ntr:]
    r = {}
    # phase (3-class)
    y = d["phase"]
    clf = LogisticRegression(max_iter=300).fit(Xtr, y[:ntr])
    r["phase_acc"] = float((clf.predict(Xte) == y[ntr:]).mean())
    r["phase_baseline"] = float(max(np.bincount(y[ntr:]) / len(y[ntr:])))
    # who is ahead (sign of stm material): 3 classes ahead/equal/behind
    sm = d["stm_material"]
    ycls = np.sign(sm).astype(int)
    clf = LogisticRegression(max_iter=300).fit(Xtr, ycls[:ntr])
    r["who_ahead_acc"] = float((clf.predict(Xte) == ycls[ntr:]).mean())
    r["who_ahead_baseline"] = float(max(np.bincount((ycls[ntr:] + 1)) / len(ycls[ntr:])))
    # material amount (regression, pawns)
    reg = Ridge(alpha=1.0).fit(Xtr, sm[:ntr])
    r["material_r2"] = float(r2_score(sm[ntr:], reg.predict(Xte)))
    # open/closed (median split on closed count)
    cl = d["closed"]
    thr = np.median(cl)
    yc = (cl > thr).astype(int)
    clf = LogisticRegression(max_iter=300).fit(Xtr, yc[:ntr])
    r["open_closed_acc"] = float((clf.predict(Xte) == yc[ntr:]).mean())
    out["concept_probes"] = r


def recon(d, enc, dec, out):
    emb = d["emb"][:3000]
    A = np.stack([board_vec(chess.Board(f)) for f in d["fen"][:3000]])
    with torch.no_grad():
        rec = torch.sigmoid(dec(torch.tensor(emb))).numpy()
    per_sq = float(((rec > 0.5) == (A > 0.5)).mean())
    # exact-board reconstruction rate
    exact = float((((rec > 0.5) == (A > 0.5)).reshape(len(A), -1).all(1)).mean())
    out["embedding_to_fen"] = {"per_square_acc": per_sq, "exact_board_rate": exact}


def viz(d, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    emb = d["emb"]
    C = {"blue": "#2a78d6", "aqua": "#1baf7a", "orange": "#eb6834", "red": "#e34948",
         "gray": "#9a9992"}
    SURF = "#fcfcfb"
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF,
                         "axes.edgecolor": "#d8d7d2", "font.size": 11})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))

    # LEFT: what the frozen 256-d embedding LINEARLY encodes (probe vs baseline)
    p = out["concept_probes"]
    names = ["phase\n(open/mid/end)", "open vs\nclosed", "who is\nahead"]
    accs = [p["phase_acc"], p["open_closed_acc"], p["who_ahead_acc"]]
    bases = [p["phase_baseline"], 0.5, p["who_ahead_baseline"]]
    x = np.arange(3)
    ax[0].bar(x - 0.2, bases, 0.38, color=C["gray"], label="majority baseline")
    ax[0].bar(x + 0.2, accs, 0.38, color=C["blue"], label="linear probe on embedding")
    for xi, a in zip(x, accs):
        ax[0].text(xi + 0.2, a + 0.02, f"{a:.0%}", ha="center", fontsize=9)
    ax[0].set_xticks(x); ax[0].set_xticklabels(names)
    ax[0].set_ylim(0, 1.05); ax[0].set_ylabel("accuracy")
    ax[0].set_title("chess2vec linearly encodes chess concepts\n(256-d, trained only on 'what follows in games')")
    ax[0].legend(frameon=False, fontsize=9, loc="upper right")
    ax[0].spines[["top", "right"]].set_visible(False)

    # RIGHT: supervised 2-D (LDA on phase) so the phase structure is visible
    idx = np.random.RandomState(0).choice(len(emb), min(4000, len(emb)), replace=False)
    ph = d["phase"][idx]
    proj = LDA(n_components=2).fit(emb[idx], ph).transform(emb[idx])
    for v, c, lab in [(0, C["aqua"], "opening"), (1, C["blue"], "middlegame"),
                      (2, C["orange"], "endgame")]:
        m = ph == v
        ax[1].scatter(proj[m, 0], proj[m, 1], s=5, c=c, label=lab, alpha=0.55)
    ax[1].set_title("phase structure in the embedding\n(supervised 2-D projection)")
    ax[1].legend(frameon=False, markerscale=3); ax[1].set_xticks([]); ax[1].set_yticks([])
    ax[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig("results/chart_chess2vec.png", dpi=140); plt.close(fig)


def main():
    d, enc, dec = load()
    out = {"n": int(len(d["emb"])), "dim": int(d["emb"].shape[1])}
    nn_examples(d, enc, out)
    probes(d, out)
    recon(d, enc, dec, out)
    viz(d, out)
    json.dump(out, open("results/chess2vec_analysis.json", "w"), indent=2)
    p = out["concept_probes"]
    print("=== chess2vec concept probes (linear, on frozen embedding) ===")
    print(f"phase       : {p['phase_acc']:.3f}  (baseline {p['phase_baseline']:.3f})")
    print(f"who-ahead   : {p['who_ahead_acc']:.3f}  (baseline {p['who_ahead_baseline']:.3f})")
    print(f"material R^2: {p['material_r2']:.3f}")
    print(f"open/closed : {p['open_closed_acc']:.3f}")
    print(f"emb->FEN per-square {out['embedding_to_fen']['per_square_acc']:.3f}, "
          f"exact board {out['embedding_to_fen']['exact_board_rate']:.3f}")


if __name__ == "__main__":
    main()
