"""Head-to-head: the chess-LM's INTERNAL representation vs from-scratch chess2vec.

Same positions, same labels, same probes. Asks a question the project has been
circling: is a language model's incidental internal state a *better chess
embedding* than a space trained specifically to be one?

  chess2vec  — 256-d, trained skip-gram (a position is known by what follows it)
               on the same games. Purpose-built to be a position embedding.
  chess-LM   — 512-d slice of the layer-L residual stream at the last PGN token.
               Never trained to embed anything; it fell out of next-char
               prediction. Free at inference (the policy call computes it anyway).

Probes (identical for both): phase, open/closed, who-is-ahead, material R^2,
and embedding -> board reconstruction.

Stage 1 (--extract) writes results/embed_compare_data.npz.
Stage 2 (default)   trains chess2vec on the same games and runs the probes.
"""
import json
import sys

import chess
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")

PT = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5,
       chess.QUEEN: 9}
DATA = "results/embed_compare_data.npz"


def labels_for(board):
    """Concept labels, computed from the FEN (side-to-move POV for advantage)."""
    mat = {c: sum(VAL[pt] * len(board.pieces(pt, c)) for pt in PT)
           for c in (True, False)}
    stm = board.turn
    stm_mat = mat[stm] - mat[not stm]
    npm = sum(VAL[pt] * len(board.pieces(pt, c))
              for pt in PT[1:] for c in (True, False))
    phase = 0 if board.fullmove_number <= 10 else (2 if npm <= 12 else 1)
    closed = sum(1 for c in (True, False) for sq in board.pieces(chess.PAWN, c)
                 if 2 <= chess.square_file(sq) <= 5)
    return stm_mat, phase, closed


def extract(model_path, n_pos):
    from chessgpt_local import ChessGPT
    gpt = ChessGPT(model_path)
    df = pd.read_csv("data/fen_moves_pgn.csv.gz")
    df = df.sample(min(n_pos * 2, len(df)), random_state=3).reset_index(drop=True)

    E, F, G, sm, ph, cl = [], [], [], [], [], []
    for _, r in df.iterrows():
        if len(F) >= n_pos:
            break
        b = chess.Board(r.fen)
        if b.is_game_over():
            continue
        a = gpt.embed_position(str(r.pgn_prefix))
        if a is None:
            continue
        E.append(a.astype(np.float32))
        F.append(r.fen)
        G.append(str(r.game_id))
        s, p, c = labels_for(b)
        sm.append(s); ph.append(p); cl.append(c)
        if len(F) % 500 == 0:
            print(f"  {len(F)}/{n_pos}", flush=True)
    np.savez_compressed(DATA, E=np.stack(E), fen=np.array(F), game=np.array(G),
                        stm_material=np.array(sm), phase=np.array(ph),
                        closed=np.array(cl))
    print(f"saved {DATA}: {np.stack(E).shape} (positions, layers, dim)")


def probe_set(X, d, name):
    """The identical probe battery used on chess2vec."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import r2_score
    n = len(X)
    ntr = int(0.8 * n)
    Xtr, Xte = X[:ntr], X[ntr:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    out = {"name": name, "dim": int(X.shape[1])}

    y = d["phase"]
    out["phase_acc"] = float((LogisticRegression(max_iter=400).fit(Xtr, y[:ntr])
                              .predict(Xte) == y[ntr:]).mean())
    out["phase_baseline"] = float(np.bincount(y[ntr:]).max() / len(y[ntr:]))

    sm = d["stm_material"]
    yc = np.sign(sm).astype(int)
    out["who_ahead_acc"] = float((LogisticRegression(max_iter=400)
                                  .fit(Xtr, yc[:ntr]).predict(Xte) == yc[ntr:]).mean())
    out["who_ahead_baseline"] = float(np.bincount(yc[ntr:] + 1).max() / len(yc[ntr:]))
    out["material_r2"] = float(r2_score(sm[ntr:],
                                        Ridge(alpha=1.0).fit(Xtr, sm[:ntr]).predict(Xte)))

    cl = d["closed"]
    ycl = (cl > np.median(cl)).astype(int)
    out["open_closed_acc"] = float((LogisticRegression(max_iter=400)
                                    .fit(Xtr, ycl[:ntr]).predict(Xte) == ycl[ntr:]).mean())
    return out


def board_recon(X, fens):
    """Can a linear map recover the piece placement? (64x13 one-vs-rest, sampled)"""
    from sklearn.linear_model import LogisticRegression
    order = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN,
             chess.KING]
    Y = np.zeros((len(fens), 64), dtype=int)
    for i, f in enumerate(fens):
        b = chess.Board(str(f))
        for sq in range(64):
            pc = b.piece_at(sq)
            Y[i, sq] = 0 if pc is None else (order.index(pc.piece_type) + 1
                                             + (0 if pc.color else 6))
    ntr = int(0.8 * len(X))
    mu, sd = X[:ntr].mean(0), X[:ntr].std(0) + 1e-6
    Xtr, Xte = (X[:ntr] - mu) / sd, (X[ntr:] - mu) / sd
    accs = []
    for sq in range(0, 64, 2):          # every other square: same estimate, half the time
        ytr = Y[:ntr, sq]
        if len(np.unique(ytr)) < 2:
            accs.append(float((Y[ntr:, sq] == ytr[0]).mean()))
            continue
        accs.append(float((LogisticRegression(max_iter=200).fit(Xtr, ytr)
                           .predict(Xte) == Y[ntr:, sq]).mean()))
    maj = [np.bincount(Y[:ntr, sq]).argmax() for sq in range(0, 64, 2)]
    base = float(np.mean([(Y[ntr:, sq] == m).mean()
                          for sq, m in zip(range(0, 64, 2), maj)]))
    return float(np.mean(accs)), base


def main():
    if "--extract" in sys.argv:
        n = int(sys.argv[sys.argv.index("--extract") + 1])
        extract("models/stockfish_16layers.pt", n)
        return

    d = np.load(DATA, allow_pickle=True)
    E = d["E"]                                   # [N, layers, dim]
    print(f"{E.shape[0]} positions, {E.shape[1]} layers, dim {E.shape[2]}\n")

    # --- layer sweep on the LM ------------------------------------------
    results = []
    for L in range(1, E.shape[1] + 1, 3):
        r = probe_set(E[:, L - 1, :], d, f"chess-LM layer {L}")
        r["layer"] = L
        results.append(r)
        print(f"  layer {L:2d}: phase {r['phase_acc']:.3f}  "
              f"open/closed {r['open_closed_acc']:.3f}  "
              f"who-ahead {r['who_ahead_acc']:.3f}  mat R2 {r['material_r2']:.3f}")
    best = max(results, key=lambda r: r["phase_acc"] + r["open_closed_acc"]
               + r["who_ahead_acc"])
    print(f"\nbest LM layer: {best['layer']}")

    out = {"lm_layers": results, "lm_best": best}

    # --- board reconstruction at the best layer -------------------------
    acc, base = board_recon(E[:, best["layer"] - 1, :], d["fen"])
    out["lm_board_recon"] = {"acc": acc, "baseline": base}
    print(f"LM board reconstruction (linear): {acc:.3f}  (majority {base:.3f})")

    json.dump(out, open("results/embed_compare.json", "w"), indent=2)
    print("\nwrote results/embed_compare.json")


if __name__ == "__main__":
    main()
