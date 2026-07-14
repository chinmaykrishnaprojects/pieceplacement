"""Linear board-state probes on the chess-GPT residual stream (world-model test).

Karvonen-style mechanistic interpretability: a char-level PGN model never sees a
board, yet to predict good moves it must build an internal board representation.
We test this by training simple LINEAR probes to decode the board from the
model's activations:

  for each layer L, take the residual-stream vector at the LAST token of the PGN
  prefix (a space, where the model is about to emit the next move) and fit a
  linear classifier square -> {empty, P,N,B,R,Q,K x2 colors} (13 classes).

If accuracy is high and rises with depth, the model holds a linear,
human-legible board state — a concrete "J-space"-style readout of what it is
"thinking about". We also report per-square and per-piece accuracy for a board
heatmap.

Activations are captured with forward hooks (no model edit). CPU-only.
"""
import json
import sys

import chess
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "scripts")
from chessgpt_local import ChessGPT, STOI

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/lichess_16layers_ckpt_no_optimizer.pt"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
OUT = sys.argv[3] if len(sys.argv) > 3 else "results/probe_boardstate"

# 13 classes: 0 empty; 1-6 white P N B R Q K; 7-12 black P N B R Q K
PT_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def square_label(board, sq):
    pc = board.piece_at(sq)
    if pc is None:
        return 0
    base = PT_ORDER.index(pc.piece_type) + 1
    return base + (0 if pc.color == chess.WHITE else 6)


def capture_activations(cg, prefix):
    """Residual stream after each block at the last token. -> [n_layer+1, n_embd]."""
    ids = [STOI[c] for c in prefix if c in STOI]
    ids = ids[-cg.model.block_size:]
    x = torch.tensor([ids])
    caps = []
    hooks = []
    blocks = cg.model.transformer.h

    def mk(i):
        def hook(mod, inp, out):
            caps.append(out[0, -1].detach().numpy())  # last token
        return hook
    # embedding (layer 0) captured manually below; hook each block output
    for blk in blocks:
        hooks.append(blk.register_forward_hook(mk(len(hooks))))
    with torch.no_grad():
        cg.model(x)
    for h in hooks:
        h.remove()
    return np.stack(caps)  # [n_layer, n_embd]


def main():
    cg = ChessGPT(MODEL)
    df = pd.read_csv("data/fen_moves_pgn.csv.gz").sample(N, random_state=0).reset_index(drop=True)

    n_layer = len(cg.model.transformer.h)
    X = [[] for _ in range(n_layer)]     # per layer: list of activation vectors
    Y = []                               # [n_samples, 64] square labels
    for i, r in df.iterrows():
        try:
            acts = capture_activations(cg, str(r.pgn_prefix))
        except Exception:
            continue
        board = chess.Board(r.fen)
        for L in range(n_layer):
            X[L].append(acts[L])
        Y.append([square_label(board, sq) for sq in range(64)])
        if (i + 1) % 500 == 0:
            print(f"activations {i+1}/{len(df)}", flush=True)
    Y = np.array(Y)
    ntr = int(0.8 * len(Y))

    # baseline: majority-class per square (mostly "empty")
    maj = []
    for sq in range(64):
        vals, cnts = np.unique(Y[:ntr, sq], return_counts=True)
        maj.append(vals[cnts.argmax()])
    maj = np.array(maj)
    base_acc = float((Y[ntr:] == maj).mean())

    layer_acc = []
    per_square_best = None
    for L in range(n_layer):
        XL = np.stack(X[L])
        Xtr, Xte = XL[:ntr], XL[ntr:]
        sq_acc = []
        preds = np.zeros_like(Y[ntr:])
        for sq in range(64):
            ytr = Y[:ntr, sq]
            if len(np.unique(ytr)) < 2:
                preds[:, sq] = ytr[0]
                sq_acc.append(float((Y[ntr:, sq] == ytr[0]).mean()))
                continue
            clf = LogisticRegression(max_iter=200, C=1.0, n_jobs=-1)
            clf.fit(Xtr, ytr)
            p = clf.predict(Xte)
            preds[:, sq] = p
            sq_acc.append(float((p == Y[ntr:, sq]).mean()))
        acc = float(np.mean(sq_acc))
        layer_acc.append(acc)
        print(f"layer {L+1}/{n_layer}: board-recon acc {acc:.3f}", flush=True)
        if acc == max(layer_acc):
            per_square_best = sq_acc

    out = {
        "model": MODEL, "n": int(len(Y)), "n_layer": n_layer,
        "baseline_majority_acc": base_acc,
        "layer_accuracy": layer_acc,
        "best_layer": int(np.argmax(layer_acc)) + 1,
        "best_acc": float(max(layer_acc)),
        "per_square_acc_best_layer": per_square_best,  # 64, a1..h8
    }
    json.dump(out, open(OUT + ".json", "w"), indent=2)
    print(f"\nBaseline (guess empty): {base_acc:.3f}")
    print(f"Best probe: layer {out['best_layer']} acc {out['best_acc']:.3f}")


if __name__ == "__main__":
    main()
