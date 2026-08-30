"""Dataset for the blunder probe: does the LM's own activation know a move loses?

For real lichess positions (with their true PGN context, so the char-LM is in
distribution), we take the model's top-K candidate moves, pull the per-move
residual-stream activation that the SAME forward pass already computed, and
label each move with Stockfish: how much eval does it throw away?

Output: results/probe_data.npz  (X = activations, y = blunder label, plus cp loss
and policy prob for analysis).

The point: labels come from Stockfish OFFLINE, at training time only. At
inference the probe is a dot product — no engine, no search, no material table.
"""
import gc
import sys
import time

import chess
import chess.engine
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from chessgpt_local import ChessGPT

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/stockfish_16layers.pt"
N_POS = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
OUT = sys.argv[3] if len(sys.argv) > 3 else "results/probe_data.npz"
LAYER = int(sys.argv[4]) if len(sys.argv) > 4 else 11
TOPK = 6
NODES = 15000
BLUNDER_CP = 150


def main():
    gpt = ChessGPT(MODEL)
    eng = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
    eng.configure({"Threads": 2, "Hash": 64})

    df = pd.read_csv("data/fen_moves_pgn.csv.gz").sample(
        N_POS * 3, random_state=11).reset_index(drop=True)

    X, y, cps, probs, plies = [], [], [], [], []
    used = 0
    t0 = time.time()
    for _, r in df.iterrows():
        if used >= N_POS:
            break
        board = chess.Board(r.fen)
        if board.is_game_over():
            continue
        pgn = str(r.pgn_prefix)
        try:
            pol, acts = gpt.policy_with_acts(board, pgn_prefix=pgn, layer=LAYER)
        except Exception:  # noqa: BLE001
            continue
        if not pol:
            continue
        ranked = sorted(pol, key=pol.get, reverse=True)[:TOPK]
        # best available eval from the mover's POV
        try:
            info = eng.analyse(board, chess.engine.Limit(nodes=NODES))
            best = info["score"].pov(board.turn).score(mate_score=3000)
        except Exception:  # noqa: BLE001
            continue
        if best is None:
            continue
        for mv in ranked:
            board.push(mv)
            try:
                i2 = eng.analyse(board, chess.engine.Limit(nodes=NODES))
                after = -i2["score"].pov(board.turn).score(mate_score=3000)
            except Exception:  # noqa: BLE001
                board.pop()
                continue
            board.pop()
            if after is None:
                continue
            loss = max(0, best - after)
            X.append(acts[mv])
            y.append(1 if loss >= BLUNDER_CP else 0)
            cps.append(loss)
            probs.append(pol[mv])
            plies.append(int(r.ply))
        used += 1
        if used % 100 == 0:
            rate = np.mean(y) if y else 0
            print(f"{used}/{N_POS} positions, {len(y)} moves, "
                  f"blunder rate {rate:.3f}, {time.time()-t0:.0f}s", flush=True)
            # checkpoint: an OOM/kill must never cost the whole run
            np.savez_compressed(OUT, X=np.stack(X).astype(np.float32),
                                y=np.array(y), cp=np.array(cps),
                                prob=np.array(probs), ply=np.array(plies),
                                layer=LAYER)
            gc.collect()

    eng.quit()
    X = np.stack(X).astype(np.float32)
    np.savez_compressed(OUT, X=X, y=np.array(y), cp=np.array(cps),
                        prob=np.array(probs), ply=np.array(plies), layer=LAYER)
    print(f"\nsaved {OUT}: {X.shape}, blunder rate {np.mean(y):.3f}")


if __name__ == "__main__":
    main()
