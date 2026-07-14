"""Gemini human-move agreement on the same lichess positions as the other models.

One API call per position (best move). Checkpoints after every position so a
quota cutoff still leaves usable data. Keep N modest for the free tier.
"""
import json
import os
import sys

import chess
import pandas as pd

sys.path.insert(0, "scripts")
from gemini_chess import GeminiChess

IN = "data/fen_moves_pgn.csv.gz"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
OUT = "results/gemini_humanmatch.json"


def to_pgn_movetext(karvonen_prefix):
    """Turn ';1.e4 e5 2.Nf3 ' into '1. e4 e5 2. Nf3' for the prompt."""
    s = karvonen_prefix.lstrip(";").strip()
    return s


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("no key; run with: set -a; source .secrets/gemini.env; set +a")
        sys.exit(1)
    df = pd.read_csv(IN)
    df["mover_elo"] = pd.to_numeric(df.mover_elo, errors="coerce").fillna(1500)
    df = df.sample(min(N, len(df)), random_state=1).reset_index(drop=True)
    eng = GeminiChess()

    t1 = n = 0
    bands = {}
    records = []
    for i, r in df.iterrows():
        board = chess.Board(r.fen)
        human = chess.Move.from_uci(r.human_move)
        if human not in board.legal_moves:
            continue
        pgn = to_pgn_movetext(str(r.pgn_prefix))
        mv, raw = eng.best_move(board, pgn)
        if mv is None:
            continue
        hit = int(chess.Move.from_uci(mv) == human)
        t1 += hit
        n += 1
        elo = r.mover_elo
        b = ("u1400" if elo < 1400 else "1400-1800" if elo < 1800
             else "1800-2200" if elo < 2200 else "2200+")
        d = bands.setdefault(b, [0, 0])
        d[0] += hit
        d[1] += 1
        records.append({"fen": r.fen, "human": r.human_move, "gemini": mv, "hit": hit})
        # checkpoint every position
        json.dump({"model": eng.model, "n": n, "top1": t1 / n,
                   "by_band_top1": {k: v[0]/v[1] for k, v in bands.items() if v[1] > 5},
                   "api_calls": eng.calls, "records": records},
                  open(OUT, "w"), indent=2)
        if n % 10 == 0:
            print(f"{n}/{len(df)} top1 {t1/n:.3f} (calls {eng.calls})", flush=True)
    print(f"Gemini/{eng.model}: top1 {t1/n:.3f}  (n={n}, calls={eng.calls})")
    print("DONE")


if __name__ == "__main__":
    main()
