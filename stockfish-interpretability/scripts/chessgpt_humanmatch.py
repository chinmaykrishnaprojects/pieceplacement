"""Local human-move agreement for the chess-GPT checkpoints (lichess + stockfish).

Runs the actual 16-layer nanoGPT weights on this server (no API). Same protocol
and same lichess positions as maia_humanmatch.py, using the stored Karvonen-
format PGN prefix so the char-level model gets its native game context.
"""
import json
import sys

import chess
import pandas as pd

sys.path.insert(0, "scripts")
from chessgpt_local import ChessGPT

MODELS = {
    "lichess": "models/lichess_16layers_ckpt_no_optimizer.pt",
    "stockfish": "models/stockfish_16layers_ckpt_no_optimizer.pt",
}
IN = "data/fen_moves_pgn.csv.gz"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000


def main():
    df = pd.read_csv(IN)
    df["mover_elo"] = pd.to_numeric(df.mover_elo, errors="coerce").fillna(1500)
    df = df.sample(min(N, len(df)), random_state=1).reset_index(drop=True)

    summ = {}
    for name, path in MODELS.items():
        eng = ChessGPT(path)
        t1 = t3 = n = 0
        bands = {}
        for _, r in df.iterrows():
            board = chess.Board(r.fen)
            human = chess.Move.from_uci(r.human_move)
            if human not in board.legal_moves:
                continue
            pol = eng.policy(board, pgn_prefix=str(r.pgn_prefix))
            if not pol:
                continue
            ranked = sorted(pol, key=pol.get, reverse=True)
            h1 = int(ranked[0] == human)
            h3 = int(human in ranked[:3])
            t1 += h1
            t3 += h3
            n += 1
            elo = r.mover_elo
            b = ("u1400" if elo < 1400 else "1400-1800" if elo < 1800
                 else "1800-2200" if elo < 2200 else "2200+")
            d = bands.setdefault(b, [0, 0])
            d[0] += h1
            d[1] += 1
            if n % 200 == 0:
                print(f"{name}: {n} top1 {t1/n:.3f}", flush=True)
        summ[name] = {"n": n, "top1": t1 / n, "top3": t3 / n,
                      "by_band_top1": {k: v[0]/v[1] for k, v in bands.items() if v[1] > 30}}
        print(f"chess-GPT/{name}: top1 {t1/n:.3f}  top3 {t3/n:.3f}  (n={n})", flush=True)
        json.dump(summ, open("results/chessgpt_humanmatch.json", "w"), indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
