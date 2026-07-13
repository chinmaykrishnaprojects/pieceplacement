"""Validate real Maia3 nets by human-move agreement, and compare to Stockfish.

For each Maia size, over the FEN+human-move sample (Elo-conditioned on the
actual mover rating), measure top-1 / top-3 agreement with the human move.
This both (a) validates the ONNX runner (should reproduce Maia's published
~50% top-1) and (b) gives the real number for the human-likeness chart, next
to the Stockfish curve measured earlier.
"""
import json
import sys

import chess
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from maia3 import MaiaEngine

MODELS = {
    "5M": "models/maia3-5m.fp32.onnx",
    "23M": "models/maia3-23m.fp32.onnx",
    "79M": "models/maia3-79m.fp32.onnx",
}
IN = "data/fen_moves.csv.gz"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000


def main():
    df = pd.read_csv(IN)
    df["mover_elo"] = pd.to_numeric(df.mover_elo, errors="coerce").fillna(1500).astype(int)
    df = df.sample(min(N, len(df)), random_state=1).reset_index(drop=True)

    summ = {}
    for name, path in MODELS.items():
        try:
            eng = MaiaEngine(path)
        except FileNotFoundError:
            print(f"{name}: model not downloaded, skipping", flush=True)
            continue
        t1 = t3 = n = 0
        bands = {}
        for _, r in df.iterrows():
            board = chess.Board(r.fen)
            human = chess.Move.from_uci(r.human_move)
            if human not in board.legal_moves:
                continue
            elo = int(np.clip(r.mover_elo, 1100, 2000))
            pol = eng.policy(board, elo, elo)
            if not pol:
                continue
            ranked = sorted(pol, key=pol.get, reverse=True)
            hit1 = int(ranked[0] == human)
            hit3 = int(human in ranked[:3])
            t1 += hit1
            t3 += hit3
            n += 1
            b = ("u1400" if r.mover_elo < 1400 else "1400-1800"
                 if r.mover_elo < 1800 else "1800-2200" if r.mover_elo < 2200
                 else "2200+")
            d = bands.setdefault(b, [0, 0])
            d[0] += hit1
            d[1] += 1
        summ[name] = {
            "n": n,
            "top1": t1 / n,
            "top3": t3 / n,
            "by_band_top1": {k: v[0] / v[1] for k, v in bands.items() if v[1] > 30},
        }
        print(f"Maia3-{name}: top1 {t1/n:.3f}  top3 {t3/n:.3f}  (n={n})", flush=True)
    json.dump(summ, open("results/maia_humanmatch.json", "w"), indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
