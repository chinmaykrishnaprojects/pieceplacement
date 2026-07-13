"""Human-move agreement for the user's chess-LLM (lichess + stockfish variants).

Same protocol as maia_humanmatch.py so the numbers sit on one axis. Uses the
remote API, so N is kept modest and requests are serialized. We reconstruct the
PGN context from the stored game is not available here (we only kept FENs), so
we pass pgn="" and rely on the FEN — the model is primarily FEN-driven in the
/analyze path (the site sends both; empty pgn is accepted, verified live).
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import chess
import pandas as pd

sys.path.insert(0, "scripts")
from chessllm import ChessLLM

IN = "data/fen_moves.csv.gz"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 250
TEMP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 8


def eval_one(args):
    model, fen, human_uci, elo = args
    board = chess.Board(fen)
    human = chess.Move.from_uci(human_uci)
    if human not in board.legal_moves:
        return None
    eng = ChessLLM(model=model, temperature=TEMP)
    pol = eng.policy(board)
    if not pol:
        return None
    ranked = sorted(pol, key=pol.get, reverse=True)
    return (int(ranked[0] == human), int(human in ranked[:3]), float(elo))


def main():
    df = pd.read_csv(IN).sample(N, random_state=1).reset_index(drop=True)
    df["mover_elo"] = pd.to_numeric(df.mover_elo, errors="coerce").fillna(1500)
    summ = {}
    for model in ("lichess", "stockfish"):
        jobs = [(model, r.fen, r.human_move, r.mover_elo) for _, r in df.iterrows()]
        t1 = t3 = n = 0
        bands = {}
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for i, res in enumerate(ex.map(eval_one, jobs)):
                if res is None:
                    continue
                h1, h3, elo = res
                t1 += h1
                t3 += h3
                n += 1
                b = ("u1400" if elo < 1400 else "1400-1800" if elo < 1800
                     else "1800-2200" if elo < 2200 else "2200+")
                d = bands.setdefault(b, [0, 0])
                d[0] += h1
                d[1] += 1
                if n % 50 == 0:
                    print(f"{model}: {n} done, top1 {t1/n:.3f}", flush=True)
        summ[model] = {"n": n, "top1": t1 / n, "top3": t3 / n,
                       "by_band_top1": {k: v[0]/v[1] for k, v in bands.items() if v[1] > 20}}
        print(f"chess-LLM/{model}: top1 {t1/n:.3f}  top3 {t3/n:.3f}  (n={n})", flush=True)
    json.dump(summ, open("results/chessllm_humanmatch.json", "w"), indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
