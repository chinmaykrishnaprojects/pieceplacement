"""Place real Maia3 nets on the node ladder by direct play vs Stockfish rungs.

Maia plays its argmax policy move (no search, ~1 forward pass/move), Elo-self
and Elo-oppo both set to a fixed value (default 1500). Each Maia size plays
GAMES games against SF at chosen node rungs; we convert the score to Elo using
the ladder anchors, giving Maia's real playing strength on the same scale as
everything else on the frontier.
"""
import csv
import random
import sys

import chess
import chess.engine

sys.path.insert(0, "scripts")
from maia3 import MaiaEngine
from node_ladder import make_opening, new_engine, MAX_PLIES

MODELS = {"5M": "models/maia3-5m.fp32.onnx",
          "23M": "models/maia3-23m.fp32.onnx",
          "79M": "models/maia3-79m.fp32.onnx"}
GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 30
RUNGS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["32", "256"])]
OUT = sys.argv[3] if len(sys.argv) > 3 else "results/maia_ladder.csv"
MAIA_ELO = 1500


def play(board, white, black, sf, rng):
    board = board.copy()
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        who = white if board.turn == chess.WHITE else black
        if isinstance(who, MaiaEngine):
            mv = who.play(board, MAIA_ELO, MAIA_ELO, temperature=0.0)
        else:
            mv = sf.play(board, chess.engine.Limit(nodes=who)).move
        if mv is None:
            break
        board.push(mv)
    if board.ply() >= MAX_PLIES and not board.is_game_over(claim_draw=True):
        return 0.5
    return {"1-0": 1.0, "0-1": 0.0}.get(board.result(claim_draw=True), 0.5)


def main():
    rng = random.Random(2024)
    sf, probe = new_engine(), new_engine()
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["maia", "rung", "game", "maia_is_white", "score_maia"])
        for name, path in MODELS.items():
            try:
                eng = MaiaEngine(path)
            except FileNotFoundError:
                continue
            for rung in RUNGS:
                pts = 0.0
                for g in range(GAMES // 2):
                    opening = make_opening(rng, probe)
                    for maia_white in (True, False):
                        s = play(opening, eng if maia_white else rung,
                                 rung if maia_white else eng, sf, rng)
                        sm = s if maia_white else 1.0 - s
                        pts += sm
                        w.writerow([name, rung, g, int(maia_white), sm])
                        fh.flush()
                print(f"Maia3-{name} vs SF@{rung}: {pts}/{GAMES} "
                      f"({pts/GAMES:.3f})", flush=True)
    sf.quit(); probe.quit()
    print("DONE")


if __name__ == "__main__":
    main()
