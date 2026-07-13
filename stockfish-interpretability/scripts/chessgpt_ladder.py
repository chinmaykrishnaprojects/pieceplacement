"""Local playing-strength ladder for the chess-GPT checkpoints vs Stockfish.

Runs the actual 16-layer weights on-server (no API). Each variant plays argmax
policy (no search) vs SF at node rungs; score -> Elo via ladder anchors. The
model gets its native Karvonen PGN context, rebuilt each move from the stack.
"""
import csv
import random
import sys

import chess
import chess.engine

sys.path.insert(0, "scripts")
from chessgpt_local import ChessGPT, board_to_pgn_prefix
from node_ladder import make_opening, new_engine, MAX_PLIES

MODELS = {"lichess": "models/lichess_16layers_ckpt_no_optimizer.pt",
          "stockfish": "models/stockfish_16layers_ckpt_no_optimizer.pt"}
GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 20
RUNGS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["32", "256"])]
OUT = sys.argv[3] if len(sys.argv) > 3 else "results/chessgpt_ladder.csv"


def play(opening, white, black, sf, rng):
    board = opening.copy()
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        who = white if board.turn == chess.WHITE else black
        if isinstance(who, ChessGPT):
            mv = who.play(board)  # rebuilds PGN prefix from move stack
        else:
            mv = sf.play(board, chess.engine.Limit(nodes=who)).move
        if mv is None:
            break
        board.push(mv)
    if not board.is_game_over(claim_draw=True):
        return 0.5
    return {"1-0": 1.0, "0-1": 0.0}.get(board.result(claim_draw=True), 0.5)


def main():
    rng = random.Random(11)
    sf, probe = new_engine(), new_engine()
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "rung", "game", "llm_is_white", "score_llm"])
        for name, path in MODELS.items():
            eng = ChessGPT(path)
            for rung in RUNGS:
                pts = 0.0
                for g in range(GAMES // 2):
                    opening = make_opening(rng, probe)
                    for llm_white in (True, False):
                        s = play(opening, eng if llm_white else rung,
                                 rung if llm_white else eng, sf, rng)
                        sl = s if llm_white else 1.0 - s
                        pts += sl
                        w.writerow([name, rung, g, int(llm_white), sl])
                        fh.flush()
                print(f"chess-GPT/{name} vs SF@{rung}: {pts}/{GAMES} "
                      f"({pts/GAMES:.3f})", flush=True)
    sf.quit(); probe.quit()
    print("DONE")


if __name__ == "__main__":
    main()
