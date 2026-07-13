"""Measure the user's chess-LLM playing strength vs Stockfish rungs.

Both variants (lichess, stockfish) play argmax policy (no search) against SF at
node rungs; score -> Elo via the ladder anchors. Remote API, so games are short
and few; openings are the shared balanced set. To keep move latency bounded we
pass the running PGN so the char-level model has its native context.
"""
import csv
import random
import sys

import chess
import chess.engine

sys.path.insert(0, "scripts")
from chessllm import ChessLLM
from node_ladder import make_opening, new_engine, MAX_PLIES

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 12
RUNGS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["32"])]
OUT = sys.argv[3] if len(sys.argv) > 3 else "results/chessllm_ladder.csv"
MAX_LLM_PLIES = 120  # cap game length to bound API calls


def board_pgn(moves):
    """Build a minimal SAN movetext from a move stack for model context."""
    b = chess.Board()
    out = []
    for i, m in enumerate(moves):
        if i % 2 == 0:
            out.append(f"{i//2+1}.")
        out.append(b.san(m))
        b.push(m)
    return " ".join(out)


def play(opening, white, black, sf, rng):
    board = opening.copy()
    hist = list(board.move_stack)
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_LLM_PLIES:
        who = white if board.turn == chess.WHITE else black
        if isinstance(who, ChessLLM):
            mv = who.play(board, pgn=board_pgn(hist))
        else:
            mv = sf.play(board, chess.engine.Limit(nodes=who)).move
        if mv is None:
            break
        board.push(mv)
        hist.append(mv)
    if not board.is_game_over(claim_draw=True):
        return 0.5
    return {"1-0": 1.0, "0-1": 0.0}.get(board.result(claim_draw=True), 0.5)


def main():
    rng = random.Random(7)
    sf, probe = new_engine(), new_engine()
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "rung", "game", "llm_is_white", "score_llm"])
        for model in ("lichess", "stockfish"):
            eng = ChessLLM(model=model, temperature=0.1)
            for rung in RUNGS:
                pts = 0.0
                for g in range(GAMES // 2):
                    opening = make_opening(rng, probe)
                    for llm_white in (True, False):
                        s = play(opening, eng if llm_white else rung,
                                 rung if llm_white else eng, sf, rng)
                        sl = s if llm_white else 1.0 - s
                        pts += sl
                        w.writerow([model, rung, g, int(llm_white), sl])
                        fh.flush()
                print(f"chess-LLM/{model} vs SF@{rung}: {pts}/{GAMES} "
                      f"({pts/GAMES:.3f})", flush=True)
    sf.quit(); probe.quit()
    print("DONE")


if __name__ == "__main__":
    main()
