"""Estimate Elo vs nodes-per-move for Stockfish via adjacent-level self-play.

Ladder levels are node limits; each level plays a match vs the next level.
Openings: for each game pair, play R random plies (uniform over legal moves,
filtered to keep |eval| < 150cp via a quick probe) then both engines take over;
colors are swapped for the second game of the pair.

Also includes MATERIAL, a pure material-count alpha-beta (depth 2 + capture
quiescence) python engine — the maximally-interpretable classical baseline.

Output: results/ladder_results.csv with one row per game.
"""
import csv
import random
import sys
import time

import chess
import chess.engine

SF = "/usr/games/stockfish"
# Effective floor is ~32 nodes (SF16 always completes a depth-2 iteration),
# so the lowest honest rung is 32.
LEVELS = [32, 64, 256, 1024, 4096, 16384, 65536]
GAMES_PER_PAIR = int(sys.argv[1]) if len(sys.argv) > 1 else 40
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/ladder_results.csv"
MAX_PLIES = 220
OPENING_PLIES = 6

MAT = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 300,
       chess.ROOK: 500, chess.QUEEN: 900}


def material_score(board: chess.Board) -> int:
    s = 0
    for pt, v in MAT.items():
        s += v * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
    return s if board.turn == chess.WHITE else -s


def qsearch(board, alpha, beta, depth=4):
    stand = material_score(board)
    if stand >= beta:
        return beta
    alpha = max(alpha, stand)
    if depth == 0:
        return alpha
    for mv in board.legal_moves:
        if not board.is_capture(mv):
            continue
        board.push(mv)
        score = -qsearch(board, -beta, -alpha, depth - 1)
        board.pop()
        if score >= beta:
            return beta
        alpha = max(alpha, score)
    return alpha


def material_engine_move(board: chess.Board, rng: random.Random) -> chess.Move:
    def ab(depth, alpha, beta):
        if board.is_checkmate():
            return -100000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        if depth == 0:
            return qsearch(board, alpha, beta)
        best = -1000000
        for mv in board.legal_moves:
            board.push(mv)
            score = -ab(depth - 1, -beta, -alpha)
            board.pop()
            if score > best:
                best = score
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best

    best_moves, best = [], -10**9
    for mv in board.legal_moves:
        board.push(mv)
        score = -ab(1, -10**9, 10**9)
        board.pop()
        if score > best:
            best, best_moves = score, [mv]
        elif score == best:
            best_moves.append(mv)
    return rng.choice(best_moves)


def make_opening(rng: random.Random, probe: chess.engine.SimpleEngine):
    """Random but roughly balanced opening."""
    for _ in range(60):
        board = chess.Board()
        ok = True
        for _ in range(OPENING_PLIES):
            moves = list(board.legal_moves)
            if not moves:
                ok = False
                break
            board.push(rng.choice(moves))
        if not ok:
            continue
        info = probe.analyse(board, chess.engine.Limit(nodes=5000))
        cp = info["score"].white().score(mate_score=10000)
        if cp is not None and abs(cp) < 150:
            return board
    return chess.Board()


def play_game(board, eng_w, eng_b, nodes_w, nodes_b, rng):
    board = board.copy()
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        if board.turn == chess.WHITE:
            eng, nodes = eng_w, nodes_w
        else:
            eng, nodes = eng_b, nodes_b
        if eng == "MATERIAL":
            mv = material_engine_move(board, rng)
        else:
            r = eng.play(board, chess.engine.Limit(nodes=nodes))
            mv = r.move
        board.push(mv)
    if board.ply() >= MAX_PLIES and not board.is_game_over(claim_draw=True):
        return 0.5
    res = board.result(claim_draw=True)
    return {"1-0": 1.0, "0-1": 0.0}.get(res, 0.5)


def new_engine():
    e = chess.engine.SimpleEngine.popen_uci(SF)
    e.configure({"Threads": 1, "Hash": 16})
    return e


def main():
    rng = random.Random(42)
    eng_a, eng_b, probe = new_engine(), new_engine(), new_engine()
    pairs = [(LEVELS[i], LEVELS[i + 1]) for i in range(len(LEVELS) - 1)]
    pairs.append(("MATERIAL", LEVELS[0]))     # material engine vs 32-node SF
    pairs.append(("MATERIAL", LEVELS[1]))     # and vs 64-node SF for a second edge

    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["low", "high", "game", "low_is_white", "score_low"])
        for low, high in pairs:
            t0 = time.time()
            pts = 0.0
            for g in range(GAMES_PER_PAIR // 2):
                opening = make_opening(rng, probe)
                for low_white in (True, False):
                    if low == "MATERIAL":
                        ew = "MATERIAL" if low_white else eng_b
                        eb = eng_b if low_white else "MATERIAL"
                        nw = 0 if low_white else high
                        nb = high if low_white else 0
                    else:
                        ew, eb = eng_a, eng_b
                        nw = low if low_white else high
                        nb = high if low_white else low
                    s = play_game(opening, ew, eb, nw, nb, rng)
                    score_low = s if low_white else 1.0 - s
                    pts += score_low
                    w.writerow([low, high, g, int(low_white), score_low])
                    fh.flush()
            n = GAMES_PER_PAIR
            print(f"{low} vs {high}: {pts}/{n} ({pts/n:.3f}) in {time.time()-t0:.0f}s",
                  flush=True)
    for e in (eng_a, eng_b, probe):
        e.quit()
    print("DONE")


if __name__ == "__main__":
    main()
