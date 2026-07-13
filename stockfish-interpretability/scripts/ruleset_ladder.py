"""Two interpretable/no-search players, placed on the existing node ladder.

RULESET  - a fits-on-a-page rule set: our Huber-regressed piece values
           (134/402/439/600/1031 cp) + Michniewski-style piece-square
           principles ("knights to the center", "rooks to the 7th", ...) with
           depth-2 alpha-beta + capture quiescence. Fully human-readable.
EVALONLY - Stockfish at Limit(depth=1): the NNUE evaluation used as a pure
           policy (one ply, no real search) - the SF analogue of
           "Leela policy-only".

Each plays 40 games vs the 32- and 64-node ladder rungs.
Output: results/ruleset_results.csv (same schema as ladder_results.csv).
"""
import csv
import random
import sys
import time

import chess
import chess.engine

from node_ladder import SF, MAX_PLIES, make_opening, new_engine

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 40
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/ruleset_results.csv"

# Our reverse-engineered (Huber) piece values, in centipawns
VAL = {chess.PAWN: 134, chess.KNIGHT: 402, chess.BISHOP: 439,
       chess.ROOK: 600, chess.QUEEN: 1031}

# Piece-square principles (white POV, a1..h1 first), Michniewski-style:
# pawns advance & hold the center; knights/bishops to the center; rooks to
# the 7th and center files; king hides (middlegame). Values in cp.
PST = {
    chess.PAWN: [0,0,0,0,0,0,0,0, 5,10,10,-20,-20,10,10,5, 5,-5,-10,0,0,-10,-5,5,
                 0,0,0,20,20,0,0,0, 5,5,10,25,25,10,5,5, 10,10,20,30,30,20,10,10,
                 50,50,50,50,50,50,50,50, 0,0,0,0,0,0,0,0],
    chess.KNIGHT: [-50,-40,-30,-30,-30,-30,-40,-50, -40,-20,0,5,5,0,-20,-40,
                   -30,5,10,15,15,10,5,-30, -30,0,15,20,20,15,0,-30,
                   -30,5,15,20,20,15,5,-30, -30,0,10,15,15,10,0,-30,
                   -40,-20,0,0,0,0,-20,-40, -50,-40,-30,-30,-30,-30,-40,-50],
    chess.BISHOP: [-20,-10,-10,-10,-10,-10,-10,-20, -10,5,0,0,0,0,5,-10,
                   -10,10,10,10,10,10,10,-10, -10,0,10,10,10,10,0,-10,
                   -10,5,5,10,10,5,5,-10, -10,0,5,10,10,5,0,-10,
                   -10,0,0,0,0,0,0,-10, -20,-10,-10,-10,-10,-10,-10,-20],
    chess.ROOK: [0,0,0,5,5,0,0,0, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5,
                 -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5,
                 5,10,10,10,10,10,10,5, 0,0,0,0,0,0,0,0],
    chess.QUEEN: [-20,-10,-10,-5,-5,-10,-10,-20, -10,0,5,0,0,0,0,-10,
                  -10,5,5,5,5,5,0,-10, 0,0,5,5,5,5,0,-5,
                  -5,0,5,5,5,5,0,-5, -10,0,5,5,5,5,0,-10,
                  -10,0,0,0,0,0,0,-10, -20,-10,-10,-5,-5,-10,-10,-20],
    chess.KING: [20,30,10,0,0,10,30,20, 20,20,0,0,0,0,20,20,
                 -10,-20,-20,-20,-20,-20,-20,-10, -20,-30,-30,-40,-40,-30,-30,-20,
                 -30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30,
                 -30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30],
}


def ruleset_score(board: chess.Board) -> int:
    s = 0
    for sq, pc in board.piece_map().items():
        v = VAL.get(pc.piece_type, 0)
        t = PST[pc.piece_type][sq if pc.color == chess.WHITE else chess.square_mirror(sq)]
        s += (v + t) if pc.color == chess.WHITE else -(v + t)
    return s if board.turn == chess.WHITE else -s


def qsearch(board, alpha, beta, depth=4):
    stand = ruleset_score(board)
    if stand >= beta:
        return beta
    alpha = max(alpha, stand)
    if depth == 0:
        return alpha
    for mv in board.legal_moves:
        if not board.is_capture(mv):
            continue
        board.push(mv)
        sc = -qsearch(board, -beta, -alpha, depth - 1)
        board.pop()
        if sc >= beta:
            return beta
        alpha = max(alpha, sc)
    return alpha


def ruleset_move(board, rng):
    def ab(depth, alpha, beta):
        if board.is_checkmate():
            return -100000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        if depth == 0:
            return qsearch(board, alpha, beta)
        best = -10**6
        for mv in board.legal_moves:
            board.push(mv)
            sc = -ab(depth - 1, -beta, -alpha)
            board.pop()
            best = max(best, sc)
            alpha = max(alpha, sc)
            if alpha >= beta:
                break
        return best

    best, moves = -10**9, []
    for mv in board.legal_moves:
        board.push(mv)
        sc = -ab(1, -10**9, 10**9)
        board.pop()
        if sc > best:
            best, moves = sc, [mv]
        elif sc == best:
            moves.append(mv)
    return rng.choice(moves)


def play(board, white, black, eng, rng):
    board = board.copy()
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        who = white if board.turn == chess.WHITE else black
        if who == "RULESET":
            mv = ruleset_move(board, rng)
        elif who == "EVALONLY":
            mv = eng.play(board, chess.engine.Limit(depth=1)).move
        else:
            mv = eng.play(board, chess.engine.Limit(nodes=who)).move
        board.push(mv)
    if board.ply() >= MAX_PLIES and not board.is_game_over(claim_draw=True):
        return 0.5
    return {"1-0": 1.0, "0-1": 0.0}.get(board.result(claim_draw=True), 0.5)


def main():
    rng = random.Random(99)
    eng, probe = new_engine(), new_engine()
    pairs = [("RULESET", 32), ("RULESET", 64), ("EVALONLY", 32), ("EVALONLY", 64)]
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["low", "high", "game", "low_is_white", "score_low"])
        for low, high in pairs:
            pts, t0 = 0.0, time.time()
            for g in range(GAMES // 2):
                opening = make_opening(rng, probe)
                for low_white in (True, False):
                    s = play(opening, low if low_white else high,
                             high if low_white else low, eng, rng)
                    sl = s if low_white else 1.0 - s
                    pts += sl
                    w.writerow([low, high, g, int(low_white), sl])
                    fh.flush()
            print(f"{low} vs {high}: {pts}/{GAMES} ({pts/GAMES:.3f}) "
                  f"in {time.time()-t0:.0f}s", flush=True)
    eng.quit(); probe.quit()
    print("DONE")


if __name__ == "__main__":
    main()
