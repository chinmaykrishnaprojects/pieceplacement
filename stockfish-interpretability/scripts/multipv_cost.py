"""Measure the Elo cost of making Stockfish interpretable via candidate lines.

An "interpretable Stockfish" that must always surface k candidate moves with
full principal variations (the human-style 'I considered these moves' output)
is just SF with MultiPV=k. At a fixed node budget the search is split across
k lines, so there is a strength price. We measure it head-to-head:

    SF(MultiPV=k, nodes=N)  vs  SF(MultiPV=1, nodes=N)

for k in {2, 3, 5} at N = 16384. The engine still *plays* its top line; it
just also has to resolve the alternatives. Output: results/multipv_results.csv
"""
import csv
import random
import sys
import time

import chess
import chess.engine

SF = "/usr/games/stockfish"
NODES = 16384
KS = [2, 3, 5]
GAMES_PER_K = int(sys.argv[1]) if len(sys.argv) > 1 else 40
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/multipv_results.csv"
MAX_PLIES = 220
OPENING_PLIES = 6


def make_opening(rng, probe):
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


def play(board, eng_w, eng_b):
    board = board.copy()
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        eng, k = eng_w if board.turn == chess.WHITE else eng_b
        if k == 1:
            r = eng.play(board, chess.engine.Limit(nodes=NODES))
            mv = r.move
        else:
            # MultiPV=k search: engine must resolve k candidate lines from the
            # same node budget, then plays the top one (python-chess manages
            # the MultiPV option via analyse()).
            infos = eng.analyse(board, chess.engine.Limit(nodes=NODES), multipv=k)
            mv = infos[0]["pv"][0]
        board.push(mv)
    if board.ply() >= MAX_PLIES and not board.is_game_over(claim_draw=True):
        return 0.5
    return {"1-0": 1.0, "0-1": 0.0}.get(board.result(claim_draw=True), 0.5)


def main():
    rng = random.Random(7)
    probe = chess.engine.SimpleEngine.popen_uci(SF)
    probe.configure({"Threads": 1, "Hash": 16})
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["k", "nodes", "game", "mpv_is_white", "score_mpv"])
        for k in KS:
            mpv = chess.engine.SimpleEngine.popen_uci(SF)
            mpv.configure({"Threads": 1, "Hash": 16})
            ref = chess.engine.SimpleEngine.popen_uci(SF)
            ref.configure({"Threads": 1, "Hash": 16})
            pts, t0 = 0.0, time.time()
            for g in range(GAMES_PER_K // 2):
                opening = make_opening(rng, probe)
                for mpv_white in (True, False):
                    s = play(opening, (mpv, k) if mpv_white else (ref, 1),
                             (ref, 1) if mpv_white else (mpv, k))
                    sm = s if mpv_white else 1.0 - s
                    pts += sm
                    w.writerow([k, NODES, g, int(mpv_white), sm])
                    fh.flush()
            print(f"MultiPV={k}: {pts}/{GAMES_PER_K} ({pts/GAMES_PER_K:.3f}) "
                  f"in {time.time()-t0:.0f}s", flush=True)
            mpv.quit()
            ref.quit()
    probe.quit()
    print("DONE")


if __name__ == "__main__":
    main()
