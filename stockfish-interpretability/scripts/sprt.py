"""Fishtest-style SPRT match harness.

Plays game pairs (shared random balanced opening, colors swapped) between a
base and a test binary at fixed nodes/move, accumulates the pentanomial pair
distribution, and after each pair computes the GSPRT log-likelihood ratio for
  H0: elo = elo0   vs   H1: elo = elo1
using the normal approximation (Van den Bergh / fishtest practice):
  LLR = N * (mean - (s0+s1)/2) * (s1 - s0) / var
with s = 1/(1+10^(-elo/400)) and var from the empirical pair distribution.
Stops at +log((1-b)/a) (accept H1) or -log((1-a)/b) (accept H0), a=b=0.05.

Usage: sprt.py BASE TEST NODES ELO0 ELO1 MAX_PAIRS OUT.csv
"""
import csv
import math
import random
import sys

import chess
import chess.engine

BASE, TEST = sys.argv[1], sys.argv[2]
NODES = int(sys.argv[3])
ELO0, ELO1 = float(sys.argv[4]), float(sys.argv[5])
MAX_PAIRS = int(sys.argv[6])
OUT = sys.argv[7]
ALPHA = BETA = 0.05
LA = math.log((1 - BETA) / ALPHA)
LB = -math.log((1 - ALPHA) / BETA)
MAX_PLIES = 220
OPENING_PLIES = 6


def elo_to_score(elo):
    return 1.0 / (1.0 + 10 ** (-elo / 400.0))


def gsprt_llr(pair_counts):
    """pair_counts: dict pair_score(0,.5,1,1.5,2) -> n. Scores are test POV per pair (max 2)."""
    N = sum(pair_counts.values())
    if N < 2:
        return 0.0
    # per-game scale: pair score / 2
    xs, ns = zip(*[(k / 2.0, v) for k, v in pair_counts.items()])
    mean = sum(x * n for x, n in zip(xs, ns)) / N
    var = sum(n * (x - mean) ** 2 for x, n in zip(xs, ns)) / N
    if var <= 1e-9:
        return 0.0
    s0, s1 = elo_to_score(ELO0), elo_to_score(ELO1)
    return N * (mean - (s0 + s1) / 2) * (s1 - s0) / var


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


def play(board, ew, eb):
    board = board.copy()
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        eng = ew if board.turn == chess.WHITE else eb
        board.push(eng.play(board, chess.engine.Limit(nodes=NODES)).move)
    if board.ply() >= MAX_PLIES and not board.is_game_over(claim_draw=True):
        return 0.5
    return {"1-0": 1.0, "0-1": 0.0}.get(board.result(claim_draw=True), 0.5)


def main():
    rng = random.Random(2026)
    eng = {}
    for name, path in (("base", BASE), ("test", TEST)):
        e = chess.engine.SimpleEngine.popen_uci(path)
        e.configure({"Threads": 1, "Hash": 16})
        eng[name] = e
    probe = chess.engine.SimpleEngine.popen_uci(BASE)
    probe.configure({"Threads": 1, "Hash": 16})

    pair_counts = {0.0: 0, 0.5: 0, 1.0: 0, 1.5: 0, 2.0: 0}
    verdict = "inconclusive (max pairs)"
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "pair_score_test", "llr"])
        for p in range(MAX_PAIRS):
            opening = make_opening(rng, probe)
            s1 = play(opening, eng["test"], eng["base"])          # test is white
            s2 = 1.0 - play(opening, eng["base"], eng["test"])    # test is black
            ps = s1 + s2
            pair_counts[ps] += 1
            llr = gsprt_llr(pair_counts)
            w.writerow([p, ps, round(llr, 4)])
            fh.flush()
            if (p + 1) % 10 == 0:
                n = 2 * (p + 1)
                sc = sum(k * v for k, v in pair_counts.items()) / n
                print(f"pairs {p+1}: score {sc:.3f} LLR {llr:+.2f} "
                      f"(bounds [{LB:.2f}, {LA:.2f}])", flush=True)
            if llr >= LA:
                verdict = "H1 accepted"
                break
            if llr <= LB:
                verdict = "H0 accepted"
                break
    n = 2 * sum(pair_counts.values())
    sc = sum(k * v for k, v in pair_counts.items()) / n if n else 0.5
    print(f"VERDICT: {verdict} after {n} games; score {sc:.3f}; "
          f"pentanomial {pair_counts}; H0 elo={ELO0} H1 elo={ELO1}")
    for e in eng.values():
        e.quit()
    probe.quit()


if __name__ == "__main__":
    main()
