"""Fast correctness check for a candidate — no model, no Stockfish, ~1 second.

Verifies the three things that cause silent forfeits in the real arena:
  1. returns a LEGAL move from every position tested
  2. never exceeds the per-move LM budget
  3. never crashes (including in check, near-mate, and endgame positions)

Uses a mock oracle returning a plausible random policy, so it is fast and
deterministic. PASSING THIS DOES NOT MEAN THE CANDIDATE IS STRONG — strength is
only measured by arena.py against Stockfish. This just stops you wasting a
tournament slot on a candidate that forfeits.

    python selftest.py candidates/mine.py --budget 4
"""
import argparse
import importlib.util
import random
import sys

import chess

from arena import scan_source, BudgetExhausted


class MockOracle:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.quota = 0
        self.used_this_move = 0
        self.total_calls = 0
        self.peak = 0

    def new_move(self, budget):
        self.quota = budget
        self.used_this_move = 0

    def policy(self, board, pgn_prefix=""):
        if self.used_this_move >= self.quota:
            raise BudgetExhausted(f"budget {self.quota} exhausted")
        self.used_this_move += 1
        self.total_calls += 1
        self.peak = max(self.peak, self.used_this_move)
        moves = list(board.legal_moves)
        if not moves:
            return {}
        w = [self.rng.random() for _ in moves]
        s = sum(w)
        return {m: x / s for m, x in zip(moves, w)}


POSITIONS = [
    (chess.STARTING_FEN, "startpos"),
    ("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", "italian"),
    ("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7", "middlegame"),
    ("8/5pk1/6p1/7p/7P/6P1/5PK1/R7 w - - 0 1", "rook endgame"),
    ("r3k2r/ppp2ppp/2n5/1B6/1b6/2N5/PPP2PPP/R3K2R w KQkq - 6 10", "castling rights"),
    ("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1", "in check (must respond)"),
    ("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1", "K+P vs K"),
    ("8/P7/8/8/8/8/8/K6k w - - 0 1", "promotion available"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--budget", type=int, default=4)
    a = ap.parse_args()

    src = open(a.candidate).read()
    hit = scan_source(src)
    if hit:
        print(f"FAIL: banned construct ({hit}) — candidate would be rejected")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("cand", a.candidate)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    oracle = MockOracle()
    ok = True
    for fen, name in POSITIONS:
        board = chess.Board(fen)
        if board.is_game_over():
            continue
        player = mod.create_player(oracle, a.budget)
        oracle.new_move(a.budget)
        try:
            mv = player.play(board, ";1.e4 e5 2.Nf3 ")
        except BudgetExhausted as e:
            print(f"FAIL [{name}]: exceeded budget ({e}) -> forfeit in arena")
            ok = False
            continue
        except Exception as e:  # noqa: BLE001
            print(f"FAIL [{name}]: crashed {type(e).__name__}: {e}")
            ok = False
            continue
        if mv is None or mv not in board.legal_moves:
            print(f"FAIL [{name}]: illegal/None move {mv} -> forfeit in arena")
            ok = False
        else:
            print(f"  ok [{name}]: {board.san(mv)}  "
                  f"({oracle.used_this_move}/{a.budget} LM calls)")

    print(f"\npeak LM calls in one move: {oracle.peak}/{a.budget}")

    # --- persistence test -------------------------------------------------
    # The arena creates ONE player per GAME and calls .play() for every move.
    # A candidate that tracks its own call counter and forgets to reset it per
    # move will starve itself after move 1 — and the per-position loop above
    # cannot catch that, because it builds a fresh player each time.
    print("\npersistence (one player, full game — catches non-reset state):")
    player = mod.create_player(oracle, a.budget)
    board = chess.Board()
    rng = random.Random(3)
    moves_ok = 0
    for ply in range(60):
        if board.is_game_over():
            break
        if ply % 2 == 0:
            oracle.new_move(a.budget)
            try:
                mv = player.play(board, ";1.e4 e5 ")
            except BudgetExhausted as e:
                print(f"  FAIL at ply {ply}: budget overrun ({e})")
                ok = False
                break
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL at ply {ply}: {type(e).__name__}: {e}")
                ok = False
                break
            if mv is None or mv not in board.legal_moves:
                print(f"  FAIL at ply {ply}: illegal/None move {mv}")
                ok = False
                break
            moves_ok += 1
        else:
            mv = rng.choice(list(board.legal_moves))
        board.push(mv)
    else:
        pass
    if ok:
        print(f"  ok: {moves_ok} consecutive moves from one player instance")

    print("\nPASS — safe to submit" if ok else "\nFAIL — fix before submitting")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
