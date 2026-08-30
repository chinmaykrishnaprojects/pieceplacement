"""BASELINE reproducing the user's deployed app settings: T=0.2, min_p=0.1.

One LM call, no search, no tactical filter — the model's own distribution,
sampled the way the live app samples it:

  temperature 0.2 : p' ∝ p**(1/T) = p**5, then renormalise. LOW temperature =
                    SHARPER than the raw distribution (it concentrates mass on
                    the model's favourites), not softer.
  min_p 0.1       : keep only moves with p' >= 0.1 * max(p'), then sample.

Both steps are pure arithmetic on the returned distribution, so they cost no
budget. This is the correct reference point for "how much does the tactical
filter add to the app as actually deployed".
"""
import random

TEMP = 0.2
MIN_P = 0.1


class Player:
    def __init__(self, oracle, budget, seed=12345):
        self.oracle = oracle
        self.budget = budget
        self.rng = random.Random(seed)

    def play(self, board, pgn):
        pol = self.oracle.policy(board, pgn)
        if not pol:
            return next(iter(board.legal_moves), None)
        # temperature: p ** (1/T), normalised
        inv = 1.0 / TEMP
        sharp = {m: (p ** inv if p > 0 else 0.0) for m, p in pol.items()}
        tot = sum(sharp.values())
        if tot <= 0:
            return max(pol, key=pol.get)
        sharp = {m: v / tot for m, v in sharp.items()}
        # min_p: keep moves within MIN_P of the best
        cut = MIN_P * max(sharp.values())
        kept = {m: v for m, v in sharp.items() if v >= cut}
        moves = list(kept)
        weights = [kept[m] for m in moves]
        total = sum(weights)
        r = self.rng.random() * total
        acc = 0.0
        for m, w in zip(moves, weights):
            acc += w
            if r <= acc:
                return m
        return moves[-1]


def create_player(oracle, budget):
    return Player(oracle, budget)
