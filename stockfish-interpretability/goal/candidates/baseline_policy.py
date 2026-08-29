"""BASELINE (control): pure chess-GPT policy argmax, one LM call, no search.

This is the frontier point we are trying to push outward: ~1212 Elo at budget=1.
Every other candidate must beat this at equal or comparable budget to count.
"""


class Player:
    def __init__(self, oracle, budget):
        self.oracle = oracle
        self.budget = budget

    def play(self, board, pgn):
        pol = self.oracle.policy(board, pgn)
        if not pol:
            return next(iter(board.legal_moves), None)
        return max(pol, key=pol.get)


def create_player(oracle, budget):
    return Player(oracle, budget)
