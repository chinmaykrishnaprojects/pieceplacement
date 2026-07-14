"""Rudimentary policy-guided search on the chess-GPT — climbing the frontier.

The char-LM plays argmax policy with zero search (one forward pass). We add the
cheapest possible search on top and measure the Elo it buys, so the interpretable
model moves along the strength axis with a countable node budget:

  depth 0  : pure policy argmax (baseline, 1 forward pass)
  depth d  : negamax to depth d, but ONLY over the top-k policy moves at each
             node (policy = move ordering + hard pruning, the interpretable part
             — you can print the few candidate lines it actually considered).
             Leaf eval = material + capture quiescence (fast, legible), OR the
             Maia value head if available (still interpretable: a WDL number).

This is the "add the most rudimentary search" idea, and the top-k policy pruning
keeps it human-followable (a handful of candidate lines, like the SF `Explain`
patch). We report nodes/move so it lands on the strength-per-node curve.
"""
import sys

import chess

sys.path.insert(0, "scripts")
from chessgpt_local import ChessGPT, board_to_pgn_prefix

MAT = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
       chess.ROOK: 500, chess.QUEEN: 900}


def material_eval(board):
    s = 0
    for pt, v in MAT.items():
        s += v * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
    return s if board.turn == chess.WHITE else -s


class SearchGPT:
    """chess-GPT policy + top-k negamax. node_count tracks leaf+internal evals."""

    def __init__(self, path, topk=3, depth=2, leaf="material"):
        self.gpt = ChessGPT(path)
        self.topk = topk
        self.depth = depth
        self.leaf = leaf
        self.nodes = 0

    def _policy_moves(self, board, pgn):
        pol = self.gpt.policy(board, pgn_prefix=pgn)
        ranked = sorted(pol, key=pol.get, reverse=True)
        return ranked[:self.topk], pol

    def _qeval(self, board, alpha, beta, qd=3):
        self.nodes += 1
        stand = material_eval(board)
        if stand >= beta:
            return beta
        if qd == 0:
            return max(stand, alpha)
        alpha = max(alpha, stand)
        for mv in board.legal_moves:
            if not board.is_capture(mv):
                continue
            board.push(mv)
            sc = -self._qeval(board, -beta, -alpha, qd - 1)
            board.pop()
            if sc >= beta:
                return beta
            alpha = max(alpha, sc)
        return alpha

    def _negamax(self, board, pgn, depth, alpha, beta):
        if board.is_game_over(claim_draw=True):
            r = board.result(claim_draw=True)
            if r == "1/2-1/2":
                return 0
            return -100000  # side to move lost/checkmated
        if depth == 0:
            return self._qeval(board, alpha, beta)
        moves, _ = self._policy_moves(board, pgn)
        best = -10 ** 9
        for mv in moves:
            san = board.san(mv)
            nxt_pgn = _extend_pgn(pgn, board, san)
            board.push(mv)
            sc = -self._negamax(board, nxt_pgn, depth - 1, -beta, -alpha)
            board.pop()
            if sc > best:
                best = sc
            alpha = max(alpha, sc)
            if alpha >= beta:
                break
        return best

    def play(self, board, pgn=None):
        pgn = pgn if pgn is not None else board_to_pgn_prefix(board)
        if self.depth == 0:
            return self.gpt.play(board, pgn_prefix=pgn)
        moves, pol = self._policy_moves(board, pgn)
        best_mv, best = moves[0], -10 ** 9
        for mv in moves:
            san = board.san(mv)
            board.push(mv)
            sc = -self._negamax(board, _extend_pgn(pgn, board, san),
                                self.depth - 1, -10 ** 9, 10 ** 9)
            board.pop()
            if sc > best:
                best, best_mv = sc, mv
        return best_mv


def _extend_pgn(pgn, board, san):
    if board.turn == chess.WHITE:
        return pgn + f"{board.fullmove_number}.{san} "
    return pgn + f"{san} "


if __name__ == "__main__":
    eng = SearchGPT("models/lichess_16layers_ckpt_no_optimizer.pt", topk=3, depth=2)
    b = chess.Board()
    for u in ["e2e4", "c7c5", "g1f3"]:
        b.push_uci(u)
    mv = eng.play(b)
    print(f"depth-2 top-3 search plays {b.san(mv)}; nodes={eng.nodes}")
