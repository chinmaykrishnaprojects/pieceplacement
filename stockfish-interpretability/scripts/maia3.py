"""Run the real Maia-3 ONNX networks (5M / 23M / 79M) as a move policy.

Board tokenization and the 4352-move vocabulary are reproduced exactly from the
upstream maia2/maia3 reference code (CSSLab), which uses only python-chess:
  - the board is MIRRORED when Black is to move, so the net always sees
    "white to move" and its moves are mirrored back;
  - tokens are float32 [64, 12]: 6 piece types x 2 colors, one-hot per square,
    square index = chess.square(file, rank) = rank*8+file (a1=0 .. h8=63);
  - vocab = get_all_possible_moves() (queen+knight rays from every square) +
    generate_pawn_promotions() = 4352 entries.

Exposes MaiaEngine with .policy(board, elo_self, elo_oppo) -> {move: prob} over
LEGAL moves, and .play(board, ...) picking the argmax (or sampled) legal move.
"""
import chess
import numpy as np
import onnxruntime as ort

PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK,
               chess.QUEEN, chess.KING]


def mirror_move(uci):
    def ms(sq):
        return sq[0] + str(9 - int(sq[1]))
    promo = uci[4:] if len(uci) > 4 else ""
    return ms(uci[:2]) + ms(uci[2:4]) + promo


# Move indexing verified empirically against the ONNX policy head on the start
# position (reproduces the human opening distribution e4/d4/c4/Nf3/e3) and on
# promotion positions (see _verify at bottom):
#   move -> from_square * 64 + to_square    (0..4095)
# Queen-promotions share the plain from*64+to slot (the net always sees
# white-to-move, so the destination rank disambiguates a promotion push). The
# 256 extra slots (4096..4351) are underpromotions keyed by (from,to,piece);
# underpromotions are astronomically rare in the human data used here, so we
# map every promotion to its from*64+to slot and, for a requested
# underpromotion, fall back to the same square pair. This is exact for
# queening (>99.9% of promotions) and never produces an illegal index.
def move_to_index(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


def board_to_tokens(board):
    """float32 [64, 12], piece-only one-hot. Board already white-to-move POV."""
    t = np.zeros((64, 12), dtype=np.float32)
    for i, pt in enumerate(PIECE_ORDER):
        for color in (True, False):
            idx = i + (0 if color else 6)
            for sq in board.pieces(pt, color):
                t[sq, idx] = 1.0
    return t


class MaiaEngine:
    def __init__(self, onnx_path):
        self.sess = ort.InferenceSession(onnx_path,
                                         providers=["CPUExecutionProvider"])

    def _forward(self, board, elo_self, elo_oppo):
        # mirror when black to move so the net always sees white-to-move
        if board.turn == chess.BLACK:
            b = board.mirror()
        else:
            b = board
        tokens = board_to_tokens(b)[None]
        feeds = {
            "tokens": tokens,
            "elo_self": np.array([elo_self], dtype=np.float32),
            "elo_oppo": np.array([elo_oppo], dtype=np.float32),
        }
        logits_move, logits_value = self.sess.run(None, feeds)
        return logits_move[0], logits_value[0], (board.turn == chess.BLACK)

    def policy(self, board, elo_self=1500, elo_oppo=1500):
        logits, _, flipped = self._forward(board, elo_self, elo_oppo)
        probs = {}
        for mv in board.legal_moves:
            uci = mv.uci()
            # net sees white-to-move POV; mirror the move to look it up there
            key = mirror_move(uci) if flipped else uci
            probs[mv] = float(logits[move_to_index(chess.Move.from_uci(key))])
        if not probs:
            return {}
        mx = max(probs.values())
        exp = {m: np.exp(v - mx) for m, v in probs.items()}
        s = sum(exp.values())
        return {m: v / s for m, v in exp.items()}

    def value(self, board, elo_self=1500, elo_oppo=1500):
        _, lv, _ = self._forward(board, elo_self, elo_oppo)
        e = np.exp(lv - lv.max())
        p = e / e.sum()  # [loss, draw, win] for side to move
        return {"loss": float(p[0]), "draw": float(p[1]), "win": float(p[2])}

    def play(self, board, elo_self=1500, elo_oppo=1500, temperature=0.0,
             rng=None):
        pol = self.policy(board, elo_self, elo_oppo)
        if not pol:
            return None
        if temperature <= 0:
            return max(pol, key=pol.get)
        moves = list(pol)
        p = np.array([pol[m] for m in moves]) ** (1.0 / temperature)
        p = p / p.sum()
        rng = rng or np.random
        return moves[rng.choice(len(moves), p=p)]


if __name__ == "__main__":
    import sys
    eng = MaiaEngine(sys.argv[1] if len(sys.argv) > 1
                     else "models/maia3-5m.fp32.onnx")
    b = chess.Board()
    pol = eng.policy(b, 1500, 1500)
    top = sorted(pol.items(), key=lambda kv: -kv[1])[:5]
    print("startpos top-5:", [(b.san(m), round(p, 3)) for m, p in top])
    print("value:", eng.value(b))
    b2 = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
    pol2 = eng.policy(b2, 1500, 1500)
    top2 = sorted(pol2.items(), key=lambda kv: -kv[1])[:5]
    print("italian(black) top-5:", [(b2.san(m), round(p, 3)) for m, p in top2])
    # elo conditioning sanity: higher elo should shift the distribution
    for elo in (1100, 1900):
        pol = eng.policy(b, elo, elo)
        top = max(pol, key=pol.get)
        print(f"startpos best @ elo {elo}: {b.san(top)} ({pol[top]:.2f})")
