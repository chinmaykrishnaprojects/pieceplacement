"""chess2vec — a learned position embedding space over lichess.

Word2vec's idea (a word is known by the company it keeps) applied to chess: a
POSITION is known by the positions that follow it in real games. We train a
shared board ENCODER f(board)->R^d with skip-gram negative sampling: f(P) should
be close to f(P_future) (same game, K plies later) and far from random positions.
Because the encoder is shared (not a lookup table), it embeds ANY FEN, including
unseen ones.

Board encoding is COLOR-AGNOSTIC: always oriented to the side-to-move's POV
(board mirrored + colors swapped when Black is to move), so the space captures
"who is to move and how they stand" rather than absolute color.

Also trains a DECODER g(emb)->board to test embedding->FEN reconstruction.

Encoding: 12 planes (6 piece types x own/enemy) x 64 = 768 dims, side-to-move POV.
"""
import sys

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PT = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
DIM = 768


def board_vec(board: chess.Board) -> np.ndarray:
    """768-dim, side-to-move POV (color-agnostic). Planes: own P N B R Q K,
    then enemy P N B R Q K. Squares from stm's perspective."""
    v = np.zeros((12, 64), dtype=np.float32)
    stm = board.turn
    for pi, pt in enumerate(PT):
        for color in (stm, not stm):
            plane = pi + (0 if color == stm else 6)
            for sq in board.pieces(pt, color):
                # orient to stm POV: if black to move, mirror vertically
                s = sq if stm == chess.WHITE else chess.square_mirror(sq)
                v[plane, s] = 1.0
    return v.reshape(-1)


class Encoder(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM, 512), nn.ReLU(),
            nn.Linear(512, 384), nn.ReLU(),
            nn.Linear(384, d))
        self.d = d

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)  # unit vectors → cosine geometry


class Decoder(nn.Module):
    """emb -> board logits (12x64), for embedding->FEN reconstruction."""
    def __init__(self, d=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 384), nn.ReLU(),
            nn.Linear(384, 512), nn.ReLU(),
            nn.Linear(512, DIM))

    def forward(self, z):
        return self.net(z)


def vec_to_board(logits: np.ndarray, stm_white=True, threshold=0.5) -> chess.Board:
    """Decode a 768 logit vector back to a Board (best-effort, stm POV)."""
    planes = logits.reshape(12, 64)
    board = chess.Board.empty()
    for plane in range(12):
        pt = PT[plane % 6]
        own = plane < 6
        for s in range(64):
            if 1 / (1 + np.exp(-planes[plane, s])) > threshold:
                sq = s if stm_white else chess.square_mirror(s)
                color = (chess.WHITE if own else chess.BLACK) if stm_white \
                    else (chess.BLACK if own else chess.WHITE)
                if board.piece_at(sq) is None:
                    board.set_piece_at(sq, chess.Piece(pt, color))
    return board


if __name__ == "__main__":
    # quick self-test of the encoding
    b = chess.Board()
    print("startpos vec sum:", board_vec(b).sum(), "(should be 32 pieces)")
    b.push_uci("e2e4")
    print("after e4, black to move, vec sum:", board_vec(b).sum())
    enc = Encoder()
    z = enc(torch.tensor(board_vec(b))[None])
    print("emb shape:", z.shape, "norm:", float(z.norm()))
