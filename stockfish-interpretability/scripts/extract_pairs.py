"""Extract (position, later-position) pairs + concept labels for chess2vec.

Streams lichess games and emits, per sampled position:
  fen, fen_ctx (K plies later, same game), and labels used later for probing:
    n_pieces, phase (0 opening/1 middle/2 end), stm_material (side-to-move minus
    opponent, pawns), open/closed proxy (locked central pawns), move_number.
Boards are encoded at train time (color-agnostic: flipped to side-to-move POV).
Output: gzipped CSV of fen pairs + labels.
"""
import csv
import gzip
import io
import sys
import urllib.request

import chess
import chess.pgn
import zstandard

URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst"
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/pairs.csv.gz"
CTX = 6  # plies ahead for the positive-context position

PT = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def features(board):
    npiece = sum(len(board.pieces(pt, c)) for pt in PT + [chess.KING]
                 for c in (True, False))
    mat = {c: sum(VAL[pt] * len(board.pieces(pt, c)) for pt in PT)
           for c in (True, False)}
    stm = board.turn
    stm_mat = mat[stm] - mat[not stm]  # side-to-move material edge
    # phase by non-pawn material remaining
    npm = sum(VAL[pt] * len(board.pieces(pt, c)) for pt in PT[1:] for c in (True, False))
    phase = 0 if board.fullmove_number <= 10 else (2 if npm <= 12 else 1)
    # closed proxy: count of blocked pawns on d/e files + total pawn contacts
    closed = 0
    for c in (True, False):
        for sq in board.pieces(chess.PAWN, c):
            f = chess.square_file(sq)
            if 2 <= f <= 5:
                closed += 1
    return npiece, stm_mat, phase, closed


def main():
    resp = urllib.request.urlopen(
        urllib.request.Request(URL, headers={"User-Agent": "research"}), timeout=60)
    text = io.TextIOWrapper(
        zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(resp),
        encoding="utf-8", errors="replace")
    kept = 0
    with gzip.open(OUT, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fen", "fen_ctx", "n_pieces", "stm_material", "phase", "closed"])
        while kept < TARGET:
            game = chess.pgn.read_game(text)
            if game is None:
                break
            moves = list(game.mainline_moves())
            if len(moves) < 20:
                continue
            board = chess.Board()
            boards = [board.fen()]
            b = chess.Board()
            for m in moves:
                b.push(m)
                boards.append(b.fen())
            # sample ~3 positions per game (mid-game, non-trivial)
            b = chess.Board()
            for i, m in enumerate(moves):
                b.push(m)
                if i >= 8 and i + 1 + CTX < len(boards) and i % 7 == 0:
                    npiece, stm_mat, phase, closed = features(b)
                    w.writerow([b.fen(), boards[i + 1 + CTX], npiece, stm_mat,
                                phase, closed])
                    kept += 1
            if kept % 5000 < 3:
                print(f"pairs {kept}", flush=True)
    print(f"DONE pairs={kept}")


if __name__ == "__main__":
    main()
