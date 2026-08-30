"""Build a NATURAL opening book from real lichess games.

Why this matters: the gen-1 arena opened with 8 *random* legal plies. Stockfish
does not care, but a PGN language model does — random openings are far outside
its training distribution, so the baseline was measured in positions the model
was never trained to understand. That deflates the LM's apparent strength and
inflates the measured benefit of any tactical patch.

This book uses real human games (same lichess dump the project already sampled),
truncated in the opening and filtered to roughly balanced positions, so both
sides get a fair, in-distribution start.
"""
import gzip
import json
import random
import sys

import chess
import chess.pgn
import io
import pandas as pd

OUT = sys.argv[1] if len(sys.argv) > 1 else "book_natural.json"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
MIN_PLY, MAX_PLY = 8, 14
MAX_CP = 70


def moves_from_prefix(prefix):
    """';1.e4 e5 2.Nf3 ' -> [Move, ...] by replaying the SAN tokens."""
    body = prefix.lstrip(";").strip()
    board = chess.Board()
    out = []
    for tok in body.split():
        if tok and tok[0].isdigit() and "." in tok:
            tok = tok.split(".", 1)[1]
            if not tok:
                continue
        try:
            mv = board.parse_san(tok)
        except Exception:  # noqa: BLE001
            return None
        out.append(mv.uci())
        board.push(mv)
    return out


def main():
    df = pd.read_csv("data/fen_moves_pgn.csv.gz")
    df["eval_cp"] = pd.to_numeric(df.eval_cp, errors="coerce")
    df = df[(df.ply >= MIN_PLY) & (df.ply <= MAX_PLY)
            & (df.eval_cp.abs() <= MAX_CP)]
    df = df.sample(frac=1.0, random_state=5)

    seen = set()
    book = []
    for _, r in df.iterrows():
        mv = moves_from_prefix(str(r.pgn_prefix))
        if not mv or not (MIN_PLY <= len(mv) <= MAX_PLY):
            continue
        b = chess.Board()
        for u in mv:
            b.push(chess.Move.from_uci(u))
        if b.is_game_over():
            continue
        key = b.board_fen()
        if key in seen:
            continue
        seen.add(key)
        book.append(mv)
        if len(book) >= N:
            break

    json.dump(book, open(OUT, "w"))
    print(f"wrote {OUT}: {len(book)} natural openings "
          f"({MIN_PLY}-{MAX_PLY} plies, |eval| <= {MAX_CP}cp)")
    for mv in book[:3]:
        b = chess.Board()
        sans = []
        for u in mv:
            m = chess.Move.from_uci(u)
            sans.append(b.san(m))
            b.push(m)
        print("  ", " ".join(sans))


if __name__ == "__main__":
    main()
