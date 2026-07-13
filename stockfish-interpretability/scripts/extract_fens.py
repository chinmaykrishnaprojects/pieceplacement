"""Sample quiet positions WITH FEN + the human move actually played.

Reuses the streaming approach of extract_lichess.py; keeps every 4th quiet
(ply>=12, not check, not capture) position of evaluated games, storing FEN,
the move the human then played (UCI), mover's Elo, and the eval.
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
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/fen_moves.csv.gz"


def main():
    resp = urllib.request.urlopen(
        urllib.request.Request(URL, headers={"User-Agent": "research-script"}), timeout=60)
    text = io.TextIOWrapper(
        zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(resp),
        encoding="utf-8", errors="replace")
    kept = rows = 0
    with gzip.open(OUT, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["game_id", "mover_elo", "ply", "fen", "human_move", "eval_cp",
                    "pgn_prefix"])
        while kept < TARGET:
            game = chess.pgn.read_game(text)
            if game is None:
                break
            node = game.next()
            if node is None or node.eval() is None:
                continue
            h = game.headers
            gid = h.get("Site", "").rsplit("/", 1)[-1]
            board = game.board()
            node, quiet_idx, any_row = game, 0, False
            pgn = ";"  # Karvonen-format running prefix ";1.e4 e5 2.Nf3 ..."
            while node.variations:
                nxt = node.variation(0)
                move = nxt.move
                san = board.san(move)
                # decide on the PRE-move position (board), like a player would
                if (board.ply() >= 12 and not board.is_check()
                        and not board.is_capture(move)  # human chose a quiet move
                        and node.eval() is not None):
                    if quiet_idx % 4 == 0:
                        elo = h.get("WhiteElo" if board.turn else "BlackElo", "0")
                        ev = node.eval().white()
                        cp = 2000 if ev.is_mate() and ev.mate() > 0 else \
                             -2000 if ev.is_mate() else max(-2000, min(2000, ev.score()))
                        w.writerow([gid, elo, board.ply(), board.fen(),
                                    move.uci(), cp, pgn])
                        rows += 1
                        any_row = True
                    quiet_idx += 1
                # advance the Karvonen prefix
                if board.turn == chess.WHITE:
                    pgn += f"{board.fullmove_number}.{san} "
                else:
                    pgn += f"{san} "
                board.push(move)
                node = nxt
            if any_row:
                kept += 1
                if kept % 500 == 0:
                    print(f"games {kept}, rows {rows}", flush=True)
    print(f"DONE games={kept} rows={rows}")


if __name__ == "__main__":
    main()
