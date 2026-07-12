"""Stream a lichess monthly PGN dump and extract positions from games with [%eval].

For each evaluated game we walk the mainline and record, per position:
  - material counts for both sides (P,N,B,R,Q)
  - side to move, ply, eval in centipawns (white POV; mates clamped)
  - final game result (white POV score 1/0.5/0)
  - castling rights gone flag, phase proxy (total non-pawn material)

Output: gzipped CSV. Stops after collecting N games with evals.
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
TARGET_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/positions.csv.gz"

MATE_CP = 2000  # clamp

RESULT_MAP = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


def material(board: chess.Board):
    vals = {}
    for color, prefix in ((chess.WHITE, "w"), (chess.BLACK, "b")):
        for pt, name in ((chess.PAWN, "p"), (chess.KNIGHT, "n"), (chess.BISHOP, "b"),
                         (chess.ROOK, "r"), (chess.QUEEN, "q")):
            vals[prefix + name] = len(board.pieces(pt, color))
    return vals


def main():
    req = urllib.request.Request(URL, headers={"User-Agent": "research-script"})
    resp = urllib.request.urlopen(req, timeout=60)
    dctx = zstandard.ZstdDecompressor(max_window_size=2**31)
    reader = dctx.stream_reader(resp)
    text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")

    games_kept = 0
    games_seen = 0
    rows = 0

    with gzip.open(OUT, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["game_id", "white_elo", "black_elo", "result", "ply", "stm",
                    "eval_cp", "wp", "wn", "wb", "wr", "wq",
                    "bp", "bn", "bb", "br", "bq", "is_capture", "is_check"])
        while games_kept < TARGET_GAMES:
            game = chess.pgn.read_game(text)
            if game is None:
                break
            games_seen += 1
            h = game.headers
            result = RESULT_MAP.get(h.get("Result", "*"))
            if result is None:
                continue
            # quick check for evals without full walk
            node = game.next()
            if node is None or node.eval() is None:
                continue
            try:
                we = int(h.get("WhiteElo", "0"))
                be = int(h.get("BlackElo", "0"))
            except ValueError:
                we = be = 0
            gid = h.get("Site", "").rsplit("/", 1)[-1]
            board = game.board()
            node = game
            kept_any = False
            while node.variations:
                nxt = node.variation(0)
                move = nxt.move
                is_capture = board.is_capture(move)
                board.push(move)
                ev = nxt.eval()
                if ev is not None:
                    pov = ev.white()
                    if pov.is_mate():
                        cp = MATE_CP if pov.mate() > 0 else -MATE_CP
                    else:
                        cp = max(-MATE_CP, min(MATE_CP, pov.score()))
                    m = material(board)
                    w.writerow([gid, we, be, result, board.ply(),
                                1 if board.turn == chess.WHITE else 0, cp,
                                m["wp"], m["wn"], m["wb"], m["wr"], m["wq"],
                                m["bp"], m["bn"], m["bb"], m["br"], m["bq"],
                                int(is_capture), int(board.is_check())])
                    rows += 1
                    kept_any = True
                node = nxt
            if kept_any:
                games_kept += 1
                if games_kept % 500 == 0:
                    print(f"games kept {games_kept} / seen {games_seen}, rows {rows}",
                          flush=True)

    print(f"DONE kept={games_kept} seen={games_seen} rows={rows}")


if __name__ == "__main__":
    main()
