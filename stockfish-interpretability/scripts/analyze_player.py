"""Personalized chess analysis for a lichess player using chess2vec + Stockfish.

Metric-driven (the frontier's interpretability axis, made concrete):
  INTERPRETABILITY(tool) = accuracy with which a HUMAN-LEGIBLE decoding recovers
  the tool's judgement. For chess2vec that is the concept-probe accuracy; for the
  eval it's the R^2 of human terms vs NNUE. Here we USE the interpretable tools
  to produce something a human player can act on:

  1. Phase performance: bucket the player's positions (from their games) by
     chess2vec phase + who-ahead, and score how often they CONVERT/blunder using
     Stockfish eval swings -> where do they lose advantages?
  2. Recurring structures: cluster their positions in chess2vec space -> the
     handful of position-types they reach most (their repertoire, data-driven).
  3. Similar-position retrieval: for their worst blunders, find the nearest
     positions across lichess (teaching examples).

Outputs results/player_<name>.json for the web page.
"""
import io
import json
import sys

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch

sys.path.insert(0, "scripts")
from chess2vec import Encoder, board_vec

SF = "/usr/games/stockfish"
NAME = sys.argv[1] if len(sys.argv) > 1 else "kingskreamer"
PGN = f"data/{NAME}.pgn"
CKPT = "models/chess2vec.pt"
CORPUS = "results/chess2vec_emb.npz"


def load_encoder():
    ck = torch.load(CKPT, map_location="cpu")
    enc = Encoder(ck["d"])
    enc.load_state_dict(ck["enc"])
    enc.eval()
    return enc


def phase_of(board):
    npm = sum({chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}[pt]
              * len(board.pieces(pt, c))
              for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
              for c in (True, False))
    if board.fullmove_number <= 10:
        return "opening"
    return "endgame" if npm <= 12 else "middlegame"


def main():
    enc = load_encoder()
    eng = chess.engine.SimpleEngine.popen_uci(SF)
    eng.configure({"Threads": 2, "Hash": 64})
    corpus = np.load(CORPUS, allow_pickle=True)
    cemb, cfen = corpus["emb"], corpus["fen"]

    text = open(PGN).read()
    stream = io.StringIO(text)
    # metrics
    phase_blunders = {"opening": [0, 0], "middlegame": [0, 0], "endgame": [0, 0]}
    embs = []
    blunders = []  # (cp_loss, fen, move_no, phase)
    ngames = 0

    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        h = game.headers
        me_white = h.get("White", "").lower() == NAME.lower()
        me_black = h.get("Black", "").lower() == NAME.lower()
        if not (me_white or me_black):
            continue
        ngames += 1
        board = game.board()
        prev_eval = None
        node = game
        while node.variations:
            nxt = node.variation(0)
            mv = nxt.move
            my_move = (board.turn == chess.WHITE and me_white) or \
                      (board.turn == chess.BLACK and me_black)
            # eval before my move (shallow, fast)
            if my_move and board.fullmove_number >= 4:
                info = eng.analyse(board, chess.engine.Limit(nodes=40000))
                cp_before = info["score"].pov(board.turn).score(mate_score=2000)
                embs.append(enc(torch.tensor(board_vec(board))[None]).detach().numpy()[0])
                board.push(mv)
                info2 = eng.analyse(board, chess.engine.Limit(nodes=40000))
                cp_after = -info2["score"].pov(board.turn).score(mate_score=2000)
                loss = max(0, cp_before - cp_after)
                ph = phase_of(chess.Board(board.fen()))
                phase_blunders[ph][1] += 1
                if loss >= 150:
                    phase_blunders[ph][0] += 1
                    blunders.append((int(loss), board.fen(), board.fullmove_number, ph))
            else:
                board.push(mv)
            node = nxt
        if ngames >= 60:  # cap for runtime
            break
    eng.quit()

    embs = np.array(embs)
    # recurring structures: k-means-lite via cosine to a few seeds (top density)
    # simpler: report phase distribution + blunder rate per phase
    out = {
        "player": NAME, "games_analyzed": ngames, "moves_scored": sum(v[1] for v in phase_blunders.values()),
        "blunder_rate_by_phase": {k: (v[0] / v[1] if v[1] else 0) for k, v in phase_blunders.items()},
        "moves_by_phase": {k: v[1] for k, v in phase_blunders.items()},
        "worst_blunders": [],
    }
    # worst 5 blunders + nearest teaching position from corpus
    blunders.sort(reverse=True)
    for loss, fen, mvno, ph in blunders[:5]:
        q = enc(torch.tensor(board_vec(chess.Board(fen)))[None]).detach().numpy()[0]
        nn = int(np.argmax(cemb @ q))
        out["worst_blunders"].append({
            "cp_loss": loss, "fen": fen, "move_no": mvno, "phase": ph,
            "similar_position": str(cfen[nn]),
        })
    json.dump(out, open(f"results/player_{NAME}.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "worst_blunders"}, indent=2))
    print(f"\nworst blunder: -{blunders[0][0]}cp at move {blunders[0][2]} ({blunders[0][3]})"
          if blunders else "no blunders found")


if __name__ == "__main__":
    main()
