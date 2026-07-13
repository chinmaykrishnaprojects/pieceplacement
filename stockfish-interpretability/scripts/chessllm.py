"""Client for the user's chess-LLM web backend (chessllmweb.vercel.app).

Reverse-engineered from the site bundle:
  POST {BASE}/analyze  {fen, pgn, temperature, model} -> {probabilities:[...]}
  each entry: {uci, san, from_square, to_square, probability}
  model in {"lichess", "stockfish"} (two separately trained 50M transformers).

The model is a character-level autoregressive transformer over PGN (Karvonen
style); it takes the game so far and predicts the next move. It has NO search
(1 forward pass) and returns a probability distribution over legal moves.

Exposes ChessLLM.policy(board, pgn) -> {chess.Move: prob} and .play(...).
"""
import time

import chess
import requests

BASE = "https://chessllm-h7kaigijia-uc.a.run.app/api"


class ChessLLM:
    def __init__(self, model="lichess", temperature=0.3, base=BASE, timeout=40):
        self.model = model
        self.temperature = temperature
        self.base = base
        self.timeout = timeout
        self.sess = requests.Session()

    def _analyze(self, fen, pgn):
        for attempt in range(4):
            try:
                r = self.sess.post(
                    f"{self.base}/analyze",
                    json={"fen": fen, "pgn": pgn,
                          "temperature": self.temperature, "model": self.model},
                    timeout=self.timeout)
                r.raise_for_status()
                return r.json().get("probabilities", [])
            except Exception:
                if attempt == 3:
                    return []
                time.sleep(2 ** attempt)
        return []

    def policy(self, board, pgn=""):
        """Return {legal chess.Move: probability} from the model."""
        probs = self._analyze(board.fen(), pgn)
        out = {}
        legal = {m.uci(): m for m in board.legal_moves}
        for e in probs:
            mv = legal.get(e.get("uci", ""))
            if mv is not None:
                out[mv] = float(e.get("probability", 0.0))
        return out

    def play(self, board, pgn="", rng=None, sample=False):
        pol = self.policy(board, pgn)
        if not pol:
            return None
        if not sample:
            return max(pol, key=pol.get)
        import numpy as np
        moves = list(pol)
        p = np.array([pol[m] for m in moves])
        p = p / p.sum()
        rng = rng or np.random
        return moves[rng.choice(len(moves), p=p)]


if __name__ == "__main__":
    for model in ("lichess", "stockfish"):
        eng = ChessLLM(model=model)
        b = chess.Board()
        pol = eng.policy(b)
        top = sorted(pol.items(), key=lambda kv: -kv[1])[:4]
        print(f"{model:10s} startpos:", [(b.san(m), round(p, 3)) for m, p in top])
        b2 = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")
        pol2 = eng.policy(b2, "1. e4 e5 2. Nf3 Nc6 3. Bc4")
        top2 = sorted(pol2.items(), key=lambda kv: -kv[1])[:4]
        print(f"{model:10s} italian: ", [(b2.san(m), round(p, 3)) for m, p in top2])
